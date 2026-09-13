//! CPU kernels for the fixed VR inference networks.

use burn_tensor::{TensorData, TensorPrimitive};
use rayon::prelude::*;

use super::T4;

pub(super) struct ChannelNorm {
    pub mean: f32,
    pub divisor: f32,
    pub weight: f32,
    pub bias: f32,
}

/// Keep the original FP32 operation order, but visit each activation only once.
/// Copy-on-write protects U-Net skips and views; an owned convolution output is
/// updated in place, without allocating another full activation tensor.
pub(super) fn normalize(x: T4, parameters: &[ChannelNorm], leaky: bool) -> T4 {
    let [_, channels, height, width] = x.dims();
    assert_eq!(channels, parameters.len());
    let mut x = x.into_primitive().tensor().to_contiguous();
    x.storage_mut::<f32>()
        .par_chunks_mut(height * width)
        .enumerate()
        .for_each(|(plane, values)| {
            let p = &parameters[plane % channels];
            if leaky {
                for value in values {
                    let normalized = (*value - p.mean) / p.divisor * p.weight + p.bias;
                    *value = if normalized < 0.0 {
                        normalized * 0.01
                    } else {
                        normalized
                    };
                }
            } else {
                for value in values {
                    let normalized = (*value - p.mean) / p.divisor * p.weight + p.bias;
                    *value = normalized.max(0.0);
                }
            }
        });
    T4::from_primitive(TensorPrimitive::Float(x))
}

/// Winograd F(4x4, 3x3), stride 1, zero padding 1. A 6x6 transformed
/// patch produces 4x4 outputs using 36 channel products instead of 144.
/// Reference formulation: Lavin & Gray, "Fast Algorithms for Convolutional
/// Neural Networks" (2016), Y = A^T [(G g G^T) .* (B^T d B)] A.
pub(super) struct Winograd3x3 {
    weight: Vec<f32>,
    input_channels: usize,
    output_channels: usize,
}

impl Winograd3x3 {
    pub(super) fn new(weight: T4) -> Self {
        let [output_channels, input_channels, kh, kw] = weight.dims();
        assert_eq!([kh, kw], [3, 3]);
        let weight = weight.into_primitive().tensor().to_contiguous();
        let source = weight.storage::<f32>();
        let mut transformed = vec![0.0_f32; 36 * input_channels * output_channels];
        const G: [[f64; 3]; 6] = [
            [0.25, 0.0, 0.0],
            [-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0],
            [-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0],
            [1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0],
            [1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0],
            [0.0, 0.0, 1.0],
        ];
        // Transform once at load time and release the original dense weights.
        // f64 preparation avoids introducing extra rounding in the fixed filters;
        // activation transforms and all matrix products remain FP32.
        for output in 0..output_channels {
            for input in 0..input_channels {
                let base = (output * input_channels + input) * 9;
                for (row, gr) in G.iter().enumerate() {
                    for (col, gc) in G.iter().enumerate() {
                        let mut value = 0.0;
                        for y in 0..3 {
                            for x in 0..3 {
                                value += gr[y] * f64::from(source[base + y * 3 + x]) * gc[x];
                            }
                        }
                        // Store each plane as [input, output]. This is the
                        // column-major equivalent of [output, input], so GEMM
                        // can consume lhs with row stride 1 and skip its
                        // per-call lhs packing for block-aligned outputs.
                        transformed[((row * 6 + col) * input_channels + input) * output_channels
                            + output] = value as f32;
                    }
                }
            }
        }
        Self {
            weight: transformed,
            input_channels,
            output_channels,
        }
    }

    pub(super) fn forward(&self, x: T4) -> T4 {
        let [batch, channels, height, width] = x.dims();
        assert_eq!(channels, self.input_channels);
        let x = x.into_primitive().tensor().to_contiguous();
        let source = x.storage::<f32>();
        let spatial = height * width;
        let tiles_wide = width.div_ceil(4);
        let tiles_total = height.div_ceil(4) * tiles_wide;
        let mut output = vec![0.0; batch * self.output_channels * spatial];

        // Bound scratch space independently of the audio window length. The
        // 36 products share one batched GEMM dispatch and the caller's Rayon pool.
        const TILE_BATCH: usize = 512;
        let max_lanes = (batch * TILE_BATCH) | 1;
        let mut input_scratch = vec![0.0; channels * 36 * max_lanes];
        let mut product_scratch = vec![0.0; 36 * self.output_channels * max_lanes];
        for start in (0..tiles_total).step_by(TILE_BATCH) {
            let count = (tiles_total - start).min(TILE_BATCH);
            let lanes = batch * count;
            let tile_positions: Vec<_> = (start..start + count)
                .map(|index| {
                    let top = index / tiles_wide * 4;
                    let left = index % tiles_wide * 4;
                    (
                        top as isize - 1,
                        left as isize - 1,
                        top,
                        left,
                        4.min(height - top),
                        4.min(width - left),
                    )
                })
                .collect();
            // An odd pitch keeps the 36 transform planes from repeatedly
            // mapping to the same L1 cache sets (channel counts are powers
            // of two). The extra zero lane is never written to the output.
            let pitch = lanes | 1;
            let input = &mut input_scratch[..channels * 36 * pitch];
            input
                .par_chunks_mut(36 * pitch)
                .enumerate()
                .for_each(|(channel, dest)| {
                    for b in 0..batch {
                        let plane = &source[(b * channels + channel) * spatial
                            ..(b * channels + channel + 1) * spatial];
                        for tile in 0..count {
                            let (top, left, _, _, _, _) = tile_positions[tile];
                            let mut rows = [[0.0; 6]; 6];
                            if top >= 0
                                && left >= 0
                                && top + 6 <= height as isize
                                && left + 6 <= width as isize
                            {
                                for (y, row) in rows.iter_mut().enumerate() {
                                    let base = (top as usize + y) * width + left as usize;
                                    *row =
                                        input_transform(plane[base..base + 6].try_into().unwrap());
                                }
                            } else {
                                for (y, row) in rows.iter_mut().enumerate() {
                                    let sy = top + y as isize;
                                    let mut values = [0.0; 6];
                                    if (0..height as isize).contains(&sy) {
                                        for (col, value) in values.iter_mut().enumerate() {
                                            let sx = left + col as isize;
                                            if (0..width as isize).contains(&sx) {
                                                *value = plane[sy as usize * width + sx as usize];
                                            }
                                        }
                                    }
                                    *row = input_transform(values);
                                }
                            }
                            let lane = b * count + tile;
                            for col in 0..6 {
                                let values = input_transform(std::array::from_fn(|y| rows[y][col]));
                                for row in 0..6 {
                                    dest[(row * 6 + col) * pitch + lane] = values[row];
                                }
                            }
                        }
                    }
                });
            // The transformed input is already in the strided layout expected
            // by gemm. Calling the kernel directly avoids materializing a
            // Burn batched tensor and rediscovering these strides for every
            // tile group; Rayon still owns the 36 independent plane calls.
            let product = &mut product_scratch[..36 * self.output_channels * pitch];
            product
                .par_chunks_mut(self.output_channels * pitch)
                .enumerate()
                .for_each(|(plane, destination)| {
                    let lhs = &self.weight[plane * self.input_channels * self.output_channels
                        ..(plane + 1) * self.input_channels * self.output_channels];
                    let rhs_offset = plane * pitch;
                    gemm_f32_strided(
                        destination,
                        lhs,
                        &input[rhs_offset..],
                        self.output_channels,
                        pitch,
                        self.input_channels,
                        36 * pitch,
                        self.output_channels,
                        1,
                    );
                });
            output
                .par_chunks_mut(spatial)
                .enumerate()
                .for_each(|(plane_index, plane)| {
                    let b = plane_index / self.output_channels;
                    let channel = plane_index % self.output_channels;
                    let product_offsets: [usize; 36] = std::array::from_fn(|product_plane| {
                        (product_plane * self.output_channels + channel) * pitch
                    });
                    for tile in 0..count {
                        let (_, _, top, left, valid_height, valid_width) = tile_positions[tile];
                        let lane = b * count + tile;
                        let mut rows = [[0.0; 4]; 6];
                        for (row, values) in rows.iter_mut().enumerate() {
                            *values = output_transform(std::array::from_fn(|col| {
                                product[product_offsets[row * 6 + col] + lane]
                            }));
                        }
                        for col in 0..valid_width {
                            let values = output_transform(std::array::from_fn(|y| rows[y][col]));
                            for row in 0..valid_height {
                                plane[(top + row) * width + left + col] = values[row];
                            }
                        }
                    }
                });
        }
        T4::from_data(
            TensorData::new(output, [batch, self.output_channels, height, width]),
            &Default::default(),
        )
    }
}

/// Run one independent Winograd plane through the GEMM micro-kernel.
///
/// The three slices are contiguous owners of the storage passed to `gemm`.
/// Their lengths and the maximum strided addresses are checked before the
/// pointer call, and the outer Rayon loop gives each invocation disjoint
/// destination storage. `read_dst = false` is valid because the destination
/// is freshly zero-initialized and beta is ignored by that mode.
#[inline]
#[allow(unsafe_code)]
fn gemm_f32_strided(
    destination: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    columns: usize,
    inner: usize,
    rhs_row_stride: usize,
    lhs_col_stride: usize,
    lhs_row_stride: usize,
) {
    assert!(inner > 0);
    assert_eq!(destination.len(), rows * columns);
    assert_eq!(lhs.len(), rows * inner);
    assert!(rhs.len() >= (inner - 1) * rhs_row_stride + columns);

    // SAFETY: all pointers come from live slices; the assertions above prove
    // every strided read/write stays within those slices. `destination` is a
    // unique mutable slice for this Rayon task, while lhs/rhs are read-only.
    unsafe {
        gemm::gemm(
            rows,
            columns,
            inner,
            destination.as_mut_ptr(),
            1,
            columns as isize,
            false,
            lhs.as_ptr(),
            lhs_col_stride as isize,
            lhs_row_stride as isize,
            rhs.as_ptr(),
            1,
            rhs_row_stride as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::None,
        );
    }
}

#[inline]
fn input_transform(x: [f32; 6]) -> [f32; 6] {
    [
        4.0 * x[0] - 5.0 * x[2] + x[4],
        -4.0 * (x[1] + x[2]) + x[3] + x[4],
        4.0 * (x[1] - x[2]) - x[3] + x[4],
        -2.0 * x[1] - x[2] + 2.0 * x[3] + x[4],
        2.0 * x[1] - x[2] - 2.0 * x[3] + x[4],
        4.0 * x[1] - 5.0 * x[3] + x[5],
    ]
}

#[inline]
fn output_transform(x: [f32; 6]) -> [f32; 4] {
    let sum12 = x[1] + x[2];
    let diff12 = x[1] - x[2];
    let sum34 = x[3] + x[4];
    let diff34 = x[3] - x[4];
    [
        x[0] + sum12 + sum34,
        diff12 + 2.0 * diff34,
        sum12 + 4.0 * sum34,
        diff12 + 8.0 * diff34 + x[5],
    ]
}

/// Bilinear resize with align_corners=true. Coordinate mapping is shared by
/// all channels, avoiding millions of repeated floor/conversion operations.
pub(super) fn resize(x: T4, height: usize, width: usize) -> T4 {
    let [batch, channels, input_height, input_width] = x.dims();
    assert!(input_height > 0 && input_width > 0 && height > 0 && width > 0);
    if [height, width] == [input_height, input_width] {
        return x;
    }
    let axis = |input: usize, output: usize| -> Vec<(usize, usize, f32)> {
        let ratio = if output > 1 {
            (input - 1) as f64 / (output - 1) as f64
        } else {
            0.0
        };
        (0..output)
            .map(|position| {
                let coordinate = position as f64 * ratio;
                let low = coordinate.floor() as usize;
                (
                    low,
                    (low + 1).min(input - 1),
                    (coordinate - low as f64) as f32,
                )
            })
            .collect()
    };
    let xs = axis(input_width, width);
    let ys = axis(input_height, height);
    let x = x.into_primitive().tensor().to_contiguous();
    let input = x.storage::<f32>();
    let mut output = vec![0.0; batch * channels * height * width];
    output
        .par_chunks_mut(height * width)
        .enumerate()
        .for_each(|(plane, dest)| {
            let base = plane * input_height * input_width;
            for (y, &(low_y, high_y, wy)) in ys.iter().enumerate() {
                let top = base + low_y * input_width;
                let bottom = base + high_y * input_width;
                for (x, &(low_x, high_x, wx)) in xs.iter().enumerate() {
                    // Preserve the backend's FP32 expression order.
                    dest[y * width + x] = input[top + low_x] * (1.0 - wx) * (1.0 - wy)
                        + input[top + high_x] * wx * (1.0 - wy)
                        + input[bottom + low_x] * (1.0 - wx) * wy
                        + input[bottom + high_x] * wx * wy;
                }
            }
        });
    T4::from_data(
        TensorData::new(output, [batch, channels, height, width]),
        &Default::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_norm_preserves_backend_arithmetic_and_shared_views() {
        use burn_tensor::activation;

        let channels = 5;
        let channel_tensor = |values: Vec<f32>| {
            T4::from_data(
                TensorData::new(values, [1, channels, 1, 1]),
                &Default::default(),
            )
        };
        let weight = vec![0.0, -0.7, 1.3, 0.002, 3.0];
        let bias = vec![0.0, 0.5, -0.4, 0.0, -2.0];
        let mean = vec![0.0, -0.2, 0.1, 0.01, 0.7];
        let variance = vec![1.0, 0.7, 0.002, 0.0, 3.0];
        let parameters: Vec<_> = (0..channels)
            .map(|i| ChannelNorm {
                weight: weight[i],
                bias: bias[i],
                mean: mean[i],
                divisor: (variance[i] + 1e-5_f32).sqrt(),
            })
            .collect();
        for (height, width) in [(1, 1), (9, 13), (41, 67)] {
            let shape = [2, channels, height, width];
            let values: Vec<_> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.73).sin())
                .collect();
            let input = T4::from_data(TensorData::new(values, shape), &Default::default());
            for leaky in [false, true] {
                // Offset and transposed inputs exercise COW and layout handling.
                for view in [
                    input.clone(),
                    input.clone().slice_dim(0, 1..2).swap_dims(2, 3),
                ] {
                    let original = view.clone().into_data().to_vec::<f32>().unwrap();
                    let expected = (view.clone() - channel_tensor(mean.clone()))
                        / (channel_tensor(variance.clone()) + 1e-5).sqrt()
                        * channel_tensor(weight.clone())
                        + channel_tensor(bias.clone());
                    let expected = if leaky {
                        activation::leaky_relu(expected, 0.01)
                    } else {
                        activation::relu(expected)
                    };
                    let actual = normalize(view.clone(), &parameters, leaky);
                    assert_eq!(actual.dims(), expected.dims());
                    assert_eq!(
                        actual.into_data().to_vec::<f32>().unwrap(),
                        expected.into_data().to_vec::<f32>().unwrap()
                    );
                    assert_eq!(view.into_data().to_vec::<f32>().unwrap(), original);
                }
            }
            // A uniquely owned contiguous buffer must survive without a copy.
            let pointer = input
                .clone()
                .into_primitive()
                .tensor()
                .storage::<f32>()
                .as_ptr();
            let output = normalize(input, &parameters, false)
                .into_primitive()
                .tensor();
            assert_eq!(output.storage::<f32>().as_ptr(), pointer);
        }
    }

    #[test]
    fn cached_resize_matches_backend_including_pooled_and_strided_inputs() {
        use burn_tensor::{
            module,
            ops::{InterpolateMode, InterpolateOptions},
        };
        for (shape, size) in [
            ([2, 3, 7, 11], [14, 22]),
            ([1, 4, 1, 17], [21, 17]),
            ([1, 2, 1, 1], [3, 5]),
            ([1, 2, 9, 7], [3, 1]),
        ] {
            let data: Vec<f32> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.73).sin())
                .collect();
            let x =
                T4::from_data(TensorData::new(data, shape), &Default::default()).swap_dims(2, 3);
            let expected = module::interpolate(
                x.clone(),
                size,
                InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(true),
            );
            let actual = resize(x, size[0], size[1]);
            assert_eq!(
                actual.into_data().to_vec::<f32>().unwrap(),
                expected.into_data().to_vec::<f32>().unwrap()
            );
        }
    }

    #[test]
    fn winograd_matches_direct_convolution_at_edges_and_tile_batches() {
        // Include partial tiles, both padding edges, multiple batches, and a
        // window exceeding TILE_BATCH. The oracle is scalar f64 convolution.
        for [batch, channels, height, width] in [[2, 3, 1, 1], [2, 3, 5, 7], [1, 16, 89, 97]] {
            let output_channels = 5;
            let input: Vec<f32> = (0..batch * channels * height * width)
                .map(|i| (i as f32 * 0.71).sin())
                .collect();
            let weight: Vec<f32> = (0..output_channels * channels * 9)
                .map(|i| (i as f32 * 0.37).cos() * 0.1)
                .collect();
            let conv = Winograd3x3::new(T4::from_data(
                TensorData::new(weight.clone(), [output_channels, channels, 3, 3]),
                &Default::default(),
            ));
            let actual = conv.forward(T4::from_data(
                TensorData::new(input.clone(), [batch, channels, height, width]),
                &Default::default(),
            ));
            let actual = actual.into_data().to_vec::<f32>().unwrap();
            for b in 0..batch {
                for out in 0..output_channels {
                    for y in 0..height {
                        for x in 0..width {
                            let mut expected = 0.0_f64;
                            for c in 0..channels {
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let sy = (y + ky) as isize - 1;
                                        let sx = (x + kx) as isize - 1;
                                        if (0..height as isize).contains(&sy)
                                            && (0..width as isize).contains(&sx)
                                        {
                                            expected += f64::from(
                                                input[((b * channels + c) * height + sy as usize)
                                                    * width
                                                    + sx as usize],
                                            ) * f64::from(
                                                weight[(out * channels + c) * 9 + ky * 3 + kx],
                                            );
                                        }
                                    }
                                }
                            }
                            let value =
                                actual[((b * output_channels + out) * height + y) * width + x];
                            assert!(
                                (f64::from(value) - expected).abs() < 3e-5 + 3e-5 * expected.abs(),
                                "[{b},{out},{y},{x}]: {value} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}
