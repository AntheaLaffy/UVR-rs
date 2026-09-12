//! CPU kernels for the fixed FP32 RoFormer.

use burn_tensor::TensorPrimitive;
use rayon::prelude::*;

use super::T4;

/// Leave the backend's reduction and its FP32 ordering intact; fuse the three
/// broadcast operations that apply the scalar length and per-feature gain.
pub(super) fn rms_norm(x: T4, gamma: T4) -> T4 {
    let dim = x.dims()[3];
    let lengths = (x.clone() * x.clone())
        .sum_dim(3)
        .sqrt()
        .clamp_min(1e-12)
        .into_primitive()
        .tensor()
        .to_contiguous();
    let gamma = gamma.into_primitive().tensor().to_contiguous();
    let gamma = gamma.storage::<f32>();
    assert_eq!(dim, gamma.len());
    let scale = (dim as f32).sqrt();
    let mut x = x.into_primitive().tensor().to_contiguous();
    x.storage_mut::<f32>()
        .par_chunks_mut(dim)
        .zip(lengths.storage::<f32>().par_iter())
        .with_min_len(32)
        .for_each(|(row, &length)| {
            for (value, &gain) in row.iter_mut().zip(gamma) {
                *value = *value / length * scale * gain;
            }
        });
    T4::from_primitive(TensorPrimitive::Float(x))
}

/// Apply adjacent-pair RoPE directly, avoiding strided even/odd broadcasting
/// and the intermediate stack. Shared projections remain protected by COW.
pub(super) fn rotate(x: T4, cos: T4, sin: T4) -> T4 {
    let [_, _, sequence, dim] = x.dims();
    assert_eq!(dim % 2, 0);
    assert_eq!(cos.dims(), [1, 1, sequence, dim / 2]);
    assert_eq!(cos.dims(), sin.dims());
    let cos = cos.into_primitive().tensor().to_contiguous();
    let sin = sin.into_primitive().tensor().to_contiguous();
    let cos = cos.storage::<f32>();
    let sin = sin.storage::<f32>();
    let mut x = x.into_primitive().tensor().to_contiguous();
    x.storage_mut::<f32>()
        .par_chunks_mut(dim)
        .enumerate()
        .with_min_len(64)
        .for_each(|(row_index, row)| {
            let offset = row_index % sequence * (dim / 2);
            for (pair_index, pair) in row.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let c = cos[offset + pair_index];
                let s = sin[offset + pair_index];
                let [even, odd] = [pair[0], pair[1]];
                pair[0] = even * c - odd * s;
                pair[1] = odd * c + even * s;
            }
        });
    T4::from_primitive(TensorPrimitive::Float(x))
}

/// Flex 0.21 evaluates GELU serially. Keep its exact erf implementation and
/// arithmetic while distributing independent elements across the existing pool.
pub(super) fn gelu(x: T4) -> T4 {
    let mut x = x.into_primitive().tensor().to_contiguous();
    let apply = |values: &mut [f32]| {
        for value in values {
            *value = 0.5
                * *value
                * (1.0 + burn_flex::ops::unary::erf_f32(*value / std::f32::consts::SQRT_2));
        }
    };
    let values = x.storage_mut::<f32>();
    if values.len() < 32768 {
        apply(values);
    } else {
        values.par_chunks_mut(16384).for_each(apply);
    }
    T4::from_primitive(TensorPrimitive::Float(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::{TensorData, activation};

    fn bits(x: T4) -> Vec<u32> {
        x.into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    #[test]
    fn fused_rms_norm_keeps_reduction_order_on_shared_and_strided_inputs() {
        for dim in [8, 512, 516] {
            let shape = [1, 3, 65, dim];
            let mut data: Vec<f32> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.071).sin())
                .collect();
            data[..dim].fill(0.0);
            data[dim..2 * dim].fill(1e-20);
            let x = T4::from_data(TensorData::new(data, shape), &Default::default());
            let gamma = T4::from_data(
                TensorData::new(
                    (0..dim)
                        .map(|i| (i as f32 * 0.037).cos())
                        .collect::<Vec<_>>(),
                    [1, 1, 1, dim],
                ),
                &Default::default(),
            );
            for view in [x.clone(), x.clone().slice_dim(2, 1..64).swap_dims(1, 2)] {
                let before = bits(view.clone());
                let length = (view.clone() * view.clone())
                    .sum_dim(3)
                    .sqrt()
                    .clamp_min(1e-12);
                let expected = view.clone() / length * (dim as f32).sqrt() * gamma.clone();
                assert_eq!(bits(rms_norm(view.clone(), gamma.clone())), bits(expected));
                assert_eq!(bits(view), before);
            }
        }
    }

    #[test]
    fn fused_rope_keeps_pair_order_on_shared_projections_and_short_caches() {
        use burn_tensor::Tensor;
        for (sequence, dim) in [(3, 8), (65, 64)] {
            let shape = [2, sequence + 2, 8, dim];
            let data: Vec<f32> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let x = T4::from_data(TensorData::new(data, shape), &Default::default());
            let values = |cosine: bool| {
                (0..(sequence + 2) * dim / 2)
                    .map(|i| {
                        if cosine {
                            (i as f32 * 0.037).cos()
                        } else {
                            (i as f32 * 0.037).sin()
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let cache = |values| {
                T4::from_data(
                    TensorData::new(values, [1, 1, sequence + 2, dim / 2]),
                    &Default::default(),
                )
                .slice_dim(2, 1..sequence + 1)
            };
            let cos = cache(values(true));
            let sin = cache(values(false));
            let view = x.clone().slice_dim(1, 1..sequence + 1).swap_dims(1, 2);
            let before = bits(x.clone());
            let pairs = view.clone().reshape([2, 8, sequence, dim / 2, 2]);
            let even = pairs.clone().slice_dim(4, 0..1).squeeze_dim::<4>(4);
            let odd = pairs.slice_dim(4, 1..2).squeeze_dim::<4>(4);
            let a = even.clone() * cos.clone() - odd.clone() * sin.clone();
            let b = odd * cos.clone() + even * sin.clone();
            let expected = Tensor::stack::<5>(vec![a, b], 4).reshape([2, 8, sequence, dim]);
            assert_eq!(bits(rotate(view, cos, sin)), bits(expected));
            assert_eq!(bits(x), before);
        }
    }

    #[test]
    fn parallel_gelu_preserves_values_views_and_owned_storage() {
        for shape in [[1, 1, 3, 17], [1, 2, 81, 2048]] {
            let mut data: Vec<f32> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.071).sin() * 12.0)
                .collect();
            data[..6].copy_from_slice(&[0.0, -0.0, 1e-30, -1e-30, f32::MAX, f32::MIN]);
            let input = T4::from_data(TensorData::new(data, shape), &Default::default());
            for view in [
                input.clone(),
                input.clone().slice_dim(2, 1..shape[2]).swap_dims(2, 3),
            ] {
                let before = view.clone().into_data().to_vec::<f32>().unwrap();
                let expected = activation::gelu(view.clone());
                let actual = gelu(view.clone());
                assert_eq!(actual.dims(), expected.dims());
                assert_eq!(bits(actual), bits(expected));
                assert_eq!(view.into_data().to_vec::<f32>().unwrap(), before);
            }
            let pointer = input
                .clone()
                .into_primitive()
                .tensor()
                .storage::<f32>()
                .as_ptr();
            let output = gelu(input).into_primitive().tensor();
            assert_eq!(output.storage::<f32>().as_ptr(), pointer);
        }
    }
}
