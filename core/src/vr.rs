//! CPU FP32 implementations of the fixed VR mask networks.
//!
//! Architecture reference: UVR `5517e0cf`, lib_v5/vr_network/{nets,layers}.py.
//! Raw models operate on normalized multiband magnitude windows.
//! `VrSeparator` connects them to the fixed stereo PCM audio pipeline.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use burn_flex::Flex;
use burn_tensor::{DType, Tensor, TensorData, activation, module, ops::ConvOptions};

type T4 = Tensor<Flex, 4>;
use crate::checkpoint::Loader;

mod cpu;
use cpu::Winograd3x3;
mod deecho;
pub use deecho::DeEchoModel;
mod audio;
pub use audio::{TaskCancelled, VrOptions, VrOutput, VrProgress, VrSeparator, VrStage, VrTimings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HpKaraokeVariant {
    Five,
    Six,
}

impl HpKaraokeVariant {
    pub fn bins(self) -> usize {
        match self {
            Self::Five => 672,
            Self::Six => 640,
        }
    }
}

/// A loaded network, reusable across consecutive windows without Python.
pub struct HpKaraokeModel {
    variant: HpKaraokeVariant,
    low: BaseNet,
    high: BaseNet,
    bridge2: Block,
    stage2: BaseNet,
    bridge3: Block,
    stage3: BaseNet,
    out: Conv,
}

impl HpKaraokeModel {
    pub const OFFSET: usize = 128;

    /// Load the original checkpoint directly with a Rust parser. Only the two
    /// audited identities are accepted; shape/key checks additionally guard
    /// against an incorrect Rust architecture or checkpoint interpretation.
    pub fn load(path: &Path) -> Result<Self> {
        let identity = crate::weights::fingerprint(path)?;
        let variant = match identity.sha256.as_str() {
            "fe00891defbb61f4261500af22f7624f1a3df8dc75fa3998d1aece02e6be4537" => {
                HpKaraokeVariant::Five
            }
            "4ce7eaaa9e56f09366b788aebf6d3a72aec8145692c56f1e090e4e7e2d7ce65f" => {
                HpKaraokeVariant::Six
            }
            _ => anyhow::bail!(
                "unsupported 5-HP/6-HP checkpoint SHA-256: {}",
                identity.sha256
            ),
        };
        let mut loader = Loader::new(path)?;
        let model = Self {
            variant,
            low: BaseNet::load(&mut loader, "stg1_low_band_net", 2, 32)?,
            high: BaseNet::load(&mut loader, "stg1_high_band_net", 2, 32)?,
            bridge2: Block::load(&mut loader, "stg2_bridge.conv", 34, 16, 1, 1, 0, false)?,
            stage2: BaseNet::load(&mut loader, "stg2_full_band_net", 16, 32)?,
            bridge3: Block::load(&mut loader, "stg3_bridge.conv", 66, 32, 1, 1, 0, false)?,
            stage3: BaseNet::load(&mut loader, "stg3_full_band_net", 32, 64)?,
            out: Conv::load(&mut loader, "out.weight", [2, 64, 1, 1], 1, 0, 1, 1)?,
        };
        // Training-only heads are present in the checkpoint. Validate them even
        // though eval mode never executes them, so unknown keys cannot slip in.
        loader.float("aux1_out.weight", &[2, 32, 1, 1])?;
        loader.float("aux2_out.weight", &[2, 32, 1, 1])?;
        loader.finish(459)?;
        Ok(model)
    }

    pub fn variant(&self) -> HpKaraokeVariant {
        self.variant
    }

    /// Input is channel/frequency/time-major `[2, bins + 1, frames]`.
    /// Output uses the same layout with `frames - 256` time positions.
    /// Callers normalize magnitudes and provide the 128-frame context on each
    /// side; this API deliberately does not interpret them as audio samples.
    pub fn predict_mask(&self, magnitude: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.predict_masks(magnitude, frames, 1)
    }

    /// Independent windows in `[batch, 2, bins + 1, frames]` order. The batch
    /// bound limits activation memory; callers still own audio window scheduling.
    pub fn predict_masks(
        &self,
        magnitude: &[f32],
        frames: usize,
        batch: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            (1..=4).contains(&batch),
            "mask batch must be between 1 and 4"
        );
        let bins = self.variant.bins();
        let x = input_windows(magnitude, frames, bins, Self::OFFSET, batch)?;
        let half = bins / 2;
        let aux1 = T4::cat(
            vec![
                self.low.forward(x.clone().slice_dim(2, 0..half)),
                self.high.forward(x.clone().slice_dim(2, half..bins)),
            ],
            2,
        );
        let aux2 = self.stage2.forward(
            self.bridge2
                .forward(T4::cat(vec![x.clone(), aux1.clone()], 1)),
        );
        let h = self
            .stage3
            .forward(self.bridge3.forward(T4::cat(vec![x, aux1, aux2], 1)));
        let mask = activation::sigmoid(self.out.forward(h));
        finish_mask(mask, frames, bins, Self::OFFSET)
    }
}

struct Conv {
    weight: ConvWeight,
    stride: usize,
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

enum ConvWeight {
    Direct(T4),
    Winograd(Winograd3x3),
}

impl Conv {
    fn load(
        loader: &mut Loader,
        name: &str,
        shape: [usize; 4],
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = T4::from_data(loader.float(name, &shape)?, &Default::default());
        // Dense stride-one 3x3 layers dominate VR decoder work. Keep the
        // backend's specialized paths for small-channel, strided and grouped layers.
        let weight = if shape[1] >= 16
            && shape[2..] == [3, 3]
            && (stride, padding, dilation, groups) == (1, 1, 1, 1)
        {
            ConvWeight::Winograd(Winograd3x3::new(weight))
        } else {
            ConvWeight::Direct(weight)
        };
        Ok(Self {
            weight,
            stride,
            padding: [padding; 2],
            dilation: [dilation; 2],
            groups,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let weight = match &self.weight {
            ConvWeight::Winograd(conv) => return conv.forward(x),
            ConvWeight::Direct(weight) => weight,
        };
        module::conv2d(
            x,
            weight.clone(),
            None,
            ConvOptions::new([self.stride; 2], self.padding, self.dilation, self.groups),
        )
    }
}

struct Norm {
    parameters: Vec<cpu::ChannelNorm>,
}

impl Norm {
    fn load(loader: &mut Loader, prefix: &str, channels: usize) -> Result<Self> {
        loader.data(&format!("{prefix}.num_batches_tracked"), &[], DType::I64)?;
        let weight = loader
            .float(&format!("{prefix}.weight"), &[channels])?
            .to_vec::<f32>()?;
        let bias = loader
            .float(&format!("{prefix}.bias"), &[channels])?
            .to_vec::<f32>()?;
        let mean = loader
            .float(&format!("{prefix}.running_mean"), &[channels])?
            .to_vec::<f32>()?;
        let variance = loader
            .float(&format!("{prefix}.running_var"), &[channels])?
            .to_vec::<f32>()?;
        Ok(Self {
            parameters: (0..channels)
                .map(|i| cpu::ChannelNorm {
                    weight: weight[i],
                    bias: bias[i],
                    mean: mean[i],
                    divisor: (variance[i] + 1e-5).sqrt(),
                })
                .collect(),
        })
    }
    fn forward(&self, x: T4, leaky: bool) -> T4 {
        cpu::normalize(x, &self.parameters, leaky)
    }
}

struct Block {
    conv: Conv,
    norm: Norm,
    leaky: bool,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        leaky: bool,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("{prefix}.0.weight"),
                [output, input, kernel, kernel],
                stride,
                padding,
                1,
                1,
            )?,
            norm: Norm::load(loader, &format!("{prefix}.1"), output)?,
            leaky,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        self.norm.forward(self.conv.forward(x), self.leaky)
    }
}

struct Separable {
    depthwise: Conv,
    pointwise: Conv,
    norm: Norm,
}

impl Separable {
    fn load(loader: &mut Loader, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            depthwise: Conv::load(
                loader,
                &format!("{prefix}.0.weight"),
                [channels, 1, 3, 3],
                1,
                dilation,
                dilation,
                channels,
            )?,
            pointwise: Conv::load(
                loader,
                &format!("{prefix}.1.weight"),
                [channels, channels, 1, 1],
                1,
                0,
                1,
                1,
            )?,
            norm: Norm::load(loader, &format!("{prefix}.2"), channels)?,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        self.norm
            .forward(self.pointwise.forward(self.depthwise.forward(x)), false)
    }
}

struct Aspp {
    pooled: Block,
    direct: Block,
    dilated: [Separable; 3],
    bottleneck: Block,
}

impl Aspp {
    fn load(loader: &mut Loader, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            pooled: Block::load(
                loader,
                &format!("{prefix}.conv1.1.conv"),
                channels,
                channels,
                1,
                1,
                0,
                false,
            )?,
            direct: Block::load(
                loader,
                &format!("{prefix}.conv2.conv"),
                channels,
                channels,
                1,
                1,
                0,
                false,
            )?,
            dilated: [
                Separable::load(loader, &format!("{prefix}.conv3.conv"), channels, 4)?,
                Separable::load(loader, &format!("{prefix}.conv4.conv"), channels, 8)?,
                Separable::load(loader, &format!("{prefix}.conv5.conv"), channels, 16)?,
            ],
            bottleneck: Block::load(
                loader,
                &format!("{prefix}.bottleneck.0.conv"),
                channels * 5,
                channels * 2,
                1,
                1,
                0,
                false,
            )?,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let [_, _, height, width] = x.dims();
        let mut features = vec![
            resize(self.pooled.forward(x.clone().mean_dim(2)), height, width),
            self.direct.forward(x.clone()),
        ];
        for layer in &self.dilated {
            features.push(layer.forward(x.clone()));
        }
        self.bottleneck.forward(T4::cat(features, 1))
    }
}

struct BaseNet {
    encoders: Vec<(Block, Block)>,
    aspp: Aspp,
    decoders: Vec<Block>,
}

impl BaseNet {
    fn load(loader: &mut Loader, prefix: &str, input: usize, channels: usize) -> Result<Self> {
        let mut encoders = Vec::new();
        let mut input_channels = input;
        for level in 0..4 {
            let output = channels << level;
            encoders.push((
                Block::load(
                    loader,
                    &format!("{prefix}.enc{}.conv1.conv", level + 1),
                    input_channels,
                    output,
                    3,
                    1,
                    1,
                    true,
                )?,
                Block::load(
                    loader,
                    &format!("{prefix}.enc{}.conv2.conv", level + 1),
                    output,
                    output,
                    3,
                    2,
                    1,
                    true,
                )?,
            ));
            input_channels = output;
        }
        let aspp = Aspp::load(loader, &format!("{prefix}.aspp"), channels * 8)?;
        let mut decoders = Vec::new();
        for level in (0..4).rev() {
            let output = channels << level;
            decoders.push(Block::load(
                loader,
                &format!("{prefix}.dec{}.conv.conv", level + 1),
                output * 3,
                output,
                3,
                1,
                1,
                false,
            )?);
        }
        Ok(Self {
            encoders,
            aspp,
            decoders,
        })
    }
    fn forward(&self, mut x: T4) -> T4 {
        let mut skips = Vec::with_capacity(4);
        for (first, second) in &self.encoders {
            let skip = first.forward(x);
            x = second.forward(skip.clone());
            skips.push(skip);
        }
        x = self.aspp.forward(x);
        for (decoder, skip) in self.decoders.iter().zip(skips.into_iter().rev()) {
            let [_, _, height, width] = x.dims();
            x = resize(x, height * 2, width * 2);
            let offset = (skip.dims()[3] - width * 2) / 2;
            x = decoder.forward(T4::cat(
                vec![x, skip.slice_dim(3, offset..offset + width * 2)],
                1,
            ));
        }
        x
    }
}

fn resize(x: T4, height: usize, width: usize) -> T4 {
    cpu::resize(x, height, width)
}

fn input_window(magnitude: &[f32], frames: usize, bins: usize, offset: usize) -> Result<T4> {
    input_windows(magnitude, frames, bins, offset, 1)
}

fn input_windows(
    magnitude: &[f32],
    frames: usize,
    bins: usize,
    offset: usize,
    batch: usize,
) -> Result<T4> {
    ensure!(
        frames > 2 * offset && frames.is_multiple_of(16),
        "window length must be a multiple of 16 and exceed {} frames",
        2 * offset
    );
    let count = 2_usize
        .checked_mul(batch)
        .and_then(|n| n.checked_mul(bins + 1))
        .and_then(|n| n.checked_mul(frames))
        .context("window size overflow")?;
    ensure!(magnitude.len() == count, "incorrect magnitude window size");
    ensure!(
        magnitude.iter().all(|v| v.is_finite() && *v >= 0.0),
        "magnitude must be finite and nonnegative"
    );
    Ok(T4::from_data(
        TensorData::new(magnitude.to_vec(), [batch, 2, bins + 1, frames]),
        &Default::default(),
    )
    .slice_dim(2, 0..bins))
}

fn finish_mask(mask: T4, frames: usize, bins: usize, offset: usize) -> Result<Vec<f32>> {
    // The reference replicates the highest predicted frequency into Nyquist.
    let nyquist = mask.clone().slice_dim(2, bins - 1..bins);
    let output = T4::cat(vec![mask, nyquist], 2).slice_dim(3, offset..frames - offset);
    let values = output
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("{error:?}"))?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "network produced nonfinite mask"
    );
    Ok(values)
}
