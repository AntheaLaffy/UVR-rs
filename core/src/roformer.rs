//! Fixed 1296 BS-RoFormer, CPU FP32, loading the original checkpoint in Rust.
//! The raw network keeps the reference ISTFT length; a file scheduler must pad/crop.

use std::{ops::ControlFlow, path::Path};

use anyhow::{Context, Result, ensure};
use burn_flex::Flex;
use burn_tensor::{Tensor, TensorData, activation};

use crate::{
    checkpoint::Loader,
    dsp::{Complex32, Padding, Spectrogram, Stft},
    task::TaskCancelled,
};

mod audio;
mod cpu;
#[cfg(feature = "openvino")]
pub mod openvino;
mod stft;
pub use audio::{RoformerOutput, RoformerProgress, RoformerStage, RoformerTimings};

type T4 = Tensor<Flex, 4>;
const DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const DEPTH: usize = 12;
const BANDS: [usize; 62] = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48,
    48, 48, 48, 128, 129,
];

pub struct RoformerModel {
    bands: Vec<BandProjection>,
    layers: Vec<[Transformer; 2]>,
    norm: RmsNorm,
    masks: Vec<MaskEstimator>,
    /// Number of independent bands/time positions processed by one
    /// attention call.  The small defaults preserve the original memory
    /// footprint; callers doing offline work can raise them when memory is
    /// available to expose more batch parallelism to Burn Flex.
    time_batch: usize,
    frequency_batch: usize,
}

impl RoformerModel {
    pub const SAMPLE_RATE: u32 = 44100;
    pub const FFT: usize = 2048;
    pub const HOP: usize = 441;
    pub const CHUNK: usize = 352800;

    pub fn load(path: &Path) -> Result<Self> {
        let identity = crate::weights::fingerprint(path)?;
        ensure!(
            identity.sha256 == "f6c94864adfb73bbb0ca58ec14d58dd0b364549e9fb61433ae51916f3e2f8d0b",
            "unsupported 1296 checkpoint SHA-256: {}",
            identity.sha256
        );
        let mut loader = Loader::new(path)?;
        let mut bands = Vec::with_capacity(BANDS.len());
        let mut masks = Vec::with_capacity(BANDS.len());
        for (index, &bins) in BANDS.iter().enumerate() {
            let input = 4 * bins; // frequency, stereo channel, complex component
            bands.push(BandProjection {
                norm: RmsNorm::load(
                    &mut loader,
                    &format!("band_split.to_features.{index}.0.gamma"),
                    input,
                )?,
                linear: Linear::load(
                    &mut loader,
                    &format!("band_split.to_features.{index}.1"),
                    input,
                    DIM,
                    true,
                )?,
            });
            masks.push(MaskEstimator {
                first: Linear::load(
                    &mut loader,
                    &format!("mask_estimators.0.to_freqs.{index}.0.0"),
                    DIM,
                    4 * DIM,
                    true,
                )?,
                last: Linear::load(
                    &mut loader,
                    &format!("mask_estimators.0.to_freqs.{index}.0.2"),
                    4 * DIM,
                    2 * input,
                    true,
                )?,
            });
        }
        let mut layers = Vec::with_capacity(DEPTH);
        for layer in 0..DEPTH {
            layers.push([
                Transformer::load(
                    &mut loader,
                    &format!("layers.{layer}.0.layers.0"),
                    Self::CHUNK / Self::HOP + 1,
                )?,
                Transformer::load(
                    &mut loader,
                    &format!("layers.{layer}.1.layers.0"),
                    BANDS.len(),
                )?,
            ]);
        }
        let norm = RmsNorm::load(&mut loader, "final_norm.gamma", DIM)?;
        loader.finish(699)?;
        let time_batch = configured_batch("UVR_ROFORMER_TIME_BATCH", 2)?;
        let frequency_batch = configured_batch("UVR_ROFORMER_FREQUENCY_BATCH", 32)?;
        Ok(Self {
            bands,
            layers,
            norm,
            masks,
            time_batch,
            frequency_batch,
        })
    }

    /// Attention batching selected when the checkpoint was loaded.  This is
    /// useful for reporting an experiment without inferring it from timing.
    pub fn batch_sizes(&self) -> (usize, usize) {
        (self.time_batch, self.frequency_batch)
    }

    /// Planar stereo input, 1025..=352800 samples/channel. Returns stereo with
    /// floor(samples/441)*441 samples/channel, exactly as the raw reference model.
    pub fn predict_window(&self, audio: &[f32], samples: usize) -> Result<Vec<f32>> {
        self.predict_window_with_progress(audio, samples, |_, _| ControlFlow::Continue(()))
    }

    /// Progress counts independent frequency/time batches, with cancellation
    /// between batches. Batching does not split either attention sequence.
    pub fn predict_window_with_progress(
        &self,
        audio: &[f32],
        samples: usize,
        mut progress: impl FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Vec<f32>> {
        ensure!(
            samples > Self::FFT / 2 && samples <= Self::CHUNK && audio.len() == 2 * samples,
            "1296 expects stereo windows with 1025..=352800 samples/channel"
        );
        ensure!(audio.iter().all(|v| v.is_finite()), "audio must be finite");
        let frames = samples / Self::HOP + 1;
        let total = 4
            + 2 * BANDS.len()
            + DEPTH
                * (BANDS.len().div_ceil(self.time_batch)
                    + frames.div_ceil(self.frequency_batch));
        if progress(0, total).is_break() {
            return Err(TaskCancelled.into());
        }
        let mut completed = 0;
        let mut advance = || -> Result<()> {
            completed += 1;
            if progress(completed, total).is_break() {
                return Err(TaskCancelled.into());
            }
            Ok(())
        };
        let mut stft = Stft::new(Self::FFT, Self::HOP, Padding::Reflect)?;
        let mut analysis = stft::PreciseStft::new();
        let mut spectra = Vec::with_capacity(2);
        for channel in audio.chunks_exact(samples) {
            let spectrum = analysis.forward(channel);
            ensure!(
                spectrum
                    .values
                    .iter()
                    .all(|v| v.re.is_finite() && v.im.is_finite()),
                "nonfinite RoFormer spectrum"
            );
            spectra.push(spectrum);
            advance()?;
        }
        let mut features = Vec::with_capacity(BANDS.len());
        let mut first_bin = 0;
        for (band, &bins) in self.bands.iter().zip(&BANDS) {
            let mut input = Vec::with_capacity(frames * bins * 4);
            for frame in 0..frames {
                for bin in first_bin..first_bin + bins {
                    for spectrum in &spectra {
                        let value = spectrum.values[bin * frames + frame];
                        input.extend_from_slice(&[value.re, value.im]);
                    }
                }
            }
            let x = T4::from_data(
                TensorData::new(input, [1, 1, frames, 4 * bins]),
                &Default::default(),
            );
            features.push(band.linear.forward(band.norm.forward(x)));
            first_bin += bins;
            advance()?;
        }
        let mut x = T4::cat(features, 1); // [1, band, time, feature]
        for [time, frequency] in &self.layers {
            x = time.forward_batches(x, self.time_batch, &mut advance)?;
            x = frequency
                .forward_batches(x.swap_dims(1, 2), self.frequency_batch, &mut advance)?
                .swap_dims(1, 2);
        }
        x = self.norm.forward(x);
        let mut masked: Vec<_> = (0..2)
            .map(|_| Spectrogram {
                bins: Self::FFT / 2 + 1,
                frames,
                values: vec![Complex32::default(); (Self::FFT / 2 + 1) * frames],
            })
            .collect();
        first_bin = 0;
        for (index, (&bins, estimator)) in BANDS.iter().zip(&self.masks).enumerate() {
            let mask = estimator
                .forward(x.clone().slice_dim(1, index..index + 1))
                .into_data()
                .to_vec::<f32>()?;
            ensure!(
                mask.iter().all(|v| v.is_finite()),
                "nonfinite RoFormer mask"
            );
            for frame in 0..frames {
                for bin in 0..bins {
                    for channel in 0..2 {
                        let i = frame * bins * 4 + bin * 4 + channel * 2;
                        let position = (first_bin + bin) * frames + frame;
                        masked[channel].values[position] = spectra[channel].values[position]
                            * Complex32::new(mask[i], mask[i + 1]);
                    }
                }
            }
            first_bin += bins;
            advance()?;
        }
        let output_samples = samples / Self::HOP * Self::HOP;
        let mut output = Vec::with_capacity(2 * output_samples);
        for spectrum in &masked {
            output.extend(stft.inverse(spectrum, None)?);
            advance()?;
        }
        ensure!(
            output.len() == 2 * output_samples && output.iter().all(|v| v.is_finite()),
            "invalid RoFormer waveform"
        );
        Ok(output)
    }
}

fn configured_batch(name: &str, default: usize) -> Result<usize> {
    let value = std::env::var(name)
        .map_or_else(|_| Ok(default), |raw| raw.parse::<usize>())
        .with_context(|| format!("{name} must be a positive integer"))?;
    ensure!(value > 0, "{name} must be a positive integer");
    Ok(value)
}

struct Linear {
    weight: T4,
    bias: Option<T4>,
    flattened: bool,
}

impl Linear {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = Tensor::<Flex, 2>::from_data(
            loader.float(&format!("{prefix}.weight"), &[output, input])?,
            &Default::default(),
        )
        .transpose()
        .reshape([1, 1, input, output]);
        let bias = if bias {
            Some(
                loader
                    .channel(&format!("{prefix}.bias"), output)?
                    .reshape([1, 1, 1, output]),
            )
        } else {
            None
        };
        let flattened = configured_linear_layout()?;
        Ok(Self {
            weight,
            bias,
            flattened,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let x = if self.flattened {
            let [a, b, c, input] = x.dims();
            let output = self.weight.dims()[3];
            x.reshape([a * b * c, input])
                .matmul(self.weight.clone().reshape([input, output]))
                .reshape([a, b, c, output])
        } else {
            x.matmul(self.weight.clone())
        };
        match &self.bias {
            Some(bias) => x + bias.clone(),
            None => x,
        }
    }
}

fn configured_linear_layout() -> Result<bool> {
    match std::env::var("UVR_LINEAR_LAYOUT").as_deref() {
        Ok("flattened") => Ok(true),
        Ok("batched") | Err(_) => Ok(false),
        Ok(value) => anyhow::bail!("UVR_LINEAR_LAYOUT must be batched or flattened, got {value}"),
    }
}

struct RmsNorm {
    gamma: T4,
}

impl RmsNorm {
    fn load(loader: &mut Loader, name: &str, dim: usize) -> Result<Self> {
        Ok(Self {
            gamma: loader.channel(name, dim)?.reshape([1, 1, 1, dim]),
        })
    }
    fn forward(&self, x: T4) -> T4 {
        cpu::rms_norm(x, self.gamma.clone())
    }
}

struct BandProjection {
    norm: RmsNorm,
    linear: Linear,
}
struct MaskEstimator {
    first: Linear,
    last: Linear,
}

impl MaskEstimator {
    fn forward(&self, x: T4) -> T4 {
        let x = self.last.forward(self.first.forward(x).tanh());
        let half = x.dims()[3] / 2;
        x.clone().slice_dim(3, 0..half) * activation::sigmoid(x.slice_dim(3, half..2 * half))
    }
}

struct Transformer {
    attention: Attention,
    norm: RmsNorm,
    first: Linear,
    last: Linear,
}

impl Transformer {
    fn load(loader: &mut Loader, prefix: &str, sequence: usize) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(loader, &format!("{prefix}.0"), sequence)?,
            norm: RmsNorm::load(loader, &format!("{prefix}.1.net.0.gamma"), DIM)?,
            first: Linear::load(loader, &format!("{prefix}.1.net.1"), DIM, 4 * DIM, true)?,
            last: Linear::load(loader, &format!("{prefix}.1.net.4"), 4 * DIM, DIM, true)?,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let x = self.attention.forward(x.clone()) + x;
        self.last
            .forward(cpu::gelu(self.first.forward(self.norm.forward(x.clone()))))
            + x
    }
    fn forward_batches(
        &self,
        x: T4,
        size: usize,
        advance: &mut impl FnMut() -> Result<()>,
    ) -> Result<T4> {
        let batches = x.dims()[1];
        let mut output = Vec::with_capacity(batches.div_ceil(size));
        for start in (0..batches).step_by(size) {
            output.push(self.forward(x.clone().slice_dim(1, start..(start + size).min(batches))));
            advance()?;
        }
        Ok(T4::cat(output, 1))
    }
}

struct Attention {
    norm: RmsNorm,
    qkv: Linear,
    gates: Linear,
    out: Linear,
    cos: T4,
    sin: T4,
}

impl Attention {
    fn load(loader: &mut Loader, prefix: &str, sequence: usize) -> Result<Self> {
        let data = loader.float(&format!("{prefix}.rotary_embed.freqs"), &[HEAD_DIM / 2])?;
        let frequencies = data
            .as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;
        let mut cos = Vec::with_capacity(sequence * HEAD_DIM / 2);
        let mut sin = Vec::with_capacity(sequence * HEAD_DIM / 2);
        for index in 0..sequence {
            for frequency in frequencies {
                let angle = index as f32 * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self {
            norm: RmsNorm::load(loader, &format!("{prefix}.norm.gamma"), DIM)?,
            qkv: Linear::load(loader, &format!("{prefix}.to_qkv"), DIM, 3 * DIM, false)?,
            gates: Linear::load(loader, &format!("{prefix}.to_gates"), DIM, HEADS, true)?,
            out: Linear::load(loader, &format!("{prefix}.to_out.0"), DIM, DIM, false)?,
            cos: T4::from_data(
                TensorData::new(cos, [1, 1, sequence, HEAD_DIM / 2]),
                &Default::default(),
            ),
            sin: T4::from_data(
                TensorData::new(sin, [1, 1, sequence, HEAD_DIM / 2]),
                &Default::default(),
            ),
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let [_, batch, sequence, _] = x.dims();
        let x = self.norm.forward(x);
        let projected = self.qkv.forward(x.clone());
        let component = |index| {
            projected
                .clone()
                .slice_dim(3, index * DIM..(index + 1) * DIM)
                .reshape([batch, sequence, HEADS, HEAD_DIM])
                .swap_dims(1, 2)
        };
        let cos = self.cos.clone().slice_dim(2, 0..sequence);
        let sin = self.sin.clone().slice_dim(2, 0..sequence);
        let q = cpu::rotate(component(0), cos.clone(), sin.clone());
        let k = cpu::rotate(component(1), cos, sin);
        let scores = q.matmul(k.transpose()) * (HEAD_DIM as f32).powf(-0.5);
        let probabilities = activation::softmax(scores, 3);
        let gates = activation::sigmoid(self.gates.forward(x))
            .reshape([batch, sequence, HEADS, 1])
            .swap_dims(1, 2);
        let values = probabilities.matmul(component(2)) * gates;
        self.out
            .forward(values.swap_dims(1, 2).reshape([1, batch, sequence, DIM]))
    }
}
