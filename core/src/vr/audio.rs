use std::{ops::ControlFlow, path::Path, time::Instant};

use anyhow::{Result, ensure};
use rayon::prelude::*;

use super::{DeEchoModel, HpKaraokeModel, HpKaraokeVariant};
pub use crate::task::TaskCancelled;
use crate::vr_dsp::{self, VrVariant};

#[derive(Debug, Clone, Copy)]
pub struct VrOptions {
    /// Multiple of 16, at most 2048, and larger than both context margins.
    pub window_frames: usize,
    /// Number of independent spectrogram windows sent through the HP network
    /// in one call.  The product default stays at one because the best batch
    /// size is model and machine dependent; DeEcho always remains batch one.
    pub inference_batch: usize,
    /// Number of independent batch-one windows evaluated concurrently.
    /// HP defaults to four in-flight windows; DeEcho forces one because its
    /// recurrent state cannot be shared across windows.
    pub window_parallelism: usize,
}

impl Default for VrOptions {
    fn default() -> Self {
        Self {
            window_frames: 512,
            inference_batch: 1,
            window_parallelism: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrStage {
    Analysis,
    Inference,
    Reconstruction,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub struct VrProgress {
    pub stage: VrStage,
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug)]
pub struct VrTimings {
    pub analysis_seconds: f64,
    pub network_seconds: f64,
    pub reconstruction_seconds: f64,
}

pub struct VrOutput {
    /// Stereo planar samples: all left samples, then all right samples.
    pub primary: Vec<f32>,
    /// Complementary mask reconstructed through the same filter bank.
    pub residual: Vec<f32>,
    pub samples_per_channel: usize,
    pub sample_rate: u32,
    pub timings: VrTimings,
}

enum Network {
    Hp(Box<HpKaraokeModel>),
    DeEcho(Box<DeEchoModel>),
}

/// Reuses original checkpoint weights across audio tasks. No files or processes
/// are created by separation; callers encode completed outputs themselves.
pub struct VrSeparator {
    variant: VrVariant,
    network: Network,
}

impl VrSeparator {
    pub fn load(variant: VrVariant, path: &Path) -> Result<Self> {
        let network = match variant {
            VrVariant::DeEcho => Network::DeEcho(Box::new(DeEchoModel::load(path)?)),
            VrVariant::HpFive | VrVariant::HpSix => {
                let model = HpKaraokeModel::load(path)?;
                let expected = if variant == VrVariant::HpFive {
                    HpKaraokeVariant::Five
                } else {
                    HpKaraokeVariant::Six
                };
                ensure!(
                    model.variant() == expected,
                    "checkpoint does not match the requested VR variant"
                );
                Network::Hp(Box::new(model))
            }
        };
        Ok(Self { variant, network })
    }

    /// Separates mono or stereo PCM with the documented basic preset.
    /// Returning Break from progress cancels before the next stage/window.
    pub fn separate(
        &self,
        channels: &[&[f32]],
        sample_rate: u32,
        options: VrOptions,
        progress: impl FnMut(VrProgress) -> ControlFlow<()>,
    ) -> Result<VrOutput> {
        separate_with(
            self.variant,
            channels,
            sample_rate,
            options,
            progress,
            |input, frames, batch| match &self.network {
                Network::Hp(model) => model.predict_masks(input, frames, batch),
                Network::DeEcho(model) => {
                    ensure!(batch == 1, "DeEcho inference batch must be one");
                    model.predict_mask(input, frames)
                }
            },
        )
    }
}

fn separate_with(
    variant: VrVariant,
    channels: &[&[f32]],
    sample_rate: u32,
    options: VrOptions,
    mut progress: impl FnMut(VrProgress) -> ControlFlow<()>,
    predict: impl Fn(&[f32], usize, usize) -> Result<Vec<f32>> + Sync,
) -> Result<VrOutput> {
    let window = options.window_frames;
    let offset = variant.offset();
    ensure!(
        window > 2 * offset && window <= 2048 && window.is_multiple_of(16),
        "window must be a multiple of 16, greater than twice model context, and at most 2048"
    );
    let batch_size = match variant {
        VrVariant::DeEcho => 1,
        VrVariant::HpFive | VrVariant::HpSix => options.inference_batch,
    };
    ensure!(
        (1..=4).contains(&batch_size),
        "inference batch must be between 1 and 4"
    );
    let window_parallelism = match variant {
        VrVariant::DeEcho => 1,
        VrVariant::HpFive | VrVariant::HpSix => options.window_parallelism,
    };
    ensure!(
        (1..=8).contains(&window_parallelism),
        "window parallelism must be between 1 and 8"
    );
    let group_width = if batch_size == 1 {
        window_parallelism
    } else {
        batch_size
    };
    let mut report = |stage, completed, total| -> Result<()> {
        if progress(VrProgress {
            stage,
            completed,
            total,
        })
        .is_break()
        {
            return Err(TaskCancelled.into());
        }
        Ok(())
    };
    report(VrStage::Analysis, 0, 1)?;
    let start = Instant::now();
    let spectrum = vr_dsp::analyze(variant, channels, sample_rate)?;
    let magnitude = spectrum.magnitude();
    let maximum = magnitude.iter().copied().fold(0.0, f32::max);
    ensure!(maximum.is_finite(), "nonfinite spectral magnitude");
    let analysis_seconds = start.elapsed().as_secs_f64();
    report(VrStage::Analysis, 1, 1)?;
    let samples_per_channel = spectrum.output_samples();
    if maximum == 0.0 {
        // Upstream normalizes by zero here. Preserve exact silence without running a network.
        report(VrStage::Inference, 0, 0)?;
        report(VrStage::Reconstruction, 1, 1)?;
        report(VrStage::Complete, 1, 1)?;
        return Ok(VrOutput {
            primary: vec![0.0; 2 * samples_per_channel],
            residual: vec![0.0; 2 * samples_per_channel],
            samples_per_channel,
            sample_rate: 44100,
            timings: VrTimings {
                analysis_seconds,
                network_seconds: 0.0,
                reconstruction_seconds: 0.0,
            },
        });
    }
    let frames = spectrum.frames();
    let planes = 2 * spectrum.bins();
    let roi = window - 2 * offset;
    // UVR make_padding adds one ROI even when the width divides exactly.
    let patches = frames / roi + 1;
    let mut mask = vec![0.0; magnitude.len()];
    report(VrStage::Inference, 0, patches)?;
    let start = Instant::now();
    // Window-parallel HP inference keeps one input buffer per in-flight
    // window. Reuse these buffers across groups so long files do not pay an
    // allocation and zero-initialization cost for every group.
    let mut parallel_inputs = if batch_size == 1 && window_parallelism > 1 {
        Some(
            (0..window_parallelism)
                .map(|_| vec![0.0; planes * window])
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    for group_start in (0..patches).step_by(group_width) {
        let group = (patches - group_start).min(group_width);
        if let Some(inputs) = parallel_inputs.as_mut() {
            for input in inputs.iter_mut().take(group) {
                input.fill(0.0);
            }
        }
        let mut batched_input = if parallel_inputs.is_none() {
            vec![0.0; group * planes * window]
        } else {
            Vec::new()
        };
        for local in 0..group {
            let patch = group_start + local;
            let frame = patch * roi;
            let source_start = frame.saturating_sub(offset);
            let destination_start = offset.saturating_sub(frame);
            let copy = (window - destination_start).min(frames - source_start);
            for plane in 0..planes {
                let source =
                    &magnitude[plane * frames + source_start..plane * frames + source_start + copy];
                let destination = if let Some(inputs) = parallel_inputs.as_mut() {
                    &mut inputs[local][plane * window + destination_start
                        ..plane * window + destination_start + copy]
                } else {
                    let offset = local * planes * window + plane * window;
                    &mut batched_input
                        [offset + destination_start..offset + destination_start + copy]
                };
                for (dst, &value) in destination.iter_mut().zip(source) {
                    *dst = value / maximum;
                }
            }
        }
        if batch_size == 1 && window_parallelism > 1 {
            // Keep each window's result separate. Flattening the vectors and
            // scanning the entire concatenation added a full-size allocation
            // and copy before the mask writeback pass.
            let predictions = parallel_inputs
                .as_ref()
                .expect("parallel buffers initialized")
                .par_iter()
                .take(group)
                .map(|input| predict(input, window, 1))
                .collect::<Result<Vec<_>>>()?;
            for (local, predicted) in predictions.iter().enumerate() {
                ensure!(
                    predicted.len() == planes * roi
                        && predicted
                            .iter()
                            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                    "network returned an invalid mask"
                );
                let patch = group_start + local;
                let frame = patch * roi;
                let keep = roi.min(frames - frame);
                for plane in 0..planes {
                    let source_start = plane * roi;
                    mask[plane * frames + frame..plane * frames + frame + keep]
                        .copy_from_slice(&predicted[source_start..source_start + keep]);
                }
                report(VrStage::Inference, patch + 1, patches)?;
            }
        } else {
            let predicted = predict(&batched_input, window, group)?;
            ensure!(
                predicted.len() == group * planes * roi
                    && predicted
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "network returned an invalid mask"
            );
            for local in 0..group {
                let patch = group_start + local;
                let frame = patch * roi;
                let keep = roi.min(frames - frame);
                for plane in 0..planes {
                    let source_start = local * planes * roi + plane * roi;
                    mask[plane * frames + frame..plane * frames + frame + keep]
                        .copy_from_slice(&predicted[source_start..source_start + keep]);
                }
                report(VrStage::Inference, patch + 1, patches)?;
            }
        }
    }
    let network_seconds = start.elapsed().as_secs_f64();
    drop(magnitude);
    report(VrStage::Reconstruction, 0, 1)?;
    let start = Instant::now();
    let [primary, residual] = spectrum.reconstruct(&mask)?;
    let reconstruction_seconds = start.elapsed().as_secs_f64();
    report(VrStage::Reconstruction, 1, 1)?;
    report(VrStage::Complete, 1, 1)?;
    Ok(VrOutput {
        primary,
        residual,
        samples_per_channel,
        sample_rate: 44100,
        timings: VrTimings {
            analysis_seconds,
            network_seconds,
            reconstruction_seconds,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_skips_network_and_keeps_resampled_length() {
        let output = separate_with(
            VrVariant::DeEcho,
            &[&[0.0; 7]],
            22050,
            VrOptions::default(),
            |_| ControlFlow::Continue(()),
            |_, _, _| panic!("silent audio must not run the network"),
        )
        .unwrap();
        assert_eq!(
            (output.samples_per_channel, output.sample_rate),
            (14, 44100)
        );
        assert_eq!(output.primary, vec![0.0; 28]);
        assert_eq!(output.residual, vec![0.0; 28]);
    }

    #[test]
    fn cancellation_stops_before_inference_and_has_a_distinct_error() {
        let mut stages = Vec::new();
        let error = separate_with(
            VrVariant::HpFive,
            &[&[0.3; 480]],
            44100,
            VrOptions::default(),
            |progress| {
                stages.push(progress.stage);
                if progress.stage == VrStage::Inference {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
            |_, _, _| panic!("cancelled task must not run inference"),
        )
        .err()
        .unwrap();
        assert!(error.is::<TaskCancelled>());
        assert!(!stages.contains(&VrStage::Complete));
    }
}
