use std::{ops::ControlFlow, time::Instant};

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

use super::RoformerModel;
use crate::{resample::Polyphase, task::TaskCancelled};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoformerStage {
    Resampling,
    Inference,
    Reconstruction,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub struct RoformerProgress {
    pub stage: RoformerStage,
    /// Number of finished windows; the current window has this zero-based index.
    pub windows_completed: usize,
    pub windows_total: usize,
    pub completed: usize,
    pub total: usize,
}

#[derive(Debug)]
pub struct RoformerTimings {
    pub resampling_seconds: f64,
    /// Includes each window's STFT, complete network and ISTFT.
    pub network_seconds: f64,
    pub reconstruction_seconds: f64,
}

pub struct RoformerOutput {
    /// Planar stereo: the complete left channel, then the right channel.
    pub vocals: Vec<f32>,
    pub instrumental: Vec<f32>,
    pub samples_per_channel: usize,
    pub sample_rate: u32,
    pub timings: RoformerTimings,
}

impl RoformerModel {
    /// Fixed 8-second / four-overlap baseline, accepting mono or stereo PCM.
    /// Progress can cancel between stages and inside each transformer window.
    pub fn separate(
        &self,
        channels: &[&[f32]],
        sample_rate: u32,
        progress: impl FnMut(RoformerProgress) -> ControlFlow<()>,
    ) -> Result<RoformerOutput> {
        // Burn Flex and the RoFormer kernels already consume the Rayon
        // pool inside every attention/linear operation. Keep window-level
        // parallelism opt-in: nesting another Rayon fan-out over full 8-second
        // windows oversubscribes the same pool and raises RSS without a stable
        // throughput gain on this model.
        let parallelism = self.options.window_parallelism;
        if parallelism == 1 {
            separate_with(channels, sample_rate, progress, |input, samples, report| {
                self.predict_window_with_progress(input, samples, report)
            })
        } else {
            separate_with_parallel(
                channels,
                sample_rate,
                parallelism,
                progress,
                |input, samples| self.predict_window(input, samples),
            )
        }
    }
}

pub(super) fn padded_length(samples: usize) -> Result<usize> {
    samples
        .max(RoformerModel::FFT / 2 + 1)
        .div_ceil(RoformerModel::HOP)
        .checked_mul(RoformerModel::HOP)
        .context("padded audio length overflow")
}

fn chunk_starts(samples: usize) -> Vec<usize> {
    let tail = samples.saturating_sub(RoformerModel::CHUNK);
    (0..tail)
        .step_by(RoformerModel::CHUNK / 4)
        .chain([tail])
        .collect()
}

pub(super) fn separate_with(
    channels: &[&[f32]],
    sample_rate: u32,
    mut progress: impl FnMut(RoformerProgress) -> ControlFlow<()>,
    mut predict: impl FnMut(
        &[f32],
        usize,
        &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Vec<f32>>,
) -> Result<RoformerOutput> {
    ensure!(
        (1..=2).contains(&channels.len()),
        "expected mono or stereo audio"
    );
    ensure!(
        !channels[0].is_empty() && channels.iter().all(|c| c.len() == channels[0].len()),
        "channels must be nonempty and equally long"
    );
    let mut emit = |stage, windows_completed, windows_total, completed, total| {
        progress(RoformerProgress {
            stage,
            windows_completed,
            windows_total,
            completed,
            total,
        })
    };
    let check = |flow: ControlFlow<()>| -> Result<()> {
        if flow.is_break() {
            Err(TaskCancelled.into())
        } else {
            Ok(())
        }
    };
    check(emit(RoformerStage::Resampling, 0, 0, 0, channels.len()))?;
    let start = Instant::now();
    let resampler = Polyphase::new(sample_rate, RoformerModel::SAMPLE_RATE)?;
    let mut audio = Vec::with_capacity(2);
    for (index, channel) in channels.iter().enumerate() {
        audio.push(resampler.process(channel)?);
        check(emit(
            RoformerStage::Resampling,
            0,
            0,
            index + 1,
            channels.len(),
        ))?;
    }
    if audio.len() == 1 {
        audio.push(audio[0].clone());
    }
    let samples = audio[0].len();
    let padded = padded_length(samples)?;
    samples.checked_mul(2).context("stereo length overflow")?;
    let resampling_seconds = start.elapsed().as_secs_f64();
    let silent = audio.iter().flatten().all(|v| *v == 0.0);
    let starts = if silent {
        Vec::new()
    } else {
        chunk_starts(padded)
    };
    let mut vocals = vec![0.0f32; 2 * samples];
    let mut counter = vec![0.0f32; samples];
    let window: Vec<_> = (0..RoformerModel::CHUNK)
        .map(|i| {
            (0.54
                - 0.46
                    * (std::f64::consts::TAU * i as f64 / (RoformerModel::CHUNK - 1) as f64).cos())
                as f32
        })
        .collect();
    check(emit(RoformerStage::Inference, 0, starts.len(), 0, 0))?;
    let mut network_seconds = 0.0;
    let mut reconstruction_seconds = 0.0;
    for (index, &offset) in starts.iter().enumerate() {
        let length = RoformerModel::CHUNK.min(padded - offset);
        let keep = length.min(samples - offset);
        let mut input = vec![0.0; 2 * length];
        for (channel, source) in audio.iter().enumerate() {
            input[channel * length..channel * length + keep]
                .copy_from_slice(&source[offset..offset + keep]);
        }
        let start = Instant::now();
        let predicted = predict(&input, length, &mut |done, count| {
            emit(RoformerStage::Inference, index, starts.len(), done, count)
        })?;
        network_seconds += start.elapsed().as_secs_f64();
        ensure!(
            predicted.len() == 2 * length && predicted.iter().all(|v| v.is_finite()),
            "invalid RoFormer window output"
        );
        check(emit(
            RoformerStage::Reconstruction,
            index,
            starts.len(),
            0,
            keep,
        ))?;
        let start = Instant::now();
        for i in 0..keep {
            for channel in 0..2 {
                vocals[channel * samples + offset + i] +=
                    predicted[channel * length + i] * window[i];
            }
            counter[offset + i] += window[i];
        }
        reconstruction_seconds += start.elapsed().as_secs_f64();
        check(emit(
            RoformerStage::Inference,
            index + 1,
            starts.len(),
            0,
            0,
        ))?;
    }
    check(emit(
        RoformerStage::Reconstruction,
        starts.len(),
        starts.len(),
        0,
        samples,
    ))?;
    let start = Instant::now();
    let mut instrumental = vec![0.0; 2 * samples];
    for (i, &weight) in counter.iter().enumerate() {
        ensure!(silent || weight > 0.0, "uncovered audio sample");
        for (channel, source) in audio.iter().enumerate() {
            let index = channel * samples + i;
            vocals[index] /= weight.max(1e-10);
            instrumental[index] = source[i] - vocals[index];
            ensure!(
                vocals[index].is_finite() && instrumental[index].is_finite(),
                "nonfinite separated audio"
            );
        }
        if i % 4096 == 0 {
            check(emit(
                RoformerStage::Reconstruction,
                starts.len(),
                starts.len(),
                i,
                samples,
            ))?;
        }
    }
    reconstruction_seconds += start.elapsed().as_secs_f64();
    check(emit(
        RoformerStage::Complete,
        starts.len(),
        starts.len(),
        samples,
        samples,
    ))?;
    Ok(RoformerOutput {
        vocals,
        instrumental,
        samples_per_channel: samples,
        sample_rate: RoformerModel::SAMPLE_RATE,
        timings: RoformerTimings {
            resampling_seconds,
            network_seconds,
            reconstruction_seconds,
        },
    })
}

/// Offline-only scheduler for independent Burn windows. Network execution is
/// parallelized in bounded groups, while overlap accumulation remains in
/// window order so floating-point output stays deterministic.
pub(super) fn separate_with_parallel(
    channels: &[&[f32]],
    sample_rate: u32,
    parallelism: usize,
    mut progress: impl FnMut(RoformerProgress) -> ControlFlow<()>,
    predict: impl Fn(&[f32], usize) -> Result<Vec<f32>> + Sync,
) -> Result<RoformerOutput> {
    ensure!(
        (2..=8).contains(&parallelism),
        "window parallelism must be 2..8"
    );
    ensure!(
        (1..=2).contains(&channels.len()),
        "expected mono or stereo audio"
    );
    ensure!(
        !channels[0].is_empty() && channels.iter().all(|c| c.len() == channels[0].len()),
        "channels must be nonempty and equally long"
    );
    let mut emit = |stage, windows_completed, windows_total, completed, total| {
        progress(RoformerProgress {
            stage,
            windows_completed,
            windows_total,
            completed,
            total,
        })
    };
    let check = |flow: ControlFlow<()>| -> Result<()> {
        if flow.is_break() {
            Err(TaskCancelled.into())
        } else {
            Ok(())
        }
    };
    check(emit(RoformerStage::Resampling, 0, 0, 0, channels.len()))?;
    let start = Instant::now();
    let resampler = Polyphase::new(sample_rate, RoformerModel::SAMPLE_RATE)?;
    let mut audio = Vec::with_capacity(2);
    for (index, channel) in channels.iter().enumerate() {
        audio.push(resampler.process(channel)?);
        check(emit(
            RoformerStage::Resampling,
            0,
            0,
            index + 1,
            channels.len(),
        ))?;
    }
    if audio.len() == 1 {
        audio.push(audio[0].clone());
    }
    let samples = audio[0].len();
    let padded = padded_length(samples)?;
    samples.checked_mul(2).context("stereo length overflow")?;
    let resampling_seconds = start.elapsed().as_secs_f64();
    let silent = audio.iter().flatten().all(|v| *v == 0.0);
    let starts = if silent {
        Vec::new()
    } else {
        chunk_starts(padded)
    };
    let mut vocals = vec![0.0f32; 2 * samples];
    let mut counter = vec![0.0f32; samples];
    let window: Vec<_> = (0..RoformerModel::CHUNK)
        .map(|i| {
            (0.54
                - 0.46
                    * (std::f64::consts::TAU * i as f64 / (RoformerModel::CHUNK - 1) as f64).cos())
                as f32
        })
        .collect();
    check(emit(RoformerStage::Inference, 0, starts.len(), 0, 0))?;
    let mut network_seconds = 0.0;
    let mut reconstruction_seconds = 0.0;
    for group_start in (0..starts.len()).step_by(parallelism) {
        let group_end = (group_start + parallelism).min(starts.len());
        let inputs: Vec<_> = (group_start..group_end)
            .map(|index| {
                let offset = starts[index];
                let length = RoformerModel::CHUNK.min(padded - offset);
                let keep = length.min(samples - offset);
                let mut input = vec![0.0; 2 * length];
                for (channel, source) in audio.iter().enumerate() {
                    input[channel * length..channel * length + keep]
                        .copy_from_slice(&source[offset..offset + keep]);
                }
                (offset, length, keep, input)
            })
            .collect();
        let start = Instant::now();
        let predicted: Result<Vec<_>> = inputs
            .par_iter()
            .map(|(_, length, _, input)| predict(input, *length))
            .collect();
        let predicted = predicted?;
        network_seconds += start.elapsed().as_secs_f64();
        for ((offset, length, keep, _), predicted) in inputs.into_iter().zip(predicted) {
            ensure!(
                predicted.len() == 2 * length && predicted.iter().all(|v| v.is_finite()),
                "invalid RoFormer window output"
            );
            check(emit(
                RoformerStage::Reconstruction,
                group_start,
                starts.len(),
                0,
                keep,
            ))?;
            let start = Instant::now();
            for i in 0..keep {
                for channel in 0..2 {
                    vocals[channel * samples + offset + i] +=
                        predicted[channel * length + i] * window[i];
                }
                counter[offset + i] += window[i];
            }
            reconstruction_seconds += start.elapsed().as_secs_f64();
            check(emit(
                RoformerStage::Inference,
                group_start + 1,
                starts.len(),
                0,
                0,
            ))?;
        }
    }
    check(emit(
        RoformerStage::Reconstruction,
        starts.len(),
        starts.len(),
        0,
        samples,
    ))?;
    let start = Instant::now();
    let mut instrumental = vec![0.0; 2 * samples];
    for (i, &weight) in counter.iter().enumerate() {
        ensure!(silent || weight > 0.0, "uncovered audio sample");
        for (channel, source) in audio.iter().enumerate() {
            let index = channel * samples + i;
            vocals[index] /= weight.max(1e-10);
            instrumental[index] = source[i] - vocals[index];
            ensure!(
                vocals[index].is_finite() && instrumental[index].is_finite(),
                "nonfinite separated audio"
            );
        }
        if i % 4096 == 0 {
            check(emit(
                RoformerStage::Reconstruction,
                starts.len(),
                starts.len(),
                i,
                samples,
            ))?;
        }
    }
    reconstruction_seconds += start.elapsed().as_secs_f64();
    check(emit(
        RoformerStage::Complete,
        starts.len(),
        starts.len(),
        1,
        1,
    ))?;
    Ok(RoformerOutput {
        vocals,
        instrumental,
        samples_per_channel: samples,
        sample_rate: RoformerModel::SAMPLE_RATE,
        timings: RoformerTimings {
            resampling_seconds,
            network_seconds,
            reconstruction_seconds,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_covers_boundaries_without_duplicate_tail() {
        for (samples, expected) in [
            (1323, vec![0]),
            (352800, vec![0]),
            (353241, vec![0, 441]),
            (441000, vec![0, 88200]),
            (441441, vec![0, 88200, 88641]),
        ] {
            assert_eq!(chunk_starts(samples), expected);
        }
        assert_eq!(padded_length(1).unwrap(), 1323);
        assert_eq!(padded_length(4411).unwrap(), 4851);
        assert!(padded_length(usize::MAX).is_err());
    }

    #[test]
    fn preserves_short_resampled_audio_and_skips_silence() {
        let result = separate_with(
            &[&[0.0; 7]],
            22050,
            |_| ControlFlow::Continue(()),
            |_, _, _| panic!("silence must skip the network"),
        )
        .unwrap();
        assert_eq!(result.vocals, vec![0.0; 28]);
        assert_eq!(result.instrumental, vec![0.0; 28]);
        let result = separate_with(
            &[&[0.5]],
            44100,
            |_| ControlFlow::Continue(()),
            |audio, samples, _| {
                assert_eq!(samples, 1323);
                assert_eq!(audio[0], 0.5);
                assert_eq!(audio[1323], 0.5);
                Ok(audio.iter().map(|v| v * 0.25).collect())
            },
        )
        .unwrap();
        assert_eq!(result.vocals, vec![0.125; 2]);
        assert_eq!(result.instrumental, vec![0.375; 2]);
    }

    #[test]
    fn cancellation_rejects_partial_output() {
        let result = separate_with(
            &[&[0.2; 2048]],
            44100,
            |p| {
                if p.stage == RoformerStage::Inference {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
            |_, _, _| panic!("must not enter cancelled inference"),
        );
        assert!(result.err().unwrap().is::<TaskCancelled>());
    }

    #[test]
    fn parallel_windows_preserve_serial_overlap_and_tail() {
        let wave = (0..RoformerModel::CHUNK + 1001)
            .map(|i| (i % 127) as f32 / 127.0 - 0.5)
            .collect::<Vec<_>>();
        let serial = separate_with(
            &[&wave],
            44100,
            |_| ControlFlow::Continue(()),
            |audio, _, _| Ok(audio.iter().map(|sample| sample * 0.25).collect()),
        )
        .unwrap();
        let parallel = separate_with_parallel(
            &[&wave],
            44100,
            2,
            |_| ControlFlow::Continue(()),
            |audio, _| Ok(audio.iter().map(|sample| sample * 0.25).collect()),
        )
        .unwrap();
        assert_eq!(serial.samples_per_channel, wave.len());
        assert_eq!(parallel.vocals, serial.vocals);
        assert_eq!(parallel.instrumental, serial.instrumental);
    }
}
