//! Shared file-to-file tasks for CLI and desktop callers.

use std::{
    ffi::OsString,
    ops::ControlFlow,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};

use crate::{
    audio_io,
    roformer::{RoformerModel, RoformerOptions, RoformerOutput, RoformerProgress, RoformerStage},
    runtime::{RuntimeBackend, RuntimeOptions},
    task::TaskCancelled,
    vr::{VrOptions, VrSeparator, VrStage},
    vr_dsp::VrVariant,
};

#[derive(Debug, Clone, Copy)]
pub enum ModelSpec {
    Vr {
        variant: VrVariant,
        options: VrOptions,
    },
    Roformer1296,
}

impl ModelSpec {
    pub fn from_key(key: &str) -> Option<Self> {
        let variant = match key {
            "1296" => return Some(Self::Roformer1296),
            "5hp" => VrVariant::HpFive,
            "6hp" => VrVariant::HpSix,
            "deecho" => VrVariant::DeEcho,
            _ => return None,
        };
        Some(Self::Vr {
            variant,
            options: VrOptions::default(),
        })
    }

    pub fn weights_name(self) -> &'static str {
        match self {
            Self::Vr {
                variant: VrVariant::HpFive,
                ..
            } => "5_HP-Karaoke-UVR.pth",
            Self::Vr {
                variant: VrVariant::HpSix,
                ..
            } => "6_HP-Karaoke-UVR.pth",
            Self::Vr {
                variant: VrVariant::DeEcho,
                ..
            } => "UVR-DeEcho-DeReverb.pth",
            Self::Roformer1296 => "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::Vr {
                variant: VrVariant::HpFive,
                ..
            } => "5hp",
            Self::Vr {
                variant: VrVariant::HpSix,
                ..
            } => "6hp",
            Self::Vr {
                variant: VrVariant::DeEcho,
                ..
            } => "deecho",
            Self::Roformer1296 => "1296",
        }
    }

    pub fn output_kinds(self) -> [&'static str; 2] {
        match self {
            Self::Roformer1296 => ["vocals", "instrumental"],
            Self::Vr { .. } => ["primary", "residual"],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStage {
    Decode,
    LoadModel,
    Analysis,
    Inference,
    Reconstruction,
    Encode,
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileProgress {
    pub stage: FileStage,
    pub completed: usize,
    pub total: usize,
    pub windows_completed: usize,
    pub windows_total: usize,
}

impl FileProgress {
    fn stage(stage: FileStage) -> Self {
        Self {
            stage,
            completed: 0,
            total: 0,
            windows_completed: 0,
            windows_total: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct FileTimings {
    pub decode_seconds: f64,
    pub load_seconds: f64,
    pub analysis_seconds: f64,
    /// RoFormer includes the per-window STFT and ISTFT here.
    pub network_seconds: f64,
    pub reconstruction_seconds: f64,
    pub encode_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug)]
pub struct FileOutput {
    pub paths: [PathBuf; 2],
    pub kinds: [&'static str; 2],
    pub sample_rate: u32,
    pub samples_per_channel: usize,
    pub timings: FileTimings,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RoformerBackend {
    #[default]
    Burn,
    /// Requires the optional OpenVINO build feature and native runtime.
    OpenvinoCpu { threads: usize },
}

enum RoformerEngine {
    Burn(Box<RoformerModel>),
    #[cfg(feature = "openvino")]
    Openvino(Box<crate::roformer::openvino::OpenvinoRoformer>),
}

impl RoformerEngine {
    fn separate(
        &mut self,
        channels: &[&[f32]],
        sample_rate: u32,
        progress: impl FnMut(RoformerProgress) -> ControlFlow<()>,
    ) -> Result<RoformerOutput> {
        match self {
            Self::Burn(model) => model.separate(channels, sample_rate, progress),
            #[cfg(feature = "openvino")]
            Self::Openvino(model) => model.separate(channels, sample_rate, progress),
        }
    }
}

enum Network {
    Vr(Box<VrSeparator>, VrOptions),
    Roformer(RoformerEngine),
}

fn check(flow: ControlFlow<()>) -> Result<()> {
    if flow.is_break() {
        Err(TaskCancelled.into())
    } else {
        Ok(())
    }
}

/// Opens the input, validates original weights, separates and publishes two WAVs.
/// Callers own the thread and cancellation state; returning Break stops the task.
/// If publishing the second track fails, the error identifies the saved first track.
pub fn separate_file(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    progress: impl FnMut(FileProgress) -> ControlFlow<()>,
) -> Result<FileOutput> {
    separate_file_with_backend(
        spec,
        weights,
        input,
        directory,
        RoformerBackend::Burn,
        progress,
    )
}

/// Explicit backend selection for the 1296 CPU experiment; defaults stay measured separately.
pub fn separate_file_with_backend(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    backend: RoformerBackend,
    progress: impl FnMut(FileProgress) -> ControlFlow<()>,
) -> Result<FileOutput> {
    separate_file_impl(spec, weights, input, directory, backend, None, progress)
}

/// Each task owns its CPU thread budget, so GUI jobs can change settings without
/// mutating process-wide environment variables or a previously initialized pool.
pub fn separate_file_with_options(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    runtime: RuntimeOptions,
    progress: impl FnMut(FileProgress) -> ControlFlow<()> + Send,
) -> Result<FileOutput> {
    let runtime = runtime.effective_for(spec)?;
    let spec = match spec {
        ModelSpec::Vr { variant, .. } => ModelSpec::Vr {
            variant,
            options: runtime.vr,
        },
        ModelSpec::Roformer1296 => ModelSpec::Roformer1296,
    };
    let backend = match runtime.backend {
        RuntimeBackend::Burn => RoformerBackend::Burn,
        RuntimeBackend::OpenvinoCpu => RoformerBackend::OpenvinoCpu {
            threads: runtime.threads,
        },
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(runtime.threads)
        .build()
        .context("cannot create inference thread pool")?
        .install(|| {
            separate_file_impl(
                spec,
                weights,
                input,
                directory,
                backend,
                Some(runtime.roformer),
                progress,
            )
        })
}

fn separate_file_impl(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    backend: RoformerBackend,
    roformer: Option<RoformerOptions>,
    mut progress: impl FnMut(FileProgress) -> ControlFlow<()>,
) -> Result<FileOutput> {
    anyhow::ensure!(
        matches!(spec, ModelSpec::Roformer1296) || backend == RoformerBackend::Burn,
        "OpenVINO backend is only implemented for 1296"
    );
    #[cfg(not(feature = "openvino"))]
    anyhow::ensure!(
        backend == RoformerBackend::Burn,
        "this build does not enable the OpenVINO CPU backend"
    );
    let stem = input.file_stem().context("input path has no filename")?;
    let kinds = spec.output_kinds();
    let paths = kinds.map(|kind| {
        let mut name = OsString::from(stem);
        name.push(format!("_{}_{kind}.wav", spec.key()));
        directory.join(name)
    });
    for path in &paths {
        audio_io::ensure_output_absent(path)?;
    }
    check(progress(FileProgress::stage(FileStage::Decode)))?;
    std::fs::create_dir_all(directory).context("cannot create output directory")?;
    let total = Instant::now();
    let start = Instant::now();
    let audio = audio_io::decode(input, || {
        progress(FileProgress::stage(FileStage::Decode)).is_continue()
    })?;
    let mut timings = FileTimings {
        decode_seconds: start.elapsed().as_secs_f64(),
        ..Default::default()
    };
    check(progress(FileProgress::stage(FileStage::LoadModel)))?;
    let start = Instant::now();
    let mut model = match spec {
        ModelSpec::Vr { variant, options } => {
            Network::Vr(Box::new(VrSeparator::load(variant, weights)?), options)
        }
        ModelSpec::Roformer1296 => Network::Roformer(match backend {
            RoformerBackend::Burn => RoformerEngine::Burn(Box::new(match roformer {
                Some(options) => RoformerModel::load_with_options(weights, options)?,
                None => RoformerModel::load(weights)?,
            })),
            RoformerBackend::OpenvinoCpu { threads } => {
                #[cfg(feature = "openvino")]
                {
                    RoformerEngine::Openvino(Box::new(
                        crate::roformer::openvino::OpenvinoRoformer::load_for_audio(
                            weights,
                            audio.channels[0].len(),
                            audio.sample_rate,
                            threads,
                            || progress(FileProgress::stage(FileStage::LoadModel)).is_continue(),
                        )?,
                    ))
                }
                #[cfg(not(feature = "openvino"))]
                {
                    let _ = threads;
                    anyhow::bail!("this build does not enable the OpenVINO CPU backend")
                }
            }
        }),
    };
    timings.load_seconds = start.elapsed().as_secs_f64();
    check(progress(FileProgress::stage(FileStage::Analysis)))?;
    let channels: Vec<_> = audio.channels.iter().map(Vec::as_slice).collect();
    let (stems, sample_rate, samples) = match &mut model {
        Network::Vr(model, options) => {
            let output = model.separate(&channels, audio.sample_rate, *options, |p| {
                progress(FileProgress {
                    stage: match p.stage {
                        VrStage::Analysis => FileStage::Analysis,
                        VrStage::Inference => FileStage::Inference,
                        VrStage::Reconstruction | VrStage::Complete => FileStage::Reconstruction,
                    },
                    completed: p.completed,
                    total: p.total,
                    windows_completed: if p.stage == VrStage::Inference {
                        p.completed
                    } else {
                        0
                    },
                    windows_total: if p.stage == VrStage::Inference {
                        p.total
                    } else {
                        0
                    },
                })
            })?;
            timings.analysis_seconds = output.timings.analysis_seconds;
            timings.network_seconds = output.timings.network_seconds;
            timings.reconstruction_seconds = output.timings.reconstruction_seconds;
            (
                [output.primary, output.residual],
                output.sample_rate,
                output.samples_per_channel,
            )
        }
        Network::Roformer(model) => {
            let output = model.separate(&channels, audio.sample_rate, |p| {
                progress(FileProgress {
                    stage: match p.stage {
                        RoformerStage::Resampling => FileStage::Analysis,
                        RoformerStage::Inference => FileStage::Inference,
                        RoformerStage::Reconstruction | RoformerStage::Complete => {
                            FileStage::Reconstruction
                        }
                    },
                    completed: p.completed,
                    total: p.total,
                    windows_completed: p.windows_completed,
                    windows_total: p.windows_total,
                })
            })?;
            timings.analysis_seconds = output.timings.resampling_seconds;
            timings.network_seconds = output.timings.network_seconds;
            timings.reconstruction_seconds = output.timings.reconstruction_seconds;
            (
                [output.vocals, output.instrumental],
                output.sample_rate,
                output.samples_per_channel,
            )
        }
    };
    drop(model);
    drop(audio);
    check(progress(FileProgress::stage(FileStage::Encode)))?;
    let start = Instant::now();
    let prepare =
        |path: &Path, values: &[f32], progress: &mut dyn FnMut(FileProgress) -> ControlFlow<()>| {
            let (left, right) = values.split_at(samples);
            audio_io::prepare_wav(path, &[left, right], sample_rate, || {
                progress(FileProgress::stage(FileStage::Encode)).is_continue()
            })
        };
    let first = prepare(&paths[0], &stems[0], &mut progress)?;
    let second = prepare(&paths[1], &stems[1], &mut progress)?;
    check(progress(FileProgress::stage(FileStage::Encode)))?;
    first.persist()?;
    second.persist().with_context(|| {
        format!(
            "first track saved to {}; publishing second track failed",
            paths[0].display()
        )
    })?;
    timings.encode_seconds = start.elapsed().as_secs_f64();
    timings.total_seconds = total.elapsed().as_secs_f64();
    // Both outputs have been published. A later cancellation cannot undo completion.
    let _ = progress(FileProgress {
        completed: 1,
        total: 1,
        ..FileProgress::stage(FileStage::Complete)
    });
    Ok(FileOutput {
        paths,
        kinds,
        sample_rate,
        samples_per_channel: samples,
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consecutive_tasks_use_their_own_thread_budget() {
        let directory = tempfile::tempdir().unwrap();
        for threads in [1, 3] {
            let mut observed_threads = 0;
            let error = separate_file_with_options(
                ModelSpec::from_key("5hp").unwrap(),
                Path::new("unused.pth"),
                Path::new("unused.wav"),
                directory.path(),
                RuntimeOptions {
                    threads,
                    ..Default::default()
                },
                |_| {
                    observed_threads = rayon::current_num_threads();
                    ControlFlow::Break(())
                },
            )
            .unwrap_err();
            assert!(error.is::<TaskCancelled>());
            assert_eq!(observed_threads, threads);
        }
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn invalid_explicit_options_fail_before_file_work() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("must-not-be-created");
        let mut progressed = false;
        let mut runtime = RuntimeOptions::default();
        runtime.roformer.frequency_batch = 0;
        let error = separate_file_with_options(
            ModelSpec::Roformer1296,
            Path::new("missing.ckpt"),
            Path::new("missing.wav"),
            &output,
            runtime,
            |_| {
                progressed = true;
                ControlFlow::Continue(())
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("frequency batch"));
        assert!(!progressed);
        assert!(!output.exists());
    }
}
