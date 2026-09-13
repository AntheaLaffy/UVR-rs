//! Optional FP32 CPU backend built directly from the original 1296 checkpoint.
//! GPU remains in the development probe until runtime stability is resolved.

use std::{collections::BTreeMap, ops::ControlFlow, path::Path, time::Instant};

use ::openvino::{
    CompiledModel, Core, DeviceType, ElementType, InferRequest, InferenceErrorKind, Model,
    PropertyKey, RwPropertyKey, Shape, Tensor,
};
use anyhow::{Context, Result, ensure};

use super::{RoformerModel, RoformerOutput, RoformerProgress, audio, stft::PreciseStft};
use crate::{dsp, resample::Polyphase, task::TaskCancelled};

mod ir;
pub use ir::RoformerIr;

/// Portable builds carry native libraries beside the executable. Explicit
/// loading keeps task setup independent of process-wide search-path variables.
pub(crate) fn create_core() -> Result<Core> {
    #[cfg(target_os = "linux")]
    if let Some(directory) = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("lib")))
    {
        let library = directory.join("libopenvino_c.so");
        if library.is_file() {
            openvino_sys::library::load_from(&library)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("cannot load bundled OpenVINO: {}", library.display()))?;
        }
    }
    Core::new().context("cannot initialize OpenVINO runtime")
}

#[derive(Debug, Clone)]
pub struct OpenvinoInfo {
    pub checkpoint_sha256: String,
    pub frames: usize,
    pub graph_seconds: f64,
    pub compile_seconds: f64,
    pub actual_settings: BTreeMap<String, String>,
}

pub struct OpenvinoRoformer {
    request: InferRequest,
    input: Tensor,
    analysis: PreciseStft,
    inverse: dsp::Stft,
    info: OpenvinoInfo,
    // Keep native owners until the request is dropped; weights are never mutated.
    _compiled: CompiledModel,
    _model: Model,
    _weights: Tensor,
    _core: Core,
}

fn check(flow: ControlFlow<()>) -> Result<()> {
    if flow.is_break() {
        Err(TaskCancelled.into())
    } else {
        Ok(())
    }
}

impl OpenvinoRoformer {
    /// All windows of one file use this shape under the fixed scheduler.
    pub fn load_for_audio(
        checkpoint: &Path,
        input_samples: usize,
        sample_rate: u32,
        threads: usize,
        keep_going: impl FnMut() -> bool,
    ) -> Result<Self> {
        ensure!(input_samples > 0, "empty audio");
        let length =
            Polyphase::new(sample_rate, RoformerModel::SAMPLE_RATE)?.output_len(input_samples)?;
        let window = audio::padded_length(length)?.min(RoformerModel::CHUNK);
        Self::load_for_frames(
            checkpoint,
            window / RoformerModel::HOP + 1,
            threads,
            keep_going,
        )
    }

    /// Native initialization and compilation are measured separately from inference.
    pub fn load_for_frames(
        checkpoint: &Path,
        frames: usize,
        threads: usize,
        mut keep_going: impl FnMut() -> bool,
    ) -> Result<Self> {
        ensure!(threads > 0, "thread count must be positive");
        let start = Instant::now();
        let graph = RoformerIr::from_checkpoint(checkpoint, frames, &mut keep_going)?;
        let graph_seconds = start.elapsed().as_secs_f64();
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        let mut core = create_core()?;
        for (key, value) in [
            (RwPropertyKey::HintInferencePrecision, "f32".to_owned()),
            (RwPropertyKey::HintPerformanceMode, "LATENCY".to_owned()),
            (RwPropertyKey::NumStreams, "1".to_owned()),
            (RwPropertyKey::InferenceNumThreads, threads.to_string()),
        ] {
            core.set_property(&DeviceType::CPU, &key, &value)?;
        }
        let start = Instant::now();
        let mut weights = Tensor::new(
            ElementType::U8,
            &Shape::new(&[i64::try_from(graph.weights.len())?])?,
        )?;
        weights
            .get_data_mut::<u8>()?
            .copy_from_slice(&graph.weights);
        let model = core.read_model_from_buffer(&graph.xml, Some(&weights))?;
        // Release the Rust staging bytes before compilation allocates working memory.
        drop(graph.weights);
        drop(graph.xml);
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        let mut compiled = core.compile_model(&model, DeviceType::CPU)?;
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        let mut request = compiled.create_infer_request()?;
        let shape = [1, i64::try_from(frames)?, 4100];
        ensure!(
            request.get_input_tensor()?.get_shape()?.get_dimensions() == shape,
            "unexpected input shape"
        );
        let input = Tensor::new(ElementType::F32, &Shape::new(&shape)?)?;
        request.set_input_tensor(&input)?;
        let mut actual_settings = BTreeMap::new();
        for name in [
            "INFERENCE_PRECISION_HINT",
            "INFERENCE_NUM_THREADS",
            "NUM_STREAMS",
            "ENABLE_CPU_PINNING",
            "ENABLE_HYPER_THREADING",
            "SCHEDULING_CORE_TYPE",
            "EXECUTION_DEVICES",
        ] {
            actual_settings.insert(
                name.into(),
                compiled
                    .get_property(&PropertyKey::Other(name.into()))?
                    .into_owned(),
            );
        }
        ensure!(
            actual_settings["INFERENCE_PRECISION_HINT"] == "f32",
            "runtime did not select FP32"
        );
        let info = OpenvinoInfo {
            checkpoint_sha256: graph.checkpoint_sha256,
            frames,
            graph_seconds,
            compile_seconds: start.elapsed().as_secs_f64(),
            actual_settings,
        };
        Ok(Self {
            request,
            input,
            analysis: PreciseStft::new(),
            inverse: dsp::Stft::new(
                RoformerModel::FFT,
                RoformerModel::HOP,
                dsp::Padding::Reflect,
            )?,
            info,
            _compiled: compiled,
            _model: model,
            _weights: weights,
            _core: core,
        })
    }

    pub fn info(&self) -> &OpenvinoInfo {
        &self.info
    }

    /// Uses the same resampling, overlap, tail handling and residual as Burn.
    pub fn separate(
        &mut self,
        channels: &[&[f32]],
        sample_rate: u32,
        progress: impl FnMut(RoformerProgress) -> ControlFlow<()>,
    ) -> Result<RoformerOutput> {
        audio::separate_with(channels, sample_rate, progress, |input, samples, report| {
            self.predict_window_with_progress(input, samples, report)
        })
    }

    /// One complete-hop stereo window. The request and FFT plans are reused.
    pub fn predict_window_with_progress(
        &mut self,
        wave: &[f32],
        samples: usize,
        mut progress: impl FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<Vec<f32>> {
        ensure!(
            (1323..=RoformerModel::CHUNK).contains(&samples)
                && samples.is_multiple_of(RoformerModel::HOP)
                && wave.len() == 2 * samples,
            "expected a complete-hop stereo window"
        );
        ensure!(wave.iter().all(|v| v.is_finite()), "nonfinite input");
        let frames = samples / RoformerModel::HOP + 1;
        ensure!(
            frames == self.info.frames,
            "audio window differs from compiled shape"
        );
        check(progress(0, 5))?;
        let mut spectra = Vec::with_capacity(2);
        for (index, channel) in wave.chunks_exact(samples).enumerate() {
            spectra.push(self.analysis.forward(channel));
            check(progress(index + 1, 5))?;
        }
        let features = self.input.get_data_mut::<f32>()?;
        for frame in 0..frames {
            for bin in 0..1025 {
                for (channel, spectrum) in spectra.iter().enumerate() {
                    let value = spectrum.values[bin * frames + frame];
                    let index = frame * 4100 + bin * 4 + channel * 2;
                    features[index] = value.re;
                    features[index + 1] = value.im;
                }
            }
        }
        ensure!(features.iter().all(|v| v.is_finite()), "nonfinite spectrum");
        check(progress(2, 5))?;
        self.request.infer_async()?;
        loop {
            match self.request.wait(50) {
                Ok(()) => break,
                Err(error) if error.kind == InferenceErrorKind::ResultNotReady => {
                    if progress(2, 5).is_break() {
                        self.request.cancel()?;
                        // The binding exposes wait_for, whose timeout is a nonnegative
                        // duration; -1 is not the separate C API's unbounded wait.
                        loop {
                            match self.request.wait(50) {
                                Ok(()) => break,
                                Err(error) if error.kind == InferenceErrorKind::InferCancelled => {
                                    break;
                                }
                                Err(error) if error.kind == InferenceErrorKind::ResultNotReady => {}
                                Err(error) => return Err(error.into()),
                            }
                        }
                        return Err(TaskCancelled.into());
                    }
                }
                Err(error) => return Err(error.into()),
            }
        }
        check(progress(3, 5))?;
        let mask_tensor = self.request.get_output_tensor()?;
        ensure!(
            mask_tensor.get_element_type()? == ElementType::F32,
            "expected FP32 mask"
        );
        let mask = mask_tensor.get_data::<f32>()?;
        ensure!(
            mask.len() == frames * 4100 && mask.iter().all(|v| v.is_finite()),
            "invalid mask"
        );
        for (channel, spectrum) in spectra.iter_mut().enumerate() {
            for bin in 0..1025 {
                for frame in 0..frames {
                    let index = frame * 4100 + bin * 4 + channel * 2;
                    spectrum.values[bin * frames + frame] *=
                        dsp::Complex32::new(mask[index], mask[index + 1]);
                }
            }
        }
        let mut output = Vec::with_capacity(2 * samples);
        for (index, spectrum) in spectra.iter().enumerate() {
            output.extend(self.inverse.inverse(spectrum, None)?);
            check(progress(4 + index, 5))?;
        }
        ensure!(
            output.len() == 2 * samples && output.iter().all(|v| v.is_finite()),
            "invalid reconstructed window"
        );
        Ok(output)
    }
}
