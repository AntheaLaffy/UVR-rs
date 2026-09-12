//! Development IR experiment; loading original weights is a separate product gate.

use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use openvino::{Core, DeviceType, ElementType, PropertyKey, RwPropertyKey, Shape, Tensor};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uvr_backend_probe::Probe;
use uvr_core::{dsp, roformer::RoformerModel};

// Exercise the same FP64 analysis as the product, without loading a second network.
#[path = "../../../../core/src/roformer/stft.rs"]
mod precise_stft;

fn digest(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn waveform_checks(actual: &[f32], expected: &[f32], stem_size: usize) -> (bool, Vec<Value>) {
    let mut passed = actual.len() == expected.len();
    let mut checks = Vec::new();
    for (index, (output, reference)) in actual
        .chunks_exact(stem_size)
        .zip(expected.chunks_exact(stem_size))
        .enumerate()
    {
        let mut maximum = 0.0_f64;
        let mut squared_error = 0.0;
        let mut energy = 0.0;
        let mut failed = 0;
        for (&a, &b) in output.iter().zip(reference) {
            let error = f64::from(a) - f64::from(b);
            maximum = maximum.max(error.abs());
            squared_error += error * error;
            energy += f64::from(b).powi(2);
            if !a.is_finite() || error.abs() > 2e-4 + 2e-3 * f64::from(b).abs() {
                failed += 1;
            }
        }
        let rmse = (squared_error / stem_size as f64).sqrt();
        let reference_rms = (energy / stem_size as f64).sqrt();
        let rms_limit = 1e-3 * reference_rms.max(1e-4);
        passed &= failed == 0 && rmse.is_finite() && rmse <= rms_limit;
        checks.push(json!({"stem": index, "max_absolute_error": maximum,
            "rmse": rmse, "reference_rms": reference_rms, "rms_limit": rms_limit,
            "failed_samples": failed}));
    }
    (passed, checks)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() >= 5, "missing probe arguments");
    let check_only = args.iter().skip(5).any(|arg| arg == "--check-only");
    let batch = args
        .iter()
        .position(|arg| arg == "--batch")
        .map(|index| {
            ensure!(index + 1 < args.len(), "--batch needs a value");
            args[index + 1].parse::<usize>().map_err(anyhow::Error::from)
        })
        .transpose()?
        .unwrap_or(1);
    ensure!((1..=8).contains(&batch), "batch must be 1..8");
    ensure!(
        args.iter().skip(5).all(|arg| arg == "--check-only" || arg == "--batch"
            || arg.parse::<usize>().is_ok()),
        "unknown optional argument"
    );
    ensure!(
        matches!(args[0].as_str(), "CPU" | "GPU"),
        "choose CPU or GPU"
    );
    let device = if args[0] == "CPU" {
        DeviceType::CPU
    } else {
        DeviceType::GPU
    };
    let ir = Path::new(&args[1]);
    let root = Path::new(&args[2]);
    let output = Path::new(&args[4]);
    ensure!(!output.exists(), "report already exists");
    let mut trace = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("jsonl"))?;
    let runtime_environment: BTreeMap<_, _> = [
        "PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "RAYON_NUM_THREADS",
        "OCL_ICD_VENDORS",
        "UVR_OV_COMPILATION_THREADS",
        "UVR_OV_CPU_PINNING",
        "UVR_OV_HYPER_THREADING",
        "UVR_OV_CORE_TYPE",
        "UVR_OV_PERFORMANCE",
        "UVR_OV_STREAMS",
        "NEOReadDebugKeys",
        "EnableDirectSubmission",
        "CLI_OpenCLFileName",
        "CLI_InOrderQueue",
        "CLI_QueueInfoLogging",
        "CLI_DumpDir",
        "CLI_UniqueFiles",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok().map(|value| (name, value)))
    .collect();
    let manifest_raw = std::fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&manifest_raw)?;
    let provenance_path = ir.with_extension("provenance.json");
    let provenance: Option<Value> = if provenance_path.exists() {
        let metadata: Value = serde_json::from_slice(&std::fs::read(&provenance_path)?)?;
        ensure!(
            metadata["xml_sha256"] == digest(ir)?
                && metadata["bin_sha256"] == digest(&ir.with_extension("bin"))?,
            "IR provenance hash mismatch"
        );
        ensure!(
            metadata["checkpoint_sha256"].as_str().is_some_and(|hash| {
                manifest["sources"]
                    .as_array()
                    .is_some_and(|sources| sources.iter().any(|source| source["sha256"] == hash))
            }),
            "IR provenance and reference checkpoint differ"
        );
        Some(metadata)
    } else {
        None
    };
    ensure!(
        manifest["sources"]
            .as_array()
            .context("missing sources")?
            .iter()
            .any(|source| source["stft_precision"]
                .as_str()
                .is_some_and(|value| value.contains("float64"))),
        "fixture must use baseline FP64 forward STFT"
    );
    let probe = Probe::load_paths(root.to_path_buf(), output.to_path_buf())?;
    let case = probe
        .cases
        .iter()
        .find(|case| case.name == args[3])
        .context("case not found")?;
    let input = &probe.tensors[&case.input];
    let expected = &probe.tensors[&case.expected];
    ensure!(
        input.shape.len() == 2 && matches!(input.shape[0], 1 | 2),
        "expected mono/stereo PCM"
    );
    let samples = input.shape[1];
    ensure!(
        (1323..=RoformerModel::CHUNK).contains(&samples) && samples % RoformerModel::HOP == 0,
        "requires a complete-hop single window, 1323..352800 samples"
    );
    let is_audio = manifest.get("audio").is_some();
    if is_audio {
        let params = &manifest["audio"]["cases"][&case.name];
        ensure!(
            params["sample_rate"] == 44100 && params["windows"] == 1,
            "only one window at 44100 Hz is supported"
        );
        ensure!(
            expected.shape == [2, 2, samples],
            "invalid two-stem fixture"
        );
    } else {
        ensure!(expected.shape == [2, samples], "invalid raw-window fixture");
    }
    let wave = if input.shape[0] == 1 {
        input.values.repeat(2)
    } else {
        input.values.clone()
    };
    let frames = samples / RoformerModel::HOP + 1;
    let threads: usize = std::env::var("RAYON_NUM_THREADS")?.parse()?;
    ensure!(threads > 0, "thread budget must be positive");
    let performance = std::env::var("UVR_OV_PERFORMANCE").unwrap_or_else(|_| "LATENCY".to_owned());
    ensure!(matches!(performance.as_str(), "LATENCY" | "THROUGHPUT"), "invalid UVR_OV_PERFORMANCE");
    let streams = std::env::var("UVR_OV_STREAMS").unwrap_or_else(|_| "1".to_owned());
    ensure!(streams.parse::<usize>()? > 0, "invalid UVR_OV_STREAMS");
    let mut settings = vec![
        (RwPropertyKey::HintInferencePrecision, "f32".to_owned()),
        (RwPropertyKey::HintPerformanceMode, performance),
        (RwPropertyKey::NumStreams, streams),
    ];
    if args[0] == "CPU" {
        settings.push((RwPropertyKey::InferenceNumThreads, threads.to_string()));
        for (name, key, allowed) in [
            (
                "UVR_OV_CPU_PINNING",
                RwPropertyKey::HintEnableCpuPinning,
                &["YES", "NO"][..],
            ),
            (
                "UVR_OV_HYPER_THREADING",
                RwPropertyKey::HintEnableHyperThreading,
                &["YES", "NO"][..],
            ),
            (
                "UVR_OV_CORE_TYPE",
                RwPropertyKey::HintSchedulingCoreType,
                &["ANY_CORE", "PCORE_ONLY", "ECORE_ONLY"][..],
            ),
        ] {
            if let Ok(value) = std::env::var(name) {
                ensure!(allowed.contains(&value.as_str()), "invalid {name}");
                settings.push((key, value));
            }
        }
    }
    if let Ok(value) = std::env::var("UVR_OV_COMPILATION_THREADS") {
        ensure!(
            value.parse::<usize>()? > 0,
            "compilation threads must be positive"
        );
        settings.push((
            RwPropertyKey::Other("COMPILATION_NUM_THREADS".into()),
            value,
        ));
    }
    let requested_settings: BTreeMap<_, _> = settings
        .iter()
        .map(|(key, value)| (key.as_ref().to_owned(), value.clone()))
        .collect();
    let start = Instant::now();
    let mut core = Core::new()?;
    let available = format!("{:?}", core.available_devices()?);
    let device_name = core.get_property(&device, &PropertyKey::DeviceFullName)?;
    let versions: Vec<_> = core
        .versions(&args[0])?
        .into_iter()
        .map(|(device, version)| {
            json!({"device": format!("{device:?}"), "build_number": version.build_number,
            "description": version.description})
        })
        .collect();
    for (key, value) in &settings {
        core.set_property(&device, key, value)?;
    }
    let model = core.read_model_from_file(
        ir.to_str().context("non-UTF8 IR")?,
        ir.with_extension("bin")
            .to_str()
            .context("non-UTF8 weights")?,
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    serde_json::to_writer(
        &mut trace,
        &json!({"event": "before_compile", "device": args[0],
        "versions": versions, "requested_settings": requested_settings,
        "runtime_environment": runtime_environment, "ir_provenance": provenance}),
    )?;
    trace.write_all(b"\n")?;
    trace.flush()?;
    eprintln!(
        "{device_name}: IR loaded in {load_seconds:.3}s; compiling with {requested_settings:?}"
    );
    let start = Instant::now();
    let mut compiled = core.compile_model(&model, device)?;
    let mut request = compiled.create_infer_request()?;
    let compile_seconds = start.elapsed().as_secs_f64();
    let dimensions = [i64::try_from(batch)?, i64::try_from(frames)?, 4100];
    let mut input_tensor = Tensor::new(ElementType::F32, &Shape::new(&dimensions)?)?;
    ensure!(
        request.get_input_tensor()?.get_shape()?.get_dimensions() == dimensions,
        "IR shape does not match this window; use a separately validated shape"
    );
    request.set_input_tensor(&input_tensor)?;
    let mut actual_settings = BTreeMap::new();
    let supported = compiled
        .get_property(&PropertyKey::SupportedProperties)?
        .into_owned();
    for name in [
        "INFERENCE_PRECISION_HINT",
        "NUM_STREAMS",
        "INFERENCE_NUM_THREADS",
        "ENABLE_CPU_PINNING",
        "SCHEDULING_CORE_TYPE",
        "ENABLE_HYPER_THREADING",
        "EXECUTION_DEVICES",
    ] {
        if supported
            .split_whitespace()
            .any(|property| property == name)
        {
            actual_settings.insert(
                name,
                compiled
                    .get_property(&PropertyKey::Other(name.into()))?
                    .into_owned(),
            );
        }
    }
    eprintln!("{device_name}: IR loaded in {load_seconds:.3}s, compiled in {compile_seconds:.3}s");
    let mut runs = Vec::new();
    let mut passed = true;
    for iteration in 0..if check_only { 1 } else { 7 } {
        eprintln!("run {} starting", iteration + 1);
        let start = Instant::now();
        let mut stft = precise_stft::PreciseStft::new();
        let spectra: Vec<_> = wave
            .chunks_exact(samples)
            .map(|channel| stft.forward(channel))
            .collect();
        let features = input_tensor.get_data_mut::<f32>()?;
        for batch_index in 0..batch {
            let batch_offset = batch_index * frames * 4100;
            for frame in 0..frames {
            for bin in 0..1025 {
                for (channel, spectrum) in spectra.iter().enumerate() {
                    let value = spectrum.values[bin * frames + frame];
                    let index = batch_offset + frame * 4100 + bin * 4 + channel * 2;
                    features[index] = value.re;
                    features[index + 1] = value.im;
                }
            }
            }
        }
        let analysis_seconds = start.elapsed().as_secs_f64();
        let network_start = Instant::now();
        request.infer()?;
        let mask_tensor = request.get_output_tensor()?;
        ensure!(
            mask_tensor.get_element_type()? == ElementType::F32,
            "expected FP32 mask"
        );
        let mask = mask_tensor.get_data::<f32>()?.to_vec();
        let network_seconds = network_start.elapsed().as_secs_f64();
        ensure!(mask.len() == batch * frames * 4100, "unexpected mask shape");
        let mut all_checks = Vec::with_capacity(batch);
        let mut run_passed = true;
        let mut failed_actual = None;
        for batch_index in 0..batch {
            let batch_offset = batch_index * frames * 4100;
            let mut batch_spectra = spectra.clone();
            for (channel, spectrum) in batch_spectra.iter_mut().enumerate() {
                for bin in 0..1025 {
                    for frame in 0..frames {
                        let index = batch_offset + frame * 4100 + bin * 4 + channel * 2;
                        spectrum.values[bin * frames + frame] *=
                            dsp::Complex32::new(mask[index], mask[index + 1]);
                    }
                }
            }
            let mut inverse = dsp::Stft::new(
                RoformerModel::FFT,
                RoformerModel::HOP,
                dsp::Padding::Reflect,
            )?;
            let mut actual = Vec::with_capacity(expected.values.len());
            for spectrum in &batch_spectra {
                actual.extend(inverse.inverse(spectrum, None)?);
            }
            ensure!(actual.len() == 2 * samples, "unexpected ISTFT length");
            if is_audio {
                for (i, sample) in wave.iter().enumerate() {
                    actual.push(sample - actual[i]);
                }
            }
            let (passed_batch, checks) = waveform_checks(&actual, &expected.values, 2 * samples);
            run_passed &= passed_batch;
            if !passed_batch && failed_actual.is_none() {
                failed_actual = Some(actual);
            }
            all_checks.push(json!({"batch": batch_index, "checks": checks, "passed": passed_batch}));
        }
        let seconds = start.elapsed().as_secs_f64();
        passed &= run_passed;
        runs.push(json!({"iteration": iteration, "execution": if check_only {"check"} else if iteration == 0 {"first"}
            else if iteration == 1 {"warmup"} else {"warm"}, "total_seconds": seconds,
            "rtf": seconds / (samples as f64 / 44100.0), "analysis_seconds": analysis_seconds,
            "network_seconds": network_seconds,
            "reconstruction_seconds": seconds - analysis_seconds - network_seconds, "checks": all_checks,
            "passed": run_passed}));
        serde_json::to_writer(&mut trace, runs.last().context("missing run")?)?;
        trace.write_all(b"\n")?;
        trace.flush()?;
        eprintln!("run {}: {seconds:.3}s, passed={run_passed}", iteration + 1);
        if !passed {
            let mut file = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(output.with_extension("failed.f32"))?,
            );
            for value in failed_actual.as_deref().unwrap_or_default() {
                file.write_all(&value.to_le_bytes())?;
            }
            file.flush()?;
            break;
        }
    }
    let report = json!({"backend": "OpenVINO C API via openvino-rs 0.11.0", "versions": versions,
        "device": args[0], "device_name": device_name, "available_devices": available,
        "runtime_environment": runtime_environment,
        "ir_provenance": provenance,
        "host_threads": threads, "requested_settings": requested_settings, "actual_settings": actual_settings,
        "supported_properties": supported, "passed": passed, "case": case.name, "validation_only": check_only,
        "shape": input.shape, "mode": if is_audio {"single_window_pcm"} else {"raw_window_pcm"},
        "model_load_seconds": load_seconds, "compile_seconds": compile_seconds, "runs": runs,
        "manifest_sha256": format!("{:x}", Sha256::digest(&manifest_raw)),
        "ir_sha256": digest(ir)?, "ir_weights_sha256": digest(&ir.with_extension("bin"))?,
        "binary_sha256": digest(&std::env::current_exe()?)?,
        "absolute_tolerance": 2e-4, "relative_tolerance": 2e-3,
        "rms_relative_tolerance": 1e-3, "rms_floor": 1e-4,
        "timing": "Rust FP64 STFT, feature packing into reusable input tensor, synchronous FP32 network, output copy, Rust FP32 ISTFT and residual; excludes IR loading, compilation, fixture I/O, encoding and verification",
        "scope": "Single-window IR execution; graph generation is a separate provenance record and product integration is not covered"});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    ensure!(
        passed,
        "complete waveform gate failed; report and failed PCM retained"
    );
    Ok(())
}
