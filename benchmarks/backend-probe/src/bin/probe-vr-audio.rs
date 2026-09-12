use std::{ops::ControlFlow, path::PathBuf, time::Instant};

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uvr_backend_probe::Probe;
use uvr_core::{
    vr::{TaskCancelled, VrOptions, VrSeparator, VrStage},
    vr_dsp::VrVariant,
};

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut measure = false;
    let mut selected_case = None;
    while let Some(flag) = args.first() {
        if flag == "--measure" {
            measure = true;
        } else if flag == "--case" {
            args.remove(0);
            ensure!(!args.is_empty(), "--case requires a fixture name");
            selected_case = Some(args[0].to_string_lossy().into_owned());
        } else {
            break;
        }
        args.remove(0);
    }
    ensure!(
        args.len() == 3,
        "usage: probe-vr-audio [--measure] [--case <name>] <original.pth> <fixtures> <report.json>"
    );
    let root = PathBuf::from(&args[1]);
    let raw = std::fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&raw)?;
    let variant = match manifest["audio"]["variant"].as_str() {
        Some("5hp") => VrVariant::HpFive,
        Some("6hp") => VrVariant::HpSix,
        Some("deecho") => VrVariant::DeEcho,
        _ => anyhow::bail!("missing audio variant"),
    };
    let probe = Probe::load_paths(root, PathBuf::from(&args[2]))?;
    let start = Instant::now();
    let model = VrSeparator::load(variant, &PathBuf::from(&args[0]))?;
    let load_seconds = start.elapsed().as_secs_f64();
    let mut checks = Vec::new();
    for case in &probe.cases {
        if selected_case
            .as_ref()
            .is_some_and(|name| name != &case.name)
        {
            continue;
        }
        let params = &manifest["audio"]["cases"][&case.name];
        let input = &probe.tensors[&case.input];
        ensure!(input.shape.len() == 2 && (1..=2).contains(&input.shape[0]) && input.shape[1] > 0);
        let channels: Vec<_> = input.values.chunks_exact(input.shape[1]).collect();
        let mut runs = Vec::new();
        for _ in 0..if measure { 7 } else { 1 } {
            let mut patches = 0;
            let mut completed = false;
            let mut window_seconds = Vec::new();
            let mut window_start = None;
            let mut first_progress_seconds = None;
            let start = Instant::now();
            let output = model.separate(
                &channels,
                params["sample_rate"].as_u64().unwrap() as u32,
                VrOptions {
                    window_frames: params["window_frames"].as_u64().unwrap() as usize,
                    // Keep historical audio probe measurements at batch one;
                    // the product default is benchmarked separately.
                    inference_batch: 1,
                },
                |progress| {
                    first_progress_seconds.get_or_insert_with(|| start.elapsed().as_secs_f64());
                    if progress.stage == VrStage::Inference {
                        if progress.completed > patches
                            && let Some(previous) = window_start
                        {
                            window_seconds
                                .push(Instant::now().duration_since(previous).as_secs_f64());
                        }
                        window_start = Some(Instant::now());
                        patches = progress.completed;
                    }
                    if progress.stage == VrStage::Complete {
                        completed = true;
                    }
                    ControlFlow::Continue(())
                },
            )?;
            let total_seconds = start.elapsed().as_secs_f64();
            ensure!(
                completed && patches == params["patches"].as_u64().unwrap() as usize,
                "progress mismatch"
            );
            ensure!(
                output.sample_rate == 44100
                    && output.samples_per_channel
                        == params["output_samples"].as_u64().unwrap() as usize
            );
            let expected = &probe.tensors[&case.expected];
            ensure!(
                expected.shape == [2, 2, output.samples_per_channel],
                "output shape mismatch"
            );
            let mut stems = Vec::new();
            for (stem, (actual, reference)) in [output.primary, output.residual]
                .iter()
                .zip(expected.values.chunks_exact(2 * output.samples_per_channel))
                .enumerate()
            {
                ensure!(actual.len() == reference.len());
                let mut maximum = 0.0f64;
                let mut squared_error = 0.0;
                let mut energy = 0.0;
                for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
                    let error = f64::from(a) - f64::from(b);
                    ensure!(
                        a.is_finite() && error.abs() <= 2e-4 + 2e-3 * f64::from(b).abs(),
                        "{}/{stem}[{i}]: actual={a}, reference={b}, error={error}",
                        case.name
                    );
                    maximum = maximum.max(error.abs());
                    squared_error += error * error;
                    energy += f64::from(b).powi(2);
                }
                let rmse = (squared_error / actual.len() as f64).sqrt();
                let reference_rms = (energy / actual.len() as f64).sqrt();
                ensure!(
                    rmse <= 1e-3 * reference_rms.max(1e-4),
                    "{}/{stem}: RMSE {rmse} exceeds audio gate (reference RMS {reference_rms})",
                    case.name
                );
                if case.name == "silence" {
                    ensure!(actual.iter().all(|v| *v == 0.0));
                }
                stems.push(json!({"stem": stem, "max_absolute_error": maximum, "rmse": rmse, "reference_rms": reference_rms}));
            }
            eprintln!(
                "{:?}/{}: audio check passed, {} windows",
                variant, case.name, patches
            );
            runs.push(json!({"name": case.name, "shape": expected.shape, "stems": stems, "patches": patches,
            "total_seconds": total_seconds, "rtf": total_seconds / (output.samples_per_channel as f64 / 44100.0),
            "first_progress_seconds": first_progress_seconds, "window_seconds": window_seconds,
            "single_execution_seconds": {"analysis": output.timings.analysis_seconds, "network": output.timings.network_seconds,
                                         "reconstruction": output.timings.reconstruction_seconds}}));
        }
        if measure {
            checks.push(json!({"name": case.name, "parameters": params,
                "first_execution": runs[0], "warmup": runs[1], "warm": runs[2..]}));
        } else {
            checks.push(runs.remove(0));
        }
    }
    ensure!(!checks.is_empty(), "unknown fixture case");
    let error = model
        .separate(&[&[0.25; 480]], 44100, VrOptions::default(), |p| {
            if p.stage == VrStage::Inference {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .err()
        .unwrap();
    ensure!(error.is::<TaskCancelled>());
    let report = json!({"mode": if measure { "measurement" } else { "verification_only" },
        "timing": "complete PCM separation; includes resampling, spectral analysis, windows and reconstruction; excludes fixture I/O, model load, encoding and verification",
        "warmup_executions": if measure { 1 } else { 0 },
        "variant": format!("{variant:?}"), "backend": "burn-flex 0.21.0 CPU FP32",
        "manifest_sha256": format!("{:x}", Sha256::digest(raw)), "threads": std::env::var("RAYON_NUM_THREADS")?,
        "model_load_seconds": load_seconds, "absolute_tolerance": 2e-4, "relative_tolerance": 2e-3,
        "rms_relative_tolerance": 1e-3, "rms_floor": 1e-4, "checks": checks});
    std::fs::write(&args[2], serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}
