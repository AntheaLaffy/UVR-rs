use std::{io::Write, ops::ControlFlow, path::PathBuf, time::Instant};

use anyhow::{Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use uvr_backend_probe::Probe;
use uvr_core::{roformer::RoformerModel, task::TaskCancelled};

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut dump = false;
    let mut measure = false;
    let mut selected_case = None;
    while let Some(flag) = args.first() {
        if flag == "--dump-output" {
            dump = true;
        } else if flag == "--measure" {
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
        "usage: probe-roformer [--dump-output] [--measure] [--case <name>] <original.ckpt> <fixtures> <report.json>"
    );
    ensure!(!PathBuf::from(&args[2]).exists(), "report already exists");
    let root = PathBuf::from(&args[1]);
    let raw = std::fs::read(root.join("manifest.json"))?;
    let probe = Probe::load_paths(root, PathBuf::from(&args[2]))?;
    let start = Instant::now();
    let model = RoformerModel::load(&PathBuf::from(&args[0]))?;
    let load_seconds = start.elapsed().as_secs_f64();
    eprintln!("1296: all 699 checkpoint tensors loaded in {load_seconds:.3}s");
    let mut checks = Vec::new();
    for (case_index, case) in probe.cases.iter().enumerate() {
        if selected_case
            .as_ref()
            .is_some_and(|name| name != &case.name)
        {
            continue;
        }
        let input = &probe.tensors[&case.input];
        let expected = &probe.tensors[&case.expected];
        ensure!(input.shape.len() == 2 && input.shape[0] == 2);
        let samples = input.shape[1];
        ensure!(expected.shape == [2, samples / 441 * 441]);
        for iteration in 0..if measure { 7 } else { 1 } {
            let mut completed = 0;
            let mut total = 0;
            let start = Instant::now();
            let mut last_step = start;
            let mut step_seconds = Vec::new();
            let actual =
                model.predict_window_with_progress(&input.values, samples, |done, count| {
                    assert!(done == 0 || done == completed + 1);
                    let now = Instant::now();
                    if done > 0 {
                        step_seconds.push((now - last_step).as_secs_f64());
                    }
                    last_step = now;
                    completed = done;
                    total = count;
                    if done % 100 == 0 || done == count {
                        eprintln!("1296/{}: {done}/{count}", case.name);
                    }
                    ControlFlow::Continue(())
                })?;
            let seconds = start.elapsed().as_secs_f64();
            ensure!(completed == total && total > 0 && actual.len() == expected.values.len());
            if dump && iteration == 0 {
                let path = PathBuf::from(&args[2]).with_extension(format!("{case_index}.f32"));
                let mut file = std::io::BufWriter::new(
                    std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path)?,
                );
                for value in &actual {
                    file.write_all(&value.to_le_bytes())?;
                }
                file.flush()?;
            }
            let mut channels = Vec::new();
            let mut total_squared_error = 0.0;
            let mut total_energy = 0.0;
            for (channel, (actual, reference)) in actual
                .chunks_exact(expected.shape[1])
                .zip(expected.values.chunks_exact(expected.shape[1]))
                .enumerate()
            {
                let mut maximum = 0.0f64;
                let mut squared_error = 0.0;
                let mut energy = 0.0;
                for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
                    let error = f64::from(a) - f64::from(b);
                    ensure!(
                        a.is_finite() && error.abs() <= 2e-4 + 2e-3 * f64::from(b).abs(),
                        "{}/{channel}[{i}]: actual={a}, reference={b}, error={error}",
                        case.name
                    );
                    maximum = maximum.max(error.abs());
                    squared_error += error * error;
                    energy += f64::from(b).powi(2);
                }
                let rmse = (squared_error / actual.len() as f64).sqrt();
                let reference_rms = (energy / actual.len() as f64).sqrt();
                total_squared_error += squared_error;
                total_energy += energy;
                if case.name == "silence" {
                    ensure!(actual.iter().all(|v| *v == 0.0));
                }
                channels.push(json!({"channel": channel, "max_absolute_error": maximum, "rmse": rmse, "reference_rms": reference_rms}));
            }
            // The agreed RMS gate is per stereo stem; channel statistics remain diagnostic.
            let rmse = (total_squared_error / actual.len() as f64).sqrt();
            let reference_rms = (total_energy / actual.len() as f64).sqrt();
            ensure!(
                rmse <= 1e-3 * reference_rms.max(1e-4),
                "{}: stem RMSE {rmse} exceeds audio gate (reference RMS {reference_rms})",
                case.name
            );
            eprintln!(
                "1296/{} run {}: {seconds:.3}s, complete waveform check passed",
                case.name,
                iteration + 1
            );
            checks.push(
            json!({"name": case.name, "shape": expected.shape, "channels": channels,
            "rmse": rmse, "reference_rms": reference_rms,
            "progress_steps": total, "single_execution_seconds": seconds,
            "iteration": iteration, "execution": if !measure { "verification" } else if iteration == 0 { "first" } else if iteration == 1 { "warmup" } else { "warm" },
            "step_seconds": step_seconds}),
        );
        }
    }
    ensure!(!checks.is_empty(), "no matching fixture cases");
    for cancel_at in [0, 3] {
        let error = model
            .predict_window_with_progress(&[0.2; 2050], 1025, |done, _| {
                if done == cancel_at {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .expect_err("cancellation must stop the network");
        ensure!(error.is::<TaskCancelled>());
    }
    ensure!(model.predict_window(&[], 0).is_err());
    ensure!(model.predict_window(&[0.0; 2048], 1024).is_err());
    ensure!(model.predict_window(&[f32::NAN; 2050], 1025).is_err());
    let report = json!({"mode": if measure { "performance" } else { "verification_only" }, "variant": "1296", "backend": "burn-flex 0.21.0 CPU FP32",
        "manifest_sha256": format!("{:x}", Sha256::digest(raw)), "threads": std::env::var("RAYON_NUM_THREADS")?,
        "model_load_seconds": load_seconds, "strict_tensor_count": 699, "absolute_tolerance": 2e-4,
        "relative_tolerance": 2e-3, "rms_relative_tolerance": 1e-3, "rms_floor": 1e-4, "checks": checks,
        "cancellation_and_invalid_input": "passed",
        "timing": "raw complete network including precise STFT, ISTFT and output materialization; excludes load, fixture I/O and validation"});
    std::fs::write(&args[2], serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}
