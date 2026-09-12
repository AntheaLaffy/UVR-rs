//! Original checkpoint + shared audio scheduler, including native cancellation/reuse.

use std::{io::Write, ops::ControlFlow, path::PathBuf, time::Instant};

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uvr_backend_probe::Probe;
use uvr_core::{
    roformer::{RoformerStage, openvino::OpenvinoRoformer},
    task::TaskCancelled,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: probe-openvino-core-audio <original.ckpt> <fixtures> <report.json>"
    );
    let checkpoint = PathBuf::from(&args[0]);
    let root = PathBuf::from(&args[1]);
    let report = PathBuf::from(&args[2]);
    ensure!(!report.exists(), "report exists");
    let mut trace = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report.with_extension("jsonl"))?;
    let raw = std::fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&raw)?;
    ensure!(
        manifest["audio"]["variant"] == "1296",
        "expected 1296 audio fixtures"
    );
    ensure!(
        manifest["sources"]
            .as_array()
            .context("missing sources")?
            .iter()
            .any(|s| s["stft_precision"]
                .as_str()
                .is_some_and(|p| p.contains("float64"))),
        "requires FP64 STFT reference"
    );
    let identity = uvr_core::weights::fingerprint(&checkpoint)?;
    ensure!(
        manifest["sources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["sha256"] == identity.sha256),
        "reference/checkpoint mismatch"
    );
    let probe = Probe::load_paths(root, report.clone())?;
    let threads: usize = std::env::var("RAYON_NUM_THREADS")?.parse()?;
    ensure!(threads > 0, "thread count must be positive");
    let mut cases = Vec::new();
    let mut cancellation = None;
    let mut passed = true;
    for case in &probe.cases {
        let params = &manifest["audio"]["cases"][&case.name];
        let input = &probe.tensors[&case.input];
        let reference = &probe.tensors[&case.expected];
        ensure!(
            input.shape.len() == 2 && matches!(input.shape[0], 1 | 2),
            "invalid input"
        );
        let sample_rate = u32::try_from(params["sample_rate"].as_u64().context("sample rate")?)?;
        let channels: Vec<_> = input.values.chunks_exact(input.shape[1]).collect();
        eprintln!("{}: loading original checkpoint", case.name);
        let start = Instant::now();
        let mut model = OpenvinoRoformer::load_for_audio(
            &checkpoint,
            input.shape[1],
            sample_rate,
            threads,
            || true,
        )?;
        let load_seconds = start.elapsed().as_secs_f64();
        serde_json::to_writer(
            &mut trace,
            &json!({"event": "loaded", "case": case.name,
            "load_seconds": load_seconds, "frames": model.info().frames,
            "actual_settings": model.info().actual_settings}),
        )?;
        writeln!(trace)?;
        trace.flush()?;
        if cancellation.is_none() && input.values.iter().any(|&v| v != 0.0) {
            let mut observations = 0;
            let mut requested = None;
            let result = model.separate(&channels, sample_rate, |p| {
                if p.stage == RoformerStage::Inference && p.total == 5 && p.completed == 2 {
                    observations += 1;
                    // Two callbacks precede submission; the third follows a timed native wait.
                    if observations >= 3 {
                        requested = Some(Instant::now());
                        return ControlFlow::Break(());
                    }
                }
                ControlFlow::Continue(())
            });
            let delay = requested
                .context("did not cancel an active native request")?
                .elapsed()
                .as_secs_f64();
            match result {
                Err(error) if error.is::<TaskCancelled>() => (),
                Err(error) => return Err(error.context("native cancellation failed")),
                Ok(_) => anyhow::bail!("native request completed instead of cancellation"),
            }
            cancellation = Some(
                json!({"case": case.name, "request_to_return_seconds": delay,
                "cancelled_after_native_submission": true, "reuse_checked_by_following_output": true}),
            );
            eprintln!(
                "{}: cancelled in {delay:.6}s; reusing same request",
                case.name
            );
        }
        let mut complete = false;
        let mut windows = 0;
        let output = model.separate(&channels, sample_rate, |p| {
            complete = p.stage == RoformerStage::Complete;
            windows = p.windows_completed;
            ControlFlow::Continue(())
        })?;
        ensure!(
            complete && windows == params["windows"].as_u64().context("windows")? as usize,
            "incomplete progress"
        );
        ensure!(
            output.sample_rate == 44100
                && output.samples_per_channel
                    == params["output_samples"].as_u64().context("length")? as usize,
            "invalid output format"
        );
        ensure!(
            reference.shape == [2, 2, output.samples_per_channel],
            "reference shape mismatch"
        );
        let actual = [output.vocals, output.instrumental];
        let mut checks = Vec::new();
        let mut case_passed = true;
        for (stem, (a, b)) in actual
            .iter()
            .zip(
                reference
                    .values
                    .chunks_exact(2 * output.samples_per_channel),
            )
            .enumerate()
        {
            ensure!(a.len() == b.len(), "stem length mismatch");
            let mut maximum = 0.0_f64;
            let mut error_energy = 0.0;
            let mut energy = 0.0;
            let mut failures = 0;
            for (&a, &b) in a.iter().zip(b) {
                let error = f64::from(a) - f64::from(b);
                maximum = maximum.max(error.abs());
                error_energy += error * error;
                energy += f64::from(b).powi(2);
                failures +=
                    usize::from(!a.is_finite() || error.abs() > 2e-4 + 2e-3 * f64::from(b).abs());
            }
            let rmse = (error_energy / a.len() as f64).sqrt();
            let rms = (energy / a.len() as f64).sqrt();
            let limit = 1e-3 * rms.max(1e-4);
            case_passed &= failures == 0 && rmse.is_finite() && rmse <= limit;
            if case.name == "silence" {
                case_passed &= a.iter().all(|&v| v == 0.0);
            }
            checks.push(
                json!({"stem": stem, "failed_samples": failures, "max_absolute_error": maximum,
                "rmse": rmse, "reference_rms": rms, "rms_limit": limit}),
            );
        }
        passed &= case_passed;
        let result = json!({"case": case.name, "passed": case_passed, "checks": checks,
            "frames": model.info().frames, "windows": windows, "shape": reference.shape,
            "load_seconds": load_seconds, "graph_seconds": model.info().graph_seconds,
            "compile_seconds": model.info().compile_seconds, "actual_settings": model.info().actual_settings,
            "resampling_seconds": output.timings.resampling_seconds,
            "network_seconds": output.timings.network_seconds, "reconstruction_seconds": output.timings.reconstruction_seconds});
        serde_json::to_writer(&mut trace, &result)?;
        writeln!(trace)?;
        trace.flush()?;
        cases.push(result);
        eprintln!("{}: complete PCM passed={case_passed}", case.name);
        if !case_passed {
            let mut file = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(report.with_extension("failed.f32"))?,
            );
            for value in actual.iter().flatten() {
                file.write_all(&value.to_le_bytes())?;
            }
            file.flush()?;
            break;
        }
    }
    let result = json!({"mode": "verification_only", "backend": "core OpenVINO CPU FP32 from original checkpoint",
        "passed": passed, "cases": cases, "cancellation": cancellation, "threads": threads,
        "checkpoint_sha256": identity.sha256, "manifest_sha256": format!("{:x}", Sha256::digest(raw)),
        "binary_sha256": uvr_core::weights::fingerprint(&std::env::current_exe()?)?.sha256,
        "absolute_tolerance": 2e-4, "relative_tolerance": 2e-3, "rms_relative_tolerance": 1e-3, "rms_floor": 1e-4,
        "python_path": std::env::var("PATH").ok(), "scope": "shared scheduler and request reuse; no file encoding or GUI"});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report)?;
    serde_json::to_writer_pretty(&mut file, &result)?;
    writeln!(file)?;
    ensure!(passed, "complete waveform gate failed; output retained");
    Ok(())
}
