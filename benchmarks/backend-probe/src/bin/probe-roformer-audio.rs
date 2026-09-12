use std::{ops::ControlFlow, path::PathBuf, time::Instant};

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uvr_backend_probe::Probe;
use uvr_core::{
    roformer::{RoformerModel, RoformerStage},
    task::TaskCancelled,
};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 3,
        "usage: probe-roformer-audio <original.ckpt> <fixtures> <report.json>"
    );
    let root = PathBuf::from(&args[1]);
    let raw = std::fs::read(root.join("manifest.json"))?;
    let manifest: Value = serde_json::from_slice(&raw)?;
    ensure!(manifest["audio"]["variant"] == "1296");
    let probe = Probe::load_paths(root, PathBuf::from(&args[2]))?;
    let start = Instant::now();
    let model = RoformerModel::load(&PathBuf::from(&args[0]))?;
    let load_seconds = start.elapsed().as_secs_f64();
    let mut checks = Vec::new();
    for case in &probe.cases {
        let params = &manifest["audio"]["cases"][&case.name];
        let input = &probe.tensors[&case.input];
        let expected = &probe.tensors[&case.expected];
        ensure!(input.shape.len() == 2 && (1..=2).contains(&input.shape[0]));
        let channels: Vec<_> = input.values.chunks_exact(input.shape[1]).collect();
        let mut windows = 0;
        let mut complete = false;
        let output = model.separate(
            &channels,
            params["sample_rate"].as_u64().unwrap() as u32,
            |p| {
                windows = p.windows_completed;
                complete = p.stage == RoformerStage::Complete;
                if p.stage == RoformerStage::Inference && p.completed % 200 == 0 {
                    eprintln!(
                        "1296/{}: window {}/{} step {}/{}",
                        case.name, p.windows_completed, p.windows_total, p.completed, p.total
                    );
                }
                ControlFlow::Continue(())
            },
        )?;
        ensure!(complete && windows == params["windows"].as_u64().unwrap() as usize);
        ensure!(
            output.sample_rate == 44100
                && output.samples_per_channel
                    == params["output_samples"].as_u64().unwrap() as usize
        );
        ensure!(expected.shape == [2, 2, output.samples_per_channel]);
        let mut stems = Vec::new();
        for (stem, (actual, reference)) in [output.vocals, output.instrumental]
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
                "{}/{stem}: RMSE {rmse} exceeds audio gate",
                case.name
            );
            if case.name == "silence" {
                ensure!(actual.iter().all(|v| *v == 0.0));
            }
            stems.push(json!({"stem": stem, "max_absolute_error": maximum, "rmse": rmse, "reference_rms": reference_rms}));
        }
        eprintln!("1296/{}: complete audio check passed", case.name);
        checks.push(json!({"name": case.name, "shape": expected.shape, "stems": stems, "windows": windows,
            "single_execution_seconds": {"resampling": output.timings.resampling_seconds,
                "network": output.timings.network_seconds, "reconstruction": output.timings.reconstruction_seconds}}));
    }
    let result = model.separate(&[&[0.2; 2048]], 44100, |p| {
        if p.stage == RoformerStage::Inference && p.completed == 3 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    ensure!(result.err().unwrap().is::<TaskCancelled>());
    let report = json!({"mode": "verification_only", "variant": "1296", "backend": "burn-flex 0.21.0 CPU FP32",
        "manifest_sha256": format!("{:x}", Sha256::digest(raw)), "threads": std::env::var("RAYON_NUM_THREADS")?,
        "model_load_seconds": load_seconds, "absolute_tolerance": 2e-4, "relative_tolerance": 2e-3,
        "rms_relative_tolerance": 1e-3, "rms_floor": 1e-4, "checks": checks, "window_internal_cancellation": "passed"});
    std::fs::write(&args[2], serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}
