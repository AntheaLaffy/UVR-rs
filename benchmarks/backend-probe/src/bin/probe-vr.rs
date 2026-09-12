use std::{path::PathBuf, time::Instant};

use anyhow::{Result, ensure};
use uvr_backend_probe::{Probe, measure, verify};
use uvr_core::vr::{DeEchoModel, HpKaraokeModel};

type Predict = Box<dyn Fn(&[f32], usize, usize) -> Result<Vec<f32>>>;

fn main() -> Result<()> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let mut check_only = false;
    let mut deecho = false;
    let mut selected_case = None;
    while let Some(flag) = args.first() {
        if flag == "--check-only" {
            check_only = true;
        } else if flag == "--deecho" {
            deecho = true;
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
        "usage: probe-vr [--check-only] [--deecho] [--case <name>] <original.pth> <fixture-directory> <report.json>"
    );
    let probe = Probe::load_paths(PathBuf::from(&args[1]), PathBuf::from(&args[2]))?;
    let start = Instant::now();
    let (name, bins, offset, predict): (String, usize, usize, Predict) = if deecho {
        let model = DeEchoModel::load(&PathBuf::from(&args[0]))?;
        (
            "DeEcho-DeReverb".into(),
            DeEchoModel::BINS + 1,
            DeEchoModel::OFFSET,
            Box::new(move |input, frames, batch| {
                ensure!(batch == 1, "DeEcho probe expects batch 1");
                model.predict_mask(input, frames)
            }),
        )
    } else {
        let model = HpKaraokeModel::load(&PathBuf::from(&args[0]))?;
        (
            format!("HP-{:?}", model.variant()),
            model.variant().bins() + 1,
            HpKaraokeModel::OFFSET,
            Box::new(move |input, frames, batch| model.predict_masks(input, frames, batch)),
        )
    };
    eprintln!(
        "{name}: checkpoint loaded in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let mut measurements = Vec::new();
    let mut checks = Vec::new();
    for case in &probe.cases {
        if selected_case
            .as_ref()
            .is_some_and(|name| name != &case.name)
        {
            continue;
        }
        let input = &probe.tensors[&case.input];
        ensure!(
            input.shape.len() == 4 && input.shape[1..3] == [2, bins],
            "incorrect fixture shape"
        );
        let frames = input.shape[3];
        let batch = input.shape[0];
        let execute = || {
            let output = predict(&input.values, frames, batch)?;
            Ok((vec![batch, 2, bins, frames - 2 * offset], output))
        };
        if check_only {
            checks.push(verify(&case.name, &probe.tensors[&case.expected], execute)?);
        } else {
            measurements.push(measure(
                &case.name,
                &probe.tensors[&case.expected],
                execute,
            )?);
        }
    }
    ensure!(
        !checks.is_empty() || !measurements.is_empty(),
        "unknown fixture case"
    );
    // Input contract failures must return errors before invoking tensor kernels.
    ensure!(predict(&[], 2 * offset, 1).is_err());
    ensure!(predict(&[], 319, 1).is_err());
    ensure!(predict(&[], 320, 1).is_err());
    ensure!(predict(&[], 320, 0).is_err());
    ensure!(predict(&[], 320, 5).is_err());
    let frames = 2 * offset + 16;
    let mut invalid = vec![0.0; 2 * bins * frames];
    invalid[0] = f32::NAN;
    ensure!(predict(&invalid, frames, 1).is_err());
    invalid[0] = -1.0;
    ensure!(predict(&invalid, frames, 1).is_err());
    let backend = format!("uvr-core {name}, burn-flex 0.21.0 CPU FP32; direct .pth import");
    if check_only {
        probe.write_verified(&backend, checks)
    } else {
        probe.write(&backend, measurements)
    }
}
