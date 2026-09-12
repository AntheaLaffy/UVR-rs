use std::{io::Write, path::PathBuf};

use anyhow::{Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex};
use uvr_backend_probe::Probe;
use uvr_core::dsp::{Padding, Stft};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2 || args.len() == 4,
        "usage: probe-roformer-stft <fixtures> <new-output-directory> [<window.f32> <f32fft|f32-product|f64-product>]"
    );
    let directory = PathBuf::from(&args[1]);
    std::fs::create_dir(&directory)?;
    let probe = Probe::load_paths(PathBuf::from(&args[0]), directory.join("unused.json"))?;
    let mut stft = Stft::new(2048, 441, Padding::Reflect)?;
    let custom_window = if args.len() == 4 {
        let bytes = std::fs::read(&args[2])?;
        ensure!(bytes.len() == 2048 * 4);
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(values.iter().all(|v| v.is_finite()));
        Some(values)
    } else {
        None
    };
    for case in &probe.cases {
        let input = &probe.tensors[&case.input];
        ensure!(input.shape.len() == 2 && input.shape[0] == 2);
        let mut file = std::io::BufWriter::new(std::fs::File::create(
            directory.join(format!("{}-stft.f32", case.name)),
        )?);
        for channel in input.values.chunks_exact(input.shape[1]) {
            let values = if let Some(window) = &custom_window {
                custom_stft(channel, window, args[3].to_str().unwrap())?
            } else {
                stft.forward(channel)?.values
            };
            for value in values {
                file.write_all(&value.re.to_le_bytes())?;
                file.write_all(&value.im.to_le_bytes())?;
            }
        }
        file.flush()?;
    }
    Ok(())
}

fn custom_stft(input: &[f32], window: &[f32], mode: &str) -> Result<Vec<Complex<f32>>> {
    ensure!(["f32fft", "f32-product", "f64-product"].contains(&mode));
    ensure!(input.len() > 1024);
    let frames = input.len() / 441 + 1;
    let mut result = vec![Complex::default(); 1025 * frames];
    let fft64 = FftPlanner::<f64>::new().plan_fft_forward(2048);
    let fft32 = FftPlanner::<f32>::new().plan_fft_forward(2048);
    let mut buffer64 = vec![Complex::default(); 2048];
    let mut buffer32 = vec![Complex::default(); 2048];
    for frame in 0..frames {
        for i in 0..2048 {
            let index = frame * 441 + i;
            let index = if index < 1024 {
                1024 - index
            } else {
                let index = index - 1024;
                if index < input.len() {
                    index
                } else {
                    2 * input.len() - 2 - index
                }
            };
            let product = input[index] * window[i];
            buffer32[i] = Complex::new(product, 0.0);
            buffer64[i] = Complex::new(
                if mode == "f64-product" {
                    f64::from(input[index]) * f64::from(window[i])
                } else {
                    f64::from(product)
                },
                0.0,
            );
        }
        if mode == "f32fft" {
            fft32.process(&mut buffer32);
        } else {
            fft64.process(&mut buffer64);
        }
        for bin in 0..1025 {
            result[bin * frames + frame] = if mode == "f32fft" {
                buffer32[bin]
            } else {
                Complex::new(buffer64[bin].re as f32, buffer64[bin].im as f32)
            };
        }
    }
    Ok(result)
}
