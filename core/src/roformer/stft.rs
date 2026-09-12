use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex64};

use super::RoformerModel;
use crate::dsp::{Complex32, Spectrogram};

/// Quiet frequency bands are normalized before projection. Doing the forward
/// transform in f64 prevents FFT rounding from dominating those band inputs.
pub(super) struct PreciseStft {
    fft: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    buffer: Vec<Complex64>,
    scratch: Vec<Complex64>,
}

impl PreciseStft {
    pub(super) fn new() -> Self {
        let size = RoformerModel::FFT;
        let fft = FftPlanner::<f64>::new().plan_fft_forward(size);
        let scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
        Self {
            fft,
            scratch,
            buffer: vec![Complex64::default(); size],
            window: (0..size)
                .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / size as f64).cos())
                .collect(),
        }
    }

    /// The model validates finite input and the reflection-padding minimum.
    pub(super) fn forward(&mut self, audio: &[f32]) -> Spectrogram {
        let size = RoformerModel::FFT;
        let pad = size / 2;
        let frames = audio.len() / RoformerModel::HOP + 1;
        let bins = pad + 1;
        let mut values = vec![Complex32::default(); bins * frames];
        for frame in 0..frames {
            for i in 0..size {
                let padded = frame * RoformerModel::HOP + i;
                let index = if padded < pad {
                    pad - padded
                } else {
                    let index = padded - pad;
                    if index < audio.len() {
                        index
                    } else {
                        2 * audio.len() - 2 - index
                    }
                };
                self.buffer[i] = Complex64::new(f64::from(audio[index]) * self.window[i], 0.0);
            }
            self.fft
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
            for bin in 0..bins {
                values[bin * frames + frame] =
                    Complex32::new(self.buffer[bin].re as f32, self.buffer[bin].im as f32);
            }
        }
        Spectrogram {
            bins,
            frames,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_float64_torch_stft_cast_to_complex32() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/roformer-dsp");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        let tensors = &manifest["tensors"];
        let read = |name: &str| {
            let entry = &tensors[name];
            let raw = std::fs::read(root.join(entry["file"].as_str().unwrap())).unwrap();
            use sha2::{Digest, Sha256};
            assert_eq!(
                format!("{:x}", Sha256::digest(&raw)),
                entry["sha256"].as_str().unwrap()
            );
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let mut stft = PreciseStft::new();
        for case in manifest["cases"].as_array().unwrap() {
            let name = case["input"].as_str().unwrap();
            let input = read(name);
            let expected = read(case["expected"].as_str().unwrap());
            let samples = tensors[name]["shape"][1].as_u64().unwrap() as usize;
            let actual: Vec<_> = input
                .chunks_exact(samples)
                .flat_map(|channel| {
                    stft.forward(channel)
                        .values
                        .into_iter()
                        .flat_map(|v| [v.re, v.im])
                })
                .collect();
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.into_iter().zip(expected) {
                assert!(
                    (a - b).abs() <= 1e-10 + 2e-7 * b.abs(),
                    "actual {a}, reference {b}"
                );
            }
        }
    }
}
