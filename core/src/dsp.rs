//! CPU, FP32, centered STFT with a periodic Hann window and explicit padding.
//! Channels are processed separately; values use the reference [frequency, time] layout.

use std::{fmt, sync::Arc};

pub use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Padding {
    Constant,
    Reflect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DspError(pub(crate) &'static str);

impl fmt::Display for DspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for DspError {}

#[derive(Debug, Clone)]
pub struct Spectrogram {
    pub bins: usize,
    pub frames: usize,
    /// Complex bins including Nyquist, indexed by `bin * frames + frame`.
    pub values: Vec<Complex32>,
}

/// Unnormalized, one-sided transforms for the even FFT sizes in the target models.
/// Reuse one instance to avoid rebuilding FFT plans, the window, and scratch buffers.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    padding: Padding,
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    buffer: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize, padding: Padding) -> Result<Self, DspError> {
        if n_fft < 2 || !n_fft.is_multiple_of(2) {
            return Err(DspError("FFT size must be even and at least 2"));
        }
        if hop == 0 || hop >= n_fft {
            return Err(DspError("hop must be between 1 and FFT size - 1"));
        }
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(n_fft);
        let inverse = planner.plan_fft_inverse(n_fft);
        let scratch_len = forward
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        let window = (0..n_fft)
            .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n_fft as f64).cos()) as f32)
            .collect();
        Ok(Self {
            n_fft,
            hop,
            padding,
            window,
            forward,
            inverse,
            buffer: vec![Complex32::default(); n_fft],
            scratch: vec![Complex32::default(); scratch_len],
        })
    }

    pub fn forward(&mut self, input: &[f32]) -> Result<Spectrogram, DspError> {
        if input.is_empty() || input.iter().any(|v| !v.is_finite()) {
            return Err(DspError("input must be nonempty and finite"));
        }
        let pad = self.n_fft / 2;
        // PyTorch reflection requires the input to be longer than each padded side.
        // Padding short audio belongs to the model scheduler, not this transform.
        if self.padding == Padding::Reflect && input.len() <= pad {
            return Err(DspError(
                "reflection padding requires input longer than half the FFT size",
            ));
        }
        input
            .len()
            .checked_add(self.n_fft)
            .ok_or(DspError("input length overflow"))?;
        let frames = input.len() / self.hop + 1;
        let bins = pad + 1;
        let size = bins
            .checked_mul(frames)
            .ok_or(DspError("spectrogram size overflow"))?;
        let mut values = vec![Complex32::default(); size];
        for frame in 0..frames {
            for i in 0..self.n_fft {
                let padded_index = frame * self.hop + i;
                let sample = if padded_index < pad {
                    match self.padding {
                        Padding::Constant => 0.0,
                        Padding::Reflect => input[pad - padded_index],
                    }
                } else {
                    let index = padded_index - pad;
                    if index < input.len() {
                        input[index]
                    } else {
                        match self.padding {
                            Padding::Constant => 0.0,
                            Padding::Reflect => input[input.len() - 2 - (index - input.len())],
                        }
                    }
                };
                self.buffer[i] = Complex32::new(sample * self.window[i], 0.0);
            }
            self.forward
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
            for bin in 0..bins {
                values[bin * frames + frame] = self.buffer[bin];
            }
        }
        Ok(Spectrogram {
            bins,
            frames,
            values,
        })
    }

    /// Restores an explicit sample count, or `(frames - 1) * hop` when omitted.
    /// As in the reference, an explicit length beyond frame coverage is zero-padded.
    pub fn inverse(
        &mut self,
        spectrum: &Spectrogram,
        length: Option<usize>,
    ) -> Result<Vec<f32>, DspError> {
        if spectrum.bins != self.n_fft / 2 + 1
            || spectrum.frames == 0
            || spectrum.bins.checked_mul(spectrum.frames) != Some(spectrum.values.len())
        {
            return Err(DspError("invalid spectrogram shape"));
        }
        if spectrum
            .values
            .iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(DspError("spectrum must be finite"));
        }
        let natural_length = (spectrum.frames - 1)
            .checked_mul(self.hop)
            .ok_or(DspError("output length overflow"))?;
        let covered = natural_length
            .checked_add(self.n_fft)
            .ok_or(DspError("output length overflow"))?;
        let mut wave = vec![0.0; covered];
        let mut energy = vec![0.0; covered];
        for frame in 0..spectrum.frames {
            for bin in 0..spectrum.bins {
                self.buffer[bin] = spectrum.values[bin * spectrum.frames + frame];
            }
            for bin in spectrum.bins..self.n_fft {
                self.buffer[bin] = self.buffer[self.n_fft - bin].conj();
            }
            self.inverse
                .process_with_scratch(&mut self.buffer, &mut self.scratch);
            for i in 0..self.n_fft {
                let index = frame * self.hop + i;
                wave[index] += self.buffer[i].re * (1.0 / self.n_fft as f32) * self.window[i];
                energy[index] += self.window[i] * self.window[i];
            }
        }
        let length = length.unwrap_or(natural_length);
        let pad = self.n_fft / 2;
        let mut output = vec![0.0; length];
        for (i, sample) in output.iter_mut().take(covered - pad).enumerate() {
            if energy[i + pad] <= 1e-11 {
                return Err(DspError("window overlap leaves an uncovered sample"));
            }
            *sample = wave[i + pad] / energy[i + pad];
        }
        Ok(output)
    }
}
