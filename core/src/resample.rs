//! Zero-phase polyphase resampling for the fixed audio baseline.
//! Matches SciPy's default Kaiser (beta=5) FIR design and zero boundaries.

use crate::dsp::DspError;

pub struct Polyphase {
    up: usize,
    down: usize,
    half: usize,
    filter: Vec<f32>,
}

impl Polyphase {
    /// Sample rates are integral and limited to 384 kHz to bound filter allocation.
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, DspError> {
        if source_rate == 0 || target_rate == 0 || source_rate.max(target_rate) > 384_000 {
            return Err(DspError("sample rates must be between 1 and 384000 Hz"));
        }
        let (mut a, mut b) = (source_rate, target_rate);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let (up, down) = ((target_rate / a) as usize, (source_rate / a) as usize);
        if up == down {
            return Ok(Self {
                up,
                down,
                half: 0,
                filter: vec![1.0],
            });
        }
        let maximum = up.max(down);
        let half = 10 * maximum;
        let cutoff = 1.0 / maximum as f64;
        let denominator = bessel_i0(5.0);
        let mut filter = Vec::with_capacity(2 * half + 1);
        for i in 0..=2 * half {
            let distance = i as f64 - half as f64;
            let phase = std::f64::consts::PI * cutoff * distance;
            let sinc = if i == half { 1.0 } else { phase.sin() / phase };
            let position = distance / half as f64;
            let window = bessel_i0(5.0 * (1.0 - position * position).max(0.0).sqrt()) / denominator;
            filter.push(cutoff * sinc * window);
        }
        let scale: f64 = filter.iter().sum();
        // The reference normalizes in f64, converts to input dtype, then multiplies by up.
        let filter = filter
            .into_iter()
            .map(|v| (v / scale) as f32 * up as f32)
            .collect();
        Ok(Self {
            up,
            down,
            half,
            filter,
        })
    }

    pub fn output_len(&self, input_len: usize) -> Result<usize, DspError> {
        input_len
            .checked_mul(self.up)
            .map(|n| n.div_ceil(self.down))
            .ok_or(DspError("resampled length overflow"))
    }

    /// Returns ceil(input samples * target rate / source rate) samples.
    pub fn process(&self, input: &[f32]) -> Result<Vec<f32>, DspError> {
        if input.is_empty() || input.iter().any(|v| !v.is_finite()) {
            return Err(DspError("resampler input must be nonempty and finite"));
        }
        if self.up == self.down {
            return Ok(input.to_vec());
        }
        let count = self.output_len(input.len())?;
        count
            .checked_mul(self.down)
            .and_then(|n| n.checked_add(self.half))
            .ok_or(DspError("resampler index overflow"))?;
        let mut output = vec![0.0; count];
        for (index, sample) in output.iter_mut().enumerate() {
            // Compensate the linear-phase delay without materializing upsampled zeros.
            let center = self.half + index * self.down;
            let first = center.saturating_sub(2 * self.half).div_ceil(self.up);
            let last = (center / self.up).min(input.len() - 1);
            for (i, &value) in input.iter().enumerate().take(last + 1).skip(first) {
                *sample += value * self.filter[center - i * self.up];
            }
            if !sample.is_finite() {
                return Err(DspError("resampling produced a nonfinite sample"));
            }
        }
        Ok(output)
    }
}

fn bessel_i0(value: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=64 {
        term *= (value * value * 0.25) / f64::from(k * k);
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
    }
    sum
}
