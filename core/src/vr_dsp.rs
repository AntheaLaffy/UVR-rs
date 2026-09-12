//! Fixed VR multi-band preprocessing and reconstruction, independent of a tensor backend.
//! Parameters and deliberate reference corrections are documented in docs/baseline.md.

use crate::{
    dsp::{Complex32, DspError, Padding, Spectrogram, Stft},
    resample::Polyphase,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrVariant {
    HpFive,
    HpSix,
    DeEcho,
}

impl VrVariant {
    pub fn bins(self) -> usize {
        if self == Self::HpSix { 640 } else { 672 }
    }
    pub fn offset(self) -> usize {
        if self == Self::DeEcho { 64 } else { 128 }
    }
    fn bands(self) -> &'static [Band] {
        if self == Self::HpSix {
            &THREE_BAND
        } else {
            &FOUR_BAND
        }
    }
}

#[derive(Clone, Copy)]
struct Band {
    rate: u32,
    hop: usize,
    fft: usize,
    crop: [usize; 2],
    low: Option<[usize; 2]>,
    high: Option<[usize; 2]>,
}

const FOUR_BAND: [Band; 4] = [
    Band {
        rate: 7350,
        hop: 80,
        fft: 640,
        crop: [0, 85],
        low: Some([25, 53]),
        high: None,
    },
    Band {
        rate: 7350,
        hop: 80,
        fft: 320,
        crop: [4, 87],
        low: Some([31, 62]),
        high: Some([25, 12]),
    },
    Band {
        rate: 14700,
        hop: 160,
        fft: 512,
        crop: [17, 216],
        low: Some([139, 210]),
        high: Some([48, 24]),
    },
    Band {
        rate: 44100,
        hop: 480,
        fft: 960,
        crop: [78, 383],
        low: None,
        high: Some([130, 86]),
    },
];
const THREE_BAND: [Band; 3] = [
    Band {
        rate: 11025,
        hop: 108,
        fft: 1024,
        crop: [0, 187],
        low: Some([92, 186]),
        high: None,
    },
    Band {
        rate: 22050,
        hop: 216,
        fft: 768,
        crop: [0, 212],
        low: Some([174, 209]),
        high: Some([68, 34]),
    },
    Band {
        rate: 44100,
        hop: 432,
        fft: 640,
        crop: [66, 307],
        low: None,
        high: Some([86, 72]),
    },
];

/// Stereo, channel-major combined spectrum. Fields are read-only outside this module.
pub struct VrSpectrum {
    variant: VrVariant,
    frames: usize,
    output_samples: usize,
    values: Vec<Complex32>,
}

impl VrSpectrum {
    pub fn variant(&self) -> VrVariant {
        self.variant
    }
    pub fn frames(&self) -> usize {
        self.frames
    }
    pub fn bins(&self) -> usize {
        self.variant.bins() + 1
    }
    pub fn output_samples(&self) -> usize {
        self.output_samples
    }
    pub fn values(&self) -> &[Complex32] {
        &self.values
    }
    pub fn magnitude(&self) -> Vec<f32> {
        self.values.iter().map(|v| v.norm()).collect()
    }

    /// Reconstruct primary and complementary spectra without independent normalization.
    pub fn reconstruct(&self, mask: &[f32]) -> Result<[Vec<f32>; 2], DspError> {
        if mask.len() != self.values.len()
            || mask
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(DspError(
                "mask must match the spectrum and contain finite values in [0,1]",
            ));
        }
        let primary: Vec<_> = self.values.iter().zip(mask).map(|(&v, &m)| v * m).collect();
        let residual: Vec<_> = self
            .values
            .iter()
            .zip(mask)
            .map(|(&v, &m)| v * (1.0 - m))
            .collect();
        Ok([self.synthesize(&primary)?, self.synthesize(&residual)?])
    }

    /// Returns one stereo waveform as [left samples, right samples].
    fn synthesize(&self, values: &[Complex32]) -> Result<Vec<f32>, DspError> {
        let bands = self.variant.bands();
        let mut output: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut offset = 0;
        for (d, band) in bands.iter().enumerate() {
            let width = band.crop[1] - band.crop[0];
            let mut transform = Stft::new(band.fft, band.hop, Padding::Constant)?;
            let mut wave: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            for (channel, result) in wave.iter_mut().enumerate() {
                // Uncovered bins must be zero: uninitialized upstream np.ndarray is not a reference.
                let mut spectrum = Spectrogram {
                    bins: band.fft / 2 + 1,
                    frames: self.frames,
                    values: vec![Complex32::default(); (band.fft / 2 + 1) * self.frames],
                };
                for i in 0..width {
                    let src = (channel * self.bins() + offset + i) * self.frames;
                    let dst = (band.crop[0] + i) * self.frames;
                    spectrum.values[dst..dst + self.frames]
                        .copy_from_slice(&values[src..src + self.frames]);
                }
                filter_band(&mut spectrum, *band, self.variant == VrVariant::DeEcho);
                *result = transform.inverse(&spectrum, None)?;
            }
            if self.variant == VrVariant::HpSix {
                for i in 0..wave[0].len() {
                    let (a, b) = (wave[0][i], wave[1][i]);
                    wave[0][i] = b / 1.25 + 0.4 * a;
                    wave[1][i] = a / 1.25 - 0.4 * b;
                }
            }
            for (destination, contribution) in output.iter_mut().zip(wave) {
                if d == 0 {
                    *destination = contribution;
                } else {
                    if destination.len() != contribution.len() {
                        return Err(DspError("inconsistent multi-band reconstruction length"));
                    }
                    for (sample, value) in destination.iter_mut().zip(contribution) {
                        *sample += value;
                    }
                }
                if let Some(next) = bands.get(d + 1) {
                    *destination = Polyphase::new(band.rate, next.rate)?.process(destination)?;
                }
            }
            offset += width;
        }
        let mut result = Vec::with_capacity(2 * self.output_samples);
        for channel in &output {
            if channel.len() < self.output_samples {
                return Err(DspError("audio was not padded to a complete frame"));
            }
            result.extend_from_slice(&channel[..self.output_samples]);
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err(DspError("reconstruction produced nonfinite audio"));
        }
        Ok(result)
    }
}

/// Accepts planar mono/stereo PCM, converts to 44.1 kHz and pads to complete hops.
pub fn analyze(
    variant: VrVariant,
    channels: &[&[f32]],
    sample_rate: u32,
) -> Result<VrSpectrum, DspError> {
    if channels.is_empty()
        || channels.len() > 2
        || channels[0].is_empty()
        || channels
            .iter()
            .any(|c| c.len() != channels[0].len() || c.iter().any(|v| !v.is_finite()))
    {
        return Err(DspError(
            "audio must contain one or two equal, nonempty, finite channels",
        ));
    }
    let resampler = Polyphase::new(sample_rate, 44100)?;
    let mut wave = [
        resampler.process(channels[0])?,
        resampler.process(channels[channels.len() - 1])?,
    ];
    let output_samples = wave[0].len();
    let bands = variant.bands();
    let hop = bands[bands.len() - 1].hop;
    let padded = output_samples
        .checked_add(hop - 1)
        .map(|n| n / hop * hop)
        .ok_or(DspError("audio padding length overflow"))?;
    for channel in &mut wave {
        channel.resize(padded, 0.0);
    }
    let frames = padded / hop + 1;
    let bins = variant.bins() + 1;
    let size = (2 * bins)
        .checked_mul(frames)
        .ok_or(DspError("multi-band spectrum size overflow"))?;
    let mut values = vec![Complex32::default(); size];
    let mut rate = 44100;
    let mut offset = variant.bins();
    for band in bands.iter().rev() {
        let width = band.crop[1] - band.crop[0];
        offset -= width;
        let resampler = Polyphase::new(rate, band.rate)?;
        for channel in &mut wave {
            *channel = resampler.process(channel)?;
        }
        rate = band.rate;
        // Old HP ignores per-band convert_channels; only HP-6 enables global mid_side_b2.
        let mut transformed = wave.clone();
        if variant == VrVariant::HpSix {
            for i in 0..wave[0].len() {
                transformed[0][i] = wave[1][i] + 0.5 * wave[0][i];
                transformed[1][i] = wave[0][i] - 0.5 * wave[1][i];
            }
        }
        let mut stft = Stft::new(band.fft, band.hop, Padding::Constant)?;
        for (channel, input) in transformed.iter().enumerate() {
            let spectrum = stft.forward(input)?;
            if spectrum.frames != frames {
                return Err(DspError("inconsistent multi-band analysis length"));
            }
            for i in 0..width {
                let src = (band.crop[0] + i) * frames;
                let dst = (channel * bins + offset + i) * frames;
                values[dst..dst + frames].copy_from_slice(&spectrum.values[src..src + frames]);
            }
        }
    }
    let (start, stop) = if variant == VrVariant::HpSix {
        (639, 640)
    } else {
        (668, 672)
    };
    let mut gain = 1.0f64;
    for bin in 0..bins {
        let factor = if variant == VrVariant::DeEcho {
            low_gain(bin, start, stop, true)
        } else if bin > start && bin < stop {
            gain = 10.0f64.powf(-((bin - start) as f64) * (3.5 - gain) / 20.0);
            gain as f32
        } else {
            1.0
        };
        for channel in 0..2 {
            for value in
                &mut values[(channel * bins + bin) * frames..(channel * bins + bin + 1) * frames]
            {
                *value *= factor;
            }
        }
    }
    if values
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err(DspError("analysis produced a nonfinite spectrum"));
    }
    Ok(VrSpectrum {
        variant,
        frames,
        output_samples,
        values,
    })
}

fn filter_band(spectrum: &mut Spectrogram, band: Band, modern: bool) {
    for bin in 0..spectrum.bins {
        let high = band
            .high
            .map_or(1.0, |[start, stop]| high_gain(bin, start, stop - 1, modern));
        let low = band
            .low
            .map_or(1.0, |[start, stop]| low_gain(bin, start, stop, modern));
        for value in &mut spectrum.values[bin * spectrum.frames..(bin + 1) * spectrum.frames] {
            *value *= high;
            *value *= low;
        }
    }
}

fn low_gain(bin: usize, start: usize, stop: usize, modern: bool) -> f32 {
    if modern {
        if bin < start {
            1.0
        } else if bin >= stop - 1 {
            0.0
        } else {
            (stop - 1 - bin) as f32 / (stop - start) as f32
        }
    } else if bin < start {
        1.0
    } else if bin >= stop {
        0.0
    } else {
        (stop - 1 - bin) as f32 / (stop - start) as f32
    }
}

fn high_gain(bin: usize, start: usize, stop: usize, modern: bool) -> f32 {
    if modern {
        if bin <= stop + 1 {
            0.0
        } else if bin > start {
            1.0
        } else {
            (bin - stop - 1) as f32 / (start - stop) as f32
        }
    } else if bin <= stop {
        0.0
    } else if bin > start {
        1.0
    } else {
        (bin - stop - 1) as f32 / (start - stop) as f32
    }
}
