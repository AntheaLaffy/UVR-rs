//! Rust audio file I/O. Decoding preserves the source rate and channels;
//! model preprocessing owns resampling and mono-to-stereo conversion.

use std::{
    fs::File,
    io::{BufWriter, ErrorKind},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_MP3, DecoderOptions},
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};
use tempfile::NamedTempFile;

use crate::task::TaskCancelled;

pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

/// Decode WAV, FLAC or MP3. MP3 gapless metadata is honored by the demuxer/decoder.
pub fn decode(path: &Path, mut keep_going: impl FnMut() -> bool) -> Result<Audio> {
    if !keep_going() {
        return Err(TaskCancelled.into());
    }
    let source =
        File::open(path).with_context(|| format!("cannot open audio {}", path.display()))?;
    let stream = MediaSourceStream::new(Box::new(source), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .context("cannot identify a supported audio format (WAV, FLAC, MP3)")?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .context("audio file has no default track")?;
    let track_id = track.id;
    // MP3 without a LAME tag may have only a bitrate-based duration estimate.
    let expected_frames = if track.codec_params.codec != CODEC_TYPE_MP3
        || track.codec_params.delay.is_some()
        || track.codec_params.padding.is_some()
    {
        track.codec_params.n_frames
    } else {
        None
    };
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions { verify: true })
        .context("unsupported audio codec")?;
    let mut rate = None;
    let mut channel_layout = None;
    let mut channels: Vec<Vec<f32>> = Vec::new();
    loop {
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("cannot read audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .context("cannot decode audio packet")?;
        let spec = *decoded.spec();
        let count = spec.channels.count();
        ensure!(
            (1..=2).contains(&count),
            "only mono and stereo audio are supported"
        );
        ensure!((1..=384000).contains(&spec.rate), "unsupported sample rate");
        if let Some(previous) = rate {
            ensure!(
                previous == spec.rate && channel_layout == Some(spec.channels),
                "audio format changes within the stream"
            );
        } else {
            rate = Some(spec.rate);
            channel_layout = Some(spec.channels);
            channels = vec![Vec::new(); count];
        }
        let frames = decoded.frames();
        let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
        buffer.copy_interleaved_ref(decoded);
        for channel in &mut channels {
            channel
                .try_reserve(frames)
                .context("audio exceeds available memory")?;
        }
        for frame in buffer.samples().chunks_exact(count) {
            for (channel, &sample) in channels.iter_mut().zip(frame) {
                ensure!(
                    sample.is_finite(),
                    "decoded audio contains nonfinite samples"
                );
                channel.push(sample);
            }
        }
    }
    ensure!(
        decoder.finalize().verify_ok != Some(false),
        "audio checksum verification failed"
    );
    ensure!(
        !channels.is_empty() && !channels[0].is_empty(),
        "audio contains no samples"
    );
    if let Some(expected) = expected_frames {
        ensure!(
            channels[0].len() as u64 == expected,
            "decoded audio length differs from its declared frame count (possibly truncated)"
        );
    }
    Ok(Audio {
        sample_rate: rate.context("audio has no sample rate")?,
        channels,
    })
}

/// Fully encoded temporary WAV. Dropping before publication removes it.
pub struct PreparedWav {
    file: NamedTempFile,
    destination: PathBuf,
}

impl PreparedWav {
    /// Publishes a complete file, failing if a file or symlink already exists.
    pub fn persist(self) -> Result<PathBuf> {
        self.file
            .persist_noclobber(&self.destination)
            .with_context(|| {
                format!(
                    "cannot publish {} without overwriting",
                    self.destination.display()
                )
            })?;
        Ok(self.destination)
    }
}

pub fn ensure_output_absent(path: &Path) -> Result<()> {
    match path.symlink_metadata() {
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Ok(_) => anyhow::bail!("output already exists: {}", path.display()),
        Err(error) => {
            Err(error).with_context(|| format!("cannot inspect output {}", path.display()))
        }
    }
}

/// Encode planar PCM to float32 WAV without gain normalization or clipping.
pub fn prepare_wav(
    path: &Path,
    channels: &[&[f32]],
    sample_rate: u32,
    mut keep_going: impl FnMut() -> bool,
) -> Result<PreparedWav> {
    ensure!(
        (1..=2).contains(&channels.len()) && !channels[0].is_empty(),
        "WAV requires nonempty mono/stereo audio"
    );
    ensure!(
        (1..=384000).contains(&sample_rate),
        "unsupported sample rate"
    );
    let samples = channels[0].len();
    ensure!(
        channels
            .iter()
            .all(|c| c.len() == samples && c.iter().all(|s| s.is_finite())),
        "WAV channels must have equal lengths and finite samples"
    );
    let bytes = samples
        .checked_mul(channels.len())
        .and_then(|n| n.checked_mul(4))
        .context("WAV length overflow")?;
    ensure!(
        bytes <= u32::MAX as usize - 128,
        "audio exceeds the RIFF WAV size limit"
    );
    ensure_output_absent(path)?;
    if !keep_going() {
        return Err(TaskCancelled.into());
    }
    let directory = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file =
        NamedTempFile::new_in(directory).context("cannot create temporary output WAV")?;
    let mut writer = hound::WavWriter::new(
        BufWriter::new(file.as_file_mut()),
        hound::WavSpec {
            channels: channels.len() as u16,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for index in 0..samples {
        if index % 4096 == 0 && !keep_going() {
            return Err(TaskCancelled.into());
        }
        for channel in channels {
            writer.write_sample(channel[index])?;
        }
    }
    writer.finalize().context("cannot finalize WAV")?;
    if !keep_going() {
        return Err(TaskCancelled.into());
    }
    Ok(PreparedWav {
        file,
        destination: path.to_path_buf(),
    })
}
