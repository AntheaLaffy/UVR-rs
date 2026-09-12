#![cfg(feature = "audio-io")]

use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use uvr_core::{audio_io, task::TaskCancelled};

#[test]
fn matches_libsndfile_decoding_including_mp3_gapless_and_unclipped_float() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let label = case["file"].as_str().unwrap();
        let path = root.join(label);
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
            case["sha256"].as_str().unwrap()
        );
        let reference = std::fs::read(root.join(case["reference"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&reference)),
            case["reference_sha256"].as_str().unwrap()
        );
        let expected: Vec<_> = reference
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let audio = audio_io::decode(&path, || true).unwrap();
        let count = case["samples_per_channel"].as_u64().unwrap() as usize;
        assert_eq!(
            audio.sample_rate as u64,
            case["sample_rate"].as_u64().unwrap()
        );
        assert_eq!(
            audio.channels.len() as u64,
            case["channels"].as_u64().unwrap()
        );
        assert!(
            audio.channels.iter().all(|c| c.len() == count),
            "{label}: sample count differs"
        );
        assert_eq!(expected.len(), audio.channels.len() * count);
        for (index, (&a, &b)) in audio.channels.iter().flatten().zip(&expected).enumerate() {
            let limit = case["absolute_tolerance"].as_f64().unwrap()
                + case["relative_tolerance"].as_f64().unwrap() * f64::from(b).abs();
            assert!(
                a.is_finite() && f64::from((a - b).abs()) <= limit,
                "{label}[{index}]: actual={a}, reference={b}"
            );
        }
    }
}

#[test]
fn incomplete_or_cancelled_outputs_are_not_published_and_existing_files_survive() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("complete.wav");
    let channels: &[&[f32]] = &[&[1.5, -2.0, 0.25], &[0.0, 0.1, -0.2]];
    let prepared = audio_io::prepare_wav(&path, channels, 44100, || true).unwrap();
    assert!(!path.exists());
    prepared.persist().unwrap();
    let decoded = audio_io::decode(&path, || true).unwrap();
    assert_eq!(decoded.channels[0], channels[0]);
    assert_eq!(decoded.channels[1], channels[1]);
    let original = std::fs::read(&path).unwrap();
    assert!(audio_io::prepare_wav(&path, channels, 44100, || true).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let cancelled = directory.path().join("cancelled.wav");
    let mut calls = 0;
    let error = audio_io::prepare_wav(&cancelled, channels, 44100, || {
        calls += 1;
        calls < 2
    })
    .err()
    .unwrap();
    assert!(error.is::<TaskCancelled>());
    assert!(!cancelled.exists());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    // A destination appearing after encoding must still not be overwritten.
    let raced = directory.path().join("appeared.wav");
    let prepared = audio_io::prepare_wav(&raced, channels, 44100, || true).unwrap();
    std::fs::write(&raced, b"existing output").unwrap();
    assert!(prepared.persist().is_err());
    assert_eq!(std::fs::read(&raced).unwrap(), b"existing output");
}

#[test]
fn rejects_unsupported_channels_empty_nonfinite_and_cancelled_input() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio");
    for name in [
        "three-channels.wav",
        "nonfinite.wav",
        "empty.wav",
        "missing.wav",
    ] {
        assert!(
            audio_io::decode(&root.join(name), || true).is_err(),
            "{name}"
        );
    }
    let error = audio_io::decode(&root.join("stereo-i16.wav"), || false)
        .err()
        .unwrap();
    assert!(error.is::<TaskCancelled>());
    let directory = tempfile::tempdir().unwrap();
    let truncated = directory.path().join("truncated.wav");
    audio_io::prepare_wav(&truncated, &[&vec![0.1; 8193]], 44100, || true)
        .unwrap()
        .persist()
        .unwrap();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&truncated)
        .unwrap();
    file.set_len(file.metadata().unwrap().len() - 100).unwrap();
    assert!(audio_io::decode(&truncated, || true).is_err());
}
