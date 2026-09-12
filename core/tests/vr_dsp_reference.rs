use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use uvr_core::{
    resample::Polyphase,
    vr_dsp::{self, VrVariant},
};

fn compare(actual: &[f32], expected: &[f32], absolute: f32, relative: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= absolute + relative * b.abs(),
            "{label}[{index}]: actual={a}, reference={b}, max_abs={maximum}"
        );
    }
}

fn read(root: &Path, case: &Value) -> Vec<f32> {
    let raw = std::fs::read(root.join(case["file"].as_str().unwrap())).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        case["sha256"].as_str().unwrap()
    );
    assert_eq!(raw.len() % 4, 0);
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn matches_scipy_polyphase_and_fixed_uvr_multiband_processing() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vr-dsp");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let resamplings = manifest["resampling"].as_array().unwrap();
    assert_eq!(resamplings.len(), 14);
    for case in resamplings {
        let data = read(&root, case);
        let count = case["samples"].as_u64().unwrap() as usize;
        let resampler = Polyphase::new(
            case["source_rate"].as_u64().unwrap() as u32,
            case["target_rate"].as_u64().unwrap() as u32,
        )
        .unwrap();
        assert_eq!(
            resampler.output_len(count).unwrap(),
            case["output_samples"].as_u64().unwrap() as usize
        );
        compare(
            &resampler.process(&data[..count]).unwrap(),
            &data[count..],
            1e-6,
            2e-5,
            case["file"].as_str().unwrap(),
        );
    }
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    for case in cases {
        let data = read(&root, case);
        let number = |key: &str| case[key].as_u64().unwrap() as usize;
        let label = case["file"].as_str().unwrap();
        let variant = match case["variant"].as_str().unwrap() {
            "5hp" => VrVariant::HpFive,
            "6hp" => VrVariant::HpSix,
            "deecho" => VrVariant::DeEcho,
            value => panic!("unknown variant {value}"),
        };
        let input_len = number("samples") * number("channels");
        let channels: Vec<_> = data[..input_len].chunks_exact(number("samples")).collect();
        let spectrum = vr_dsp::analyze(variant, &channels, number("sample_rate") as u32).unwrap();
        assert_eq!(spectrum.variant(), variant);
        assert_eq!(
            (
                spectrum.bins(),
                spectrum.frames(),
                spectrum.output_samples()
            ),
            (number("bins"), number("frames"), number("output_samples"))
        );
        let elements = 2 * spectrum.bins() * spectrum.frames();
        assert_eq!(
            data.len(),
            input_len + 3 * elements + 4 * number("output_samples")
        );
        let complex: Vec<_> = spectrum
            .values()
            .iter()
            .flat_map(|v| [v.re, v.im])
            .collect();
        compare(
            &complex,
            &data[input_len..input_len + 2 * elements],
            8e-5,
            2e-5,
            label,
        );
        let mask = &data[input_len + 2 * elements..input_len + 3 * elements];
        let output = spectrum.reconstruct(mask).unwrap();
        let start = input_len + 3 * elements;
        let waveform_len = 2 * number("output_samples");
        compare(
            &output[0],
            &data[start..start + waveform_len],
            1e-5,
            2e-5,
            label,
        );
        compare(&output[1], &data[start + waveform_len..], 1e-5, 2e-5, label);
        if case["kind"] == "silence" {
            assert!(spectrum.magnitude().iter().all(|v| *v == 0.0));
            assert!(output.iter().flatten().all(|v| *v == 0.0));
        }
    }
}

#[test]
fn rejects_invalid_audio_rates_and_masks_without_panicking() {
    for (source, target) in [(0, 44100), (44100, 0), (384001, 44100)] {
        assert!(Polyphase::new(source, target).is_err());
    }
    let resampler = Polyphase::new(22050, 44100).unwrap();
    assert!(resampler.output_len(usize::MAX).is_err());
    for input in [&[][..], &[f32::NAN][..], &[f32::INFINITY][..]] {
        assert!(resampler.process(input).is_err());
    }
    let v = VrVariant::HpFive;
    for channels in [
        vec![],
        vec![&[][..]],
        vec![&[0.0][..]; 3],
        vec![&[0.0][..], &[0.0, 1.0][..]],
        vec![&[f32::NAN][..]],
    ] {
        assert!(vr_dsp::analyze(v, &channels, 44100).is_err());
    }
    let spectrum = vr_dsp::analyze(v, &[&[0.1, 0.2, 0.3]], 44100).unwrap();
    assert!(spectrum.reconstruct(&[]).is_err());
    for value in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
        assert!(
            spectrum
                .reconstruct(&vec![value; spectrum.values().len()])
                .is_err()
        );
    }
}
