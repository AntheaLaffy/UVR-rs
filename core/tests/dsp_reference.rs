use sha2::{Digest, Sha256};
use uvr_core::dsp::{Complex32, Padding, Spectrogram, Stft};

fn compare(actual: &[f32], expected: &[f32], absolute: f32, relative: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= absolute + relative * b.abs(),
            "{label}[{i}]: actual={a}, reference={b}"
        );
    }
}

#[test]
fn matches_pytorch_spectra_and_modified_spectrum_reconstruction() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsp");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let cases = manifest["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let label = case["file"].as_str().unwrap();
        let raw = std::fs::read(root.join(label)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            case["sha256"].as_str().unwrap()
        );
        assert_eq!(raw.len() % 4, 0);
        let data: Vec<_> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let number = |key: &str| case[key].as_u64().unwrap() as usize;
        let (n_fft, hop, samples, frames) = (
            number("n_fft"),
            number("hop"),
            number("samples"),
            number("frames"),
        );
        let bins = n_fft / 2 + 1;
        let spectrum_len = 2 * bins * frames;
        assert_eq!(data.len(), 3 * samples + 2 * spectrum_len);
        let input = &data[..samples];
        let expected_spectrum = &data[samples..samples + spectrum_len];
        let modified_spectrum = &data[samples + spectrum_len..samples + 2 * spectrum_len];
        let expected_wave = &data[samples + 2 * spectrum_len..2 * samples + 2 * spectrum_len];
        let expected_filtered = &data[2 * samples + 2 * spectrum_len..];
        let padding = match case["padding"].as_str().unwrap() {
            "constant" => Padding::Constant,
            "reflect" => Padding::Reflect,
            other => panic!("unknown padding {other}"),
        };
        let mut stft = Stft::new(n_fft, hop, padding).unwrap();
        let spectrum = stft.forward(input).unwrap();
        assert_eq!((spectrum.bins, spectrum.frames), (bins, frames));
        let actual: Vec<_> = spectrum.values.iter().flat_map(|v| [v.re, v.im]).collect();
        compare(&actual, expected_spectrum, 8e-5, 2e-5, label);
        compare(
            &stft.inverse(&spectrum, Some(samples)).unwrap(),
            expected_wave,
            2e-6,
            2e-5,
            label,
        );
        compare(
            &stft.inverse(&spectrum, None).unwrap(),
            &expected_wave[..(frames - 1) * hop],
            2e-6,
            2e-5,
            label,
        );
        let modified = Spectrogram {
            bins,
            frames,
            values: modified_spectrum
                .as_chunks::<2>()
                .0
                .iter()
                .map(|v| Complex32::new(v[0], v[1]))
                .collect(),
        };
        compare(
            &stft.inverse(&modified, Some(samples)).unwrap(),
            expected_filtered,
            2e-6,
            2e-5,
            label,
        );
        // Reusing plans and work buffers must not retain a previous channel's data.
        let silent = stft.forward(&vec![0.0; samples]).unwrap();
        assert!(silent.values.iter().all(|v| *v == Complex32::default()));
    }
}

#[test]
fn rejects_invalid_parameters_and_data() {
    for (fft, hop) in [(0, 1), (1, 1), (7, 2), (8, 0), (8, 8)] {
        assert!(Stft::new(fft, hop, Padding::Constant).is_err());
    }
    let mut stft = Stft::new(8, 2, Padding::Reflect).unwrap();
    assert!(stft.forward(&[]).is_err());
    assert!(stft.forward(&[0.0; 4]).is_err());
    assert!(stft.forward(&[f32::NAN; 8]).is_err());
    assert!(stft.forward(&[f32::INFINITY; 8]).is_err());
    assert!(stft.forward(&[0.0; 5]).is_ok());
    let mut spectrum = stft.forward(&[1.0; 8]).unwrap();
    spectrum.values.pop();
    assert!(stft.inverse(&spectrum, None).is_err());
    let mut spectrum = stft.forward(&[1.0; 8]).unwrap();
    spectrum.values[0].im = f32::NAN;
    assert!(stft.inverse(&spectrum, None).is_err());
}
