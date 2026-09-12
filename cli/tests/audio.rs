use std::{path::Path, process::Command};

#[test]
fn audio_inspection_and_invalid_separation_arguments() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/fixtures/audio");
    let output = Command::new(env!("CARGO_BIN_EXE_uvr"))
        .arg("inspect-audio")
        .arg(root.join("stereo-i16.wav"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .starts_with("sample_rate: 44100\nchannels: 2\nsamples_per_channel: 1025\n")
    );
    for file in [
        "three-channels.wav",
        "nonfinite.wav",
        "empty.wav",
        "missing.wav",
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_uvr"))
            .arg("inspect-audio")
            .arg(root.join(file))
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
    }
    for args in [
        vec!["inspect-audio"],
        vec!["separate-1296", "weight", "input"],
        vec![
            "separate-1296",
            "weight",
            "input",
            "output",
            "--window-frames",
            "512",
        ],
        vec!["separate-vr", "bad", "weight", "input", "output"],
        vec![
            "separate-vr",
            "5hp",
            "weight",
            "input",
            "output",
            "--window-frames",
            "256",
        ],
        vec![
            "separate-vr",
            "5hp",
            "weight",
            "input",
            "output",
            "--window-frames",
            "513",
        ],
        vec![
            "separate-vr",
            "deecho",
            "weight",
            "input",
            "output",
            "--window-frames",
            "0",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_uvr"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(2));
    }
}
