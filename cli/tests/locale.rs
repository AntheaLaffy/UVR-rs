use std::{path::Path, process::Command};

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_uvr"));
    command.env_remove("UVR_LANG");
    command
}

#[test]
fn help_and_argument_errors_follow_the_selected_language() {
    for (language, help, unknown, number) in [
        ("zh-CN", "用法：", "未知选项", "正整数"),
        ("en", "Usage:", "Unknown option", "positive integer"),
        ("ja", "使い方：", "不明なオプション", "正の整数"),
    ] {
        let output = command()
            .args(["--lang", language, "--help"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains(help));
        for (option, value, expected) in [("--unknown", "1", unknown), ("--threads", "0", number)] {
            let output = command()
                .args([
                    "--lang",
                    language,
                    "separate-vr",
                    "5hp",
                    "weights",
                    "input",
                    "output",
                    option,
                    value,
                ])
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(2));
            assert!(output.stdout.is_empty());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains(expected), "{language}: {error}");
        }
    }
}

#[test]
fn explicit_language_overrides_environment_and_must_precede_command() {
    let output = command()
        .args(["--lang", "en", "--help"])
        .env("UVR_LANG", "invalid")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
    let output = command()
        .arg("--help")
        .env("UVR_LANG", "ja")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("使い方："));
    let output = command()
        .arg("--help")
        .env("UVR_LANG", "invalid")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("UVR_LANG"));
    for args in [
        vec!["--lang"],
        vec!["--lang", "invalid", "--help"],
        vec!["--lang", "en", "--lang", "ja", "--help"],
        vec![
            "separate-vr",
            "5hp",
            "weights",
            "input",
            "output",
            "--lang",
            "en",
        ],
    ] {
        let output = command().args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("--lang"));
    }
}

#[test]
fn language_preserves_machine_readable_audio_output() {
    let input =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/tests/fixtures/audio/stereo-i16.wav");
    let mut baseline = None;
    for language in ["zh-CN", "en", "ja"] {
        let output = command()
            .args(["--lang", language, "inspect-audio"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(output.status.success());
        if let Some(baseline) = &baseline {
            assert_eq!(&output.stdout, baseline);
        } else {
            assert!(
                String::from_utf8_lossy(&output.stdout)
                    .starts_with("sample_rate: 44100\nchannels: 2\n")
            );
            baseline = Some(output.stdout);
        }
    }
}
