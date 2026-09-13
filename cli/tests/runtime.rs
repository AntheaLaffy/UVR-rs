use std::process::Command;

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_uvr"));
    for variable in [
        "UVR_LANG",
        "RAYON_NUM_THREADS",
        "UVR_ROFORMER_TIME_BATCH",
        "UVR_ROFORMER_FREQUENCY_BATCH",
        "UVR_ROFORMER_WINDOW_PARALLELISM",
        "UVR_LINEAR_LAYOUT",
    ] {
        command.env_remove(variable);
    }
    command
}

#[test]
fn malformed_runtime_arguments_exit_two_with_specific_errors() {
    for (args, expected) in [
        (
            vec!["separate-vr", "5hp", "w", "i", "o", "--threads"],
            "--threads 缺少值",
        ),
        (
            vec![
                "separate-vr",
                "5hp",
                "w",
                "i",
                "o",
                "--inference-batch",
                "5",
            ],
            "--inference-batch",
        ),
        (
            vec![
                "separate-vr",
                "deecho",
                "w",
                "i",
                "o",
                "--parallel-windows",
                "0",
            ],
            "--parallel-windows",
        ),
        (
            vec!["separate-vr", "5hp", "w", "i", "o", "--backend", "burn"],
            "不适用于",
        ),
        (
            vec![
                "separate-vr",
                "5hp",
                "w",
                "i",
                "o",
                "--threads",
                "2",
                "--threads",
                "3",
            ],
            "选项重复",
        ),
        (
            vec!["separate-1296", "w", "i", "o", "--unknown", "1"],
            "未知选项",
        ),
        (
            vec![
                "separate-1296",
                "w",
                "i",
                "o",
                "--backend",
                "burn",
                "--frequency-batch",
                "0",
            ],
            "--frequency-batch",
        ),
        (
            vec![
                "separate-1296",
                "w",
                "i",
                "o",
                "--backend",
                "openvino-cpu",
                "--linear-layout",
                "flattened",
            ],
            "仅适用于 Burn",
        ),
    ] {
        let output = command().args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains(expected), "{args:?}: {error}");
        assert!(output.stdout.is_empty());
    }
    let output = command()
        .args(["separate-1296", "w", "i", "o", "--backend", "burn"])
        .env("UVR_ROFORMER_TIME_BATCH", "invalid")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("UVR_ROFORMER_TIME_BATCH"));
}

#[test]
fn explicit_settings_reach_task_and_report_effective_values() {
    let directory = std::env::temp_dir().join(format!("uvr-runtime-test-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    for (args, expected) in [
        (
            vec![
                "separate-vr",
                "5hp",
                "missing.pth",
                "missing.wav",
                "output",
                "--threads",
                "3",
                "--window-frames",
                "768",
                "--inference-batch",
                "2",
                "--parallel-windows",
                "4",
            ],
            "生效参数：窗口 768 帧，推理 batch 2，窗口并发 1",
        ),
        (
            vec![
                "separate-vr",
                "deecho",
                "missing.pth",
                "missing.wav",
                "output",
                "--threads",
                "3",
                "--inference-batch",
                "4",
                "--parallel-windows",
                "8",
            ],
            "生效参数：窗口 512 帧，推理 batch 1，窗口并发 1",
        ),
        (
            vec![
                "separate-1296",
                "missing.ckpt",
                "missing.wav",
                "output",
                "--backend",
                "burn",
                "--threads",
                "3",
                "--time-batch",
                "8",
                "--frequency-batch",
                "64",
                "--parallel-windows",
                "2",
                "--linear-layout",
                "batched",
            ],
            "生效参数：时间 batch 8，频率 batch 64，窗口并发 2，线性布局 batched",
        ),
    ] {
        let output = command()
            .args(&args)
            .current_dir(&directory)
            .env("RAYON_NUM_THREADS", "invalid")
            .env("UVR_ROFORMER_TIME_BATCH", "invalid")
            .env("UVR_ROFORMER_FREQUENCY_BATCH", "invalid")
            .env("UVR_ROFORMER_WINDOW_PARALLELISM", "invalid")
            .env("UVR_LINEAR_LAYOUT", "invalid")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("运行时：Burn CPU，3 线程"), "{error}");
        assert!(error.contains(expected), "{error}");
        assert!(error.contains("解码：missing.wav"), "{error}");
        assert!(output.stdout.is_empty());
        assert_eq!(
            std::fs::read_dir(directory.join("output")).unwrap().count(),
            0
        );
    }
}
