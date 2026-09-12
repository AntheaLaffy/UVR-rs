use std::process::Command;

#[test]
fn inspection_and_cli_exit_codes() {
    let directory =
        std::env::temp_dir().join(format!("uvr-inspection-test-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(directory.clone());
    let path = directory.join("weight with spaces.pth");
    std::fs::write(&path, b"abc").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_uvr"))
        .arg("inspect-weights")
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        concat!(
            "size_bytes: 3\n",
            "sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n",
            "uvr_md5: 900150983cd24fb0d6963f7d28e17f72\n",
        )
    );
    for path in [&directory, &directory.join("missing.pth")] {
        let output = Command::new(env!("CARGO_BIN_EXE_uvr"))
            .arg("inspect-weights")
            .arg(path)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
    for args in [
        vec!["inspect-weights"],
        vec!["inspect-weights", "a", "b"],
        vec!["unknown"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_uvr"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
    }
    for args in [vec![], vec!["--help"], vec!["--version"]] {
        assert!(
            Command::new(env!("CARGO_BIN_EXE_uvr"))
                .args(args)
                .output()
                .unwrap()
                .status
                .success()
        );
    }
}
