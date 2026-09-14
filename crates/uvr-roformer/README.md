# uvr-roformer

[![crates.io](https://img.shields.io/crates/v/uvr-roformer?logo=rust&label=crates.io)](https://crates.io/crates/uvr-roformer)
[![Documentation](https://docs.rs/uvr-roformer/badge.svg)](https://docs.rs/uvr-roformer)

CPU inference and PCM separation for the BS-RoFormer 1296 checkpoint supported by [UVR Rust](https://github.com/IronHpc/UVR-rs). Burn CPU is always available; enable `openvino` for the optional OpenVINO CPU backend.

```toml
[dependencies]
uvr-roformer = { version = "0.1.1", features = ["openvino"] }
```

Original model weights and OpenVINO native libraries are not bundled. Full-track and multi-model-chain performance tuning remains incomplete.

The Burn CPU backend is the portable default. OpenVINO is an opt-in feature because it requires a separately installed native runtime; consult the repository's [runtime guide](https://github.com/IronHpc/UVR-rs/blob/main/docs/runtime.md) before enabling it.

Licensed under MIT.
