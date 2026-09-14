# uvr-roformer

CPU inference and PCM separation for the BS-RoFormer 1296 checkpoint supported by [UVR Rust](https://github.com/IronHpc/UVR-rs). Burn CPU is always available; enable `openvino` for the optional OpenVINO CPU backend.

```toml
[dependencies]
uvr-roformer = { version = "0.1.0", features = ["openvino"] }
```

Original model weights and OpenVINO native libraries are not bundled. Full-track and multi-model-chain performance tuning remains incomplete.

Licensed under MIT.
