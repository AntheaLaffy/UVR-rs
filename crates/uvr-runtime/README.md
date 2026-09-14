# uvr-runtime

[![crates.io](https://img.shields.io/crates/v/uvr-runtime.svg?logo=rust&label=crates.io)](https://crates.io/crates/uvr-runtime)

The complete file-oriented API used by the [UVR Rust](https://github.com/IronHpc/UVR-rs) CLI and desktop app: audio codecs, model selection, runtime options, cancellation and protected output publication.

```toml
[dependencies]
uvr-runtime = "0.1.0"
```

Enable `openvino` for the optional Linux 1296 CPU backend or `model-download` for model cache management. Model weights and native OpenVINO libraries remain external.

Licensed under MIT.
