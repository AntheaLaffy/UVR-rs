# uvr-runtime

[![crates.io](https://img.shields.io/crates/v/uvr-runtime?logo=rust&label=crates.io)](https://crates.io/crates/uvr-runtime)
[![Documentation](https://docs.rs/uvr-runtime/badge.svg)](https://docs.rs/uvr-runtime)

The complete file-oriented API used by the [UVR Rust](https://github.com/IronHpc/UVR-rs) CLI and desktop app: audio codecs, model selection, runtime options, cancellation and protected output publication.

```toml
[dependencies]
uvr-runtime = "0.1.1"
```

Enable `openvino` for the optional Linux 1296 CPU backend or `model-download` for model cache management. Model weights and native OpenVINO libraries remain external.

Call [`separate_file`](https://docs.rs/uvr-runtime/latest/uvr_runtime/fn.separate_file.html) for a one-shot operation with recommended defaults. The returned [`FileOutput`](https://docs.rs/uvr-runtime/latest/uvr_runtime/struct.FileOutput.html) contains both output paths, track labels, sample metadata, and stage timings.

Licensed under MIT.
