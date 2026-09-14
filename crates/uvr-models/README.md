# uvr-models

[![crates.io](https://img.shields.io/crates/v/uvr-models.svg?logo=rust&label=crates.io)](https://crates.io/crates/uvr-models)

Supported-model metadata and weight fingerprinting from [UVR Rust](https://github.com/IronHpc/UVR-rs). Enable `download` to add verified model download and cache management.

```toml
[dependencies]
uvr-models = { version = "0.1.0", features = ["download"] }
```

Model weights remain external and subject to their authors' terms. This crate does not redistribute them.

Licensed under MIT.
