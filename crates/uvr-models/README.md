# uvr-models

[![crates.io](https://img.shields.io/crates/v/uvr-models?logo=rust&label=crates.io)](https://crates.io/crates/uvr-models)
[![Documentation](https://docs.rs/uvr-models/badge.svg)](https://docs.rs/uvr-models)

Supported-model metadata and weight fingerprinting from [UVR Rust](https://github.com/IronHpc/UVR-rs). Enable `download` to add verified model download and cache management.

```toml
[dependencies]
uvr-models = { version = "0.1.2", features = ["download"] }
```

Model weights remain external and subject to their authors' terms. This crate does not redistribute them.

Use [`ModelInfo::from_key`](https://docs.rs/uvr-models/latest/uvr_models/struct.ModelInfo.html) to resolve a supported model and [`fingerprint`](https://docs.rs/uvr-models/latest/uvr_models/fn.fingerprint.html) to verify a downloaded file. The optional `download` feature adds cache management and checksum verification.

Licensed under MIT.
