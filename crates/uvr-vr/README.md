# uvr-vr

[![crates.io](https://img.shields.io/crates/v/uvr-vr.svg?logo=rust&label=crates.io)](https://crates.io/crates/uvr-vr)

CPU inference and PCM separation for the 5-HP, 6-HP and DeEcho models supported by [UVR Rust](https://github.com/IronHpc/UVR-rs).

```toml
[dependencies]
uvr-vr = "0.1.0"
```

Original model weights are loaded at runtime and are not bundled. DeEcho batch and window concurrency are currently fixed at one while tuning remains incomplete.

Licensed under MIT.
