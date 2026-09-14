# uvr-vr

[![crates.io](https://img.shields.io/crates/v/uvr-vr?logo=rust&label=crates.io)](https://crates.io/crates/uvr-vr)
[![Documentation](https://docs.rs/uvr-vr/badge.svg)](https://docs.rs/uvr-vr)

CPU inference and PCM separation for the 5-HP, 6-HP and DeEcho models supported by [UVR Rust](https://github.com/IronHpc/UVR-rs).

```toml
[dependencies]
uvr-vr = "0.1.1"
```

Original model weights are loaded at runtime and are not bundled. DeEcho batch and window concurrency are currently fixed at one while tuning remains incomplete.

The inference methods operate on interleaved 44.1 kHz PCM and report progress through callbacks. Use `uvr-runtime` when you need decoding, resampling, WAV output, or cancellation wiring.

Licensed under MIT.
