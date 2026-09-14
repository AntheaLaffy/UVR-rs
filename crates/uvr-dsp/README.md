# uvr-dsp

[![crates.io](https://img.shields.io/crates/v/uvr-dsp?logo=rust&label=crates.io)](https://crates.io/crates/uvr-dsp)
[![Documentation](https://docs.rs/uvr-dsp/badge.svg)](https://docs.rs/uvr-dsp)

Reusable STFT, spectrogram, resampling and VR spectral-analysis primitives from [UVR Rust](https://github.com/IronHpc/UVR-rs).

```toml
[dependencies]
uvr-dsp = "0.1.2"
```

This crate performs DSP only. It does not include model weights, neural-network inference, file codecs or Python/PyTorch.

The public API re-exports the core DSP types, so applications can depend on this small facade while keeping model and runtime choices separate. See the [API documentation](https://docs.rs/uvr-dsp) for parameter constraints and error behavior.

Licensed under MIT.
