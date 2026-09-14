# uvr-core

Shared implementation for [UVR Rust](https://github.com/IronHpc/UVR-rs). It contains audio DSP, model identification and the feature-gated VR and BS-RoFormer CPU inference paths used by the CLI and desktop app.

Most applications should depend on one of the focused facade crates instead:

- [`uvr-dsp`](https://crates.io/crates/uvr-dsp) for STFT and resampling primitives.
- [`uvr-models`](https://crates.io/crates/uvr-models) for supported-model metadata, fingerprints and optional downloads.
- [`uvr-vr`](https://crates.io/crates/uvr-vr) for 5-HP, 6-HP and DeEcho inference.
- [`uvr-roformer`](https://crates.io/crates/uvr-roformer) for BS-RoFormer 1296 inference.
- [`uvr-runtime`](https://crates.io/crates/uvr-runtime) for complete file separation and runtime selection.

Model weights are not bundled. See the repository's [runtime guide](https://github.com/IronHpc/UVR-rs/blob/main/docs/runtime.md) and [third-party notice](https://github.com/IronHpc/UVR-rs/blob/main/THIRD_PARTY_NOTICES.md).

Licensed under MIT.
