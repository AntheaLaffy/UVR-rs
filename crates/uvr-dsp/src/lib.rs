//! Audio DSP primitives used by UVR Rust.
//!
//! The crate is intentionally model-agnostic: use [`Stft`] and [`Polyphase`]
//! to build an audio pipeline without pulling in a neural-network runtime.
//!
//! # Example
//!
//! ```
//! use uvr_dsp::{Padding, Stft};
//!
//! let mut stft = Stft::new(2048, 441, Padding::Reflect).unwrap();
//! let spectrum = stft.forward(&[0.0; 2048]).unwrap();
//! assert_eq!(spectrum.bins, 1025);
//! ```

pub use uvr_core::dsp::{Complex32, Padding, Spectrogram, Stft};
pub use uvr_core::resample::Polyphase;
pub use uvr_core::{dsp, resample, vr_dsp};
