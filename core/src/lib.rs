//! Shared audio separation library for the CLI and desktop application.
//!
//! Most applications should use one of the focused facade crates (`uvr-dsp`,
//! `uvr-models`, `uvr-vr`, `uvr-roformer`, or `uvr-runtime`). This crate is the
//! shared implementation and exposes feature-gated modules for advanced users.
//!
//! The optional CPU network API is experimental until audio-pipeline validation.

pub mod dsp;
pub mod model_catalog;
pub mod resample;
pub mod task;
pub mod vr_dsp;
pub mod weights;

#[cfg(feature = "model-download")]
pub mod model_store;

#[cfg(feature = "audio-io")]
pub mod audio_io;

#[cfg(all(feature = "audio-io", feature = "burn-cpu"))]
pub mod file_task;

#[cfg(all(feature = "audio-io", feature = "burn-cpu"))]
pub mod runtime;

#[cfg(feature = "burn-cpu")]
pub mod vr;

#[cfg(feature = "burn-cpu")]
mod checkpoint;

#[cfg(feature = "burn-cpu")]
pub mod roformer;
