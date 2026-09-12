//! Shared audio separation library for the CLI and desktop application.
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

#[cfg(feature = "burn-cpu")]
pub mod vr;

#[cfg(feature = "burn-cpu")]
mod checkpoint;

#[cfg(feature = "burn-cpu")]
pub mod roformer;
