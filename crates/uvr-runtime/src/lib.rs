//! Complete file separation and runtime selection used by the UVR applications.
//!
//! This is the highest-level crate: it decodes WAV/FLAC/MP3, validates model
//! identity, runs inference, and atomically publishes two protected WAV files.
//! Use [`separate_file`] for recommended defaults or
//! [`separate_file_with_options`] when reproducible runtime settings are needed.

pub use uvr_core::file_task::{
    FileOutput, FileProgress, FileStage, FileTimings, ModelSpec, RoformerBackend, separate_file,
    separate_file_with_backend, separate_file_with_options,
};
pub use uvr_core::runtime::{RuntimeBackend, RuntimeOptions, openvino_available};
pub use uvr_core::{audio_io, file_task, model_catalog, runtime, task, weights};

#[cfg(feature = "model-download")]
pub use uvr_core::model_store;
