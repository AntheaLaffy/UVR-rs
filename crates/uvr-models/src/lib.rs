//! Model catalog and identity helpers used by UVR Rust.

pub use uvr_core::model_catalog::ModelInfo;
pub use uvr_core::weights::{WeightFingerprint, fingerprint};
pub use uvr_core::{model_catalog, weights};

#[cfg(feature = "download")]
pub use uvr_core::model_store;
