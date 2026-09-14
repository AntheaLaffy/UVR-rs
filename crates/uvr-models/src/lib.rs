//! Model catalog and identity helpers used by UVR Rust.
//!
//! [`ModelInfo`] describes the four checkpoints supported by this release.
//! Weight files are deliberately external; [`fingerprint`] lets applications
//! verify a local file before handing it to an inference crate.
//!
//! # Example
//!
//! ```
//! use uvr_models::ModelInfo;
//!
//! let model = ModelInfo::from_key("5hp").expect("supported model");
//! assert_eq!(model.key, "5hp");
//! ```

pub use uvr_core::model_catalog::ModelInfo;
pub use uvr_core::weights::{WeightFingerprint, fingerprint};
pub use uvr_core::{model_catalog, weights};

#[cfg(feature = "download")]
pub use uvr_core::model_store;
