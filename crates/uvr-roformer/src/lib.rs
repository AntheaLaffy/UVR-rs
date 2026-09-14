//! CPU inference and PCM separation for BS-RoFormer 1296.
//!
//! The model accepts the original `model_bs_roformer_ep_368_sdr_12.9628.ckpt`
//! checkpoint. [`RoformerModel::load`] provides the Burn CPU path; enable the
//! `openvino` feature only when the native OpenVINO runtime is installed.

pub use uvr_core::roformer::{
    LinearLayout, RoformerModel, RoformerOptions, RoformerOutput, RoformerProgress, RoformerStage,
    RoformerTimings,
};
pub use uvr_core::{roformer, task, weights};

#[cfg(feature = "openvino")]
pub use uvr_core::roformer::openvino;
