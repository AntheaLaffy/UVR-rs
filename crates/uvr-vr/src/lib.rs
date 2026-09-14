//! CPU inference and PCM separation for the UVR VR model family.
//!
//! Load an original UVR `.pth` checkpoint with [`HpKaraokeModel::load`] or
//! [`DeEchoModel::load`], then feed 44.1 kHz mono PCM to the prediction API.
//! For file decoding, output publication and cancellation, use `uvr-runtime`.

pub use uvr_core::vr::{
    DeEchoModel, HpKaraokeModel, HpKaraokeVariant, TaskCancelled, VrOptions, VrOutput, VrProgress,
    VrSeparator, VrStage, VrTimings,
};
pub use uvr_core::vr_dsp::VrVariant;
pub use uvr_core::{task, vr, vr_dsp, weights};
