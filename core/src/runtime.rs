//! Explicit inference settings shared by interactive and command-line callers.

use anyhow::{Result, ensure};

pub use crate::roformer::{LinearLayout, RoformerOptions};
use crate::{file_task::ModelSpec, vr::VrOptions};

/// Reports native CPU availability, including portable executable-side libraries.
/// Detection happens once; starting a task still reports any initialization error.
pub fn openvino_available() -> bool {
    #[cfg(feature = "openvino")]
    {
        static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            crate::roformer::openvino::create_core().is_ok_and(|core| {
                core.get_property(
                    &openvino::DeviceType::CPU,
                    &openvino::PropertyKey::Other("FULL_DEVICE_NAME".into()),
                )
                .is_ok()
            })
        })
    }
    #[cfg(not(feature = "openvino"))]
    {
        false
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RuntimeBackend {
    #[default]
    Burn,
    OpenvinoCpu,
}

impl RuntimeBackend {
    pub fn available(self) -> bool {
        match self {
            Self::Burn => true,
            Self::OpenvinoCpu => openvino_available(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// OpenVINO CPU supports the 1296 model; VR runs on Burn.
    pub backend: RuntimeBackend,
    /// Budget for the task's Rayon pool and, when selected, OpenVINO inference.
    pub threads: usize,
    pub vr: VrOptions,
    /// Applies only to the Burn implementation of 1296.
    pub roformer: RoformerOptions,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            backend: RuntimeBackend::Burn,
            threads: std::thread::available_parallelism().map_or(8, |count| count.get().min(8)),
            vr: VrOptions::default(),
            roformer: RoformerOptions::default(),
        }
    }
}

impl RuntimeOptions {
    /// Choose the measured CPU backend when its native runtime is usable.
    pub fn for_model(spec: ModelSpec) -> Self {
        let mut options = Self::default();
        match spec {
            ModelSpec::Roformer1296 if openvino_available() => {
                options.backend = RuntimeBackend::OpenvinoCpu;
            }
            ModelSpec::Vr { options: vr, .. } => options.vr = vr,
            ModelSpec::Roformer1296 => {}
        }
        options
    }

    /// Check applicable settings before any input decoding or output creation.
    pub fn validate(self, spec: ModelSpec) -> Result<()> {
        self.effective_for(spec).map(|_| ())
    }

    /// Resolve scheduling modes whose controls cannot be combined. Settings for
    /// inactive model families remain available for a later task in the UI.
    pub fn effective_for(mut self, spec: ModelSpec) -> Result<Self> {
        ensure!(self.threads > 0, "thread count must be positive");
        match spec {
            ModelSpec::Vr { variant, .. } => {
                ensure!(
                    self.backend == RuntimeBackend::Burn,
                    "OpenVINO backend is only implemented for 1296"
                );
                self.vr = self.vr.effective_for(variant)?;
            }
            ModelSpec::Roformer1296 => {
                if self.backend == RuntimeBackend::Burn {
                    self.roformer.validate()?;
                } else {
                    ensure!(
                        cfg!(feature = "openvino"),
                        "this build does not enable the OpenVINO CPU backend"
                    );
                    ensure!(
                        openvino_available(),
                        "OpenVINO CPU runtime is unavailable; install it or choose the Burn backend"
                    );
                }
            }
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vr_dsp::VrVariant;

    #[test]
    fn validates_active_model_before_starting_a_task() {
        let mut options = RuntimeOptions {
            threads: 0,
            ..Default::default()
        };
        assert!(options.validate(ModelSpec::Roformer1296).is_err());
        options.threads = 2;
        options.roformer.time_batch = 0;
        assert!(options.validate(ModelSpec::Roformer1296).is_err());
        let hp = ModelSpec::from_key("5hp").unwrap();
        assert!(options.validate(hp).is_ok());
        options.vr.window_frames = 256;
        assert!(options.validate(hp).is_err());
        let deecho = ModelSpec::from_key("deecho").unwrap();
        assert!(options.validate(deecho).is_ok());
    }

    #[test]
    fn deecho_and_batched_hp_report_effective_parallelism() {
        let mut options = RuntimeOptions::default();
        options.vr.inference_batch = 4;
        options.vr.window_parallelism = 8;
        let hp = options
            .effective_for(ModelSpec::from_key("6hp").unwrap())
            .unwrap();
        assert_eq!((hp.vr.inference_batch, hp.vr.window_parallelism), (4, 1));
        let deecho = options
            .effective_for(ModelSpec::from_key("deecho").unwrap())
            .unwrap();
        assert_eq!(
            (deecho.vr.inference_batch, deecho.vr.window_parallelism),
            (1, 1)
        );
        assert_eq!(options.vr.window_parallelism, 8);
    }

    #[test]
    fn unsupported_backends_are_rejected() {
        let options = RuntimeOptions {
            backend: RuntimeBackend::OpenvinoCpu,
            ..Default::default()
        };
        assert!(
            options
                .validate(ModelSpec::Vr {
                    variant: VrVariant::HpFive,
                    options: VrOptions::default(),
                })
                .is_err()
        );
        assert_eq!(
            options.validate(ModelSpec::Roformer1296).is_ok(),
            openvino_available()
        );
    }
}
