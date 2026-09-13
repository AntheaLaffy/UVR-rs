use serde::{Deserialize, Serialize};
use uvr_core::{
    file_task::ModelSpec,
    runtime::{self, LinearLayout, RoformerOptions, RuntimeBackend, RuntimeOptions},
    vr::VrOptions,
};

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Burn,
    OpenvinoCpu,
}

impl From<RuntimeBackend> for Backend {
    fn from(value: RuntimeBackend) -> Self {
        match value {
            RuntimeBackend::Burn => Self::Burn,
            RuntimeBackend::OpenvinoCpu => Self::OpenvinoCpu,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    Flattened,
    Batched,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VrSettings {
    window_frames: usize,
    inference_batch: usize,
    window_parallelism: usize,
}

impl From<VrOptions> for VrSettings {
    fn from(value: VrOptions) -> Self {
        Self {
            window_frames: value.window_frames,
            inference_batch: value.inference_batch,
            window_parallelism: value.window_parallelism,
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RoformerSettings {
    time_batch: usize,
    frequency_batch: usize,
    window_parallelism: usize,
    linear_layout: Layout,
}

impl From<RoformerOptions> for RoformerSettings {
    fn from(value: RoformerOptions) -> Self {
        Self {
            time_batch: value.time_batch,
            frequency_batch: value.frequency_batch,
            window_parallelism: value.window_parallelism,
            linear_layout: match value.linear_layout {
                LinearLayout::Flattened => Layout::Flattened,
                LinearLayout::Batched => Layout::Batched,
            },
        }
    }
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeRequest {
    backend: Option<Backend>,
    threads: Option<usize>,
    vr: Option<VrSettings>,
    roformer: Option<RoformerSettings>,
}

impl RuntimeRequest {
    pub fn resolve(self, spec: ModelSpec) -> Result<RuntimeOptions, String> {
        let mut options = RuntimeOptions::for_model(spec);
        options.threads = self.threads.unwrap_or_else(default_threads);
        if let Some(backend) = self.backend {
            options.backend = match backend {
                Backend::Burn => RuntimeBackend::Burn,
                Backend::OpenvinoCpu => RuntimeBackend::OpenvinoCpu,
            };
        }
        if self.vr.is_some() && matches!(spec, ModelSpec::Roformer1296) {
            return Err("1296 不接受 VR 参数".into());
        }
        if self.roformer.is_some()
            && (!matches!(spec, ModelSpec::Roformer1296) || options.backend != RuntimeBackend::Burn)
        {
            return Err("RoFormer 批次与布局参数仅适用于 1296 的 Burn 后端".into());
        }
        if let Some(vr) = self.vr {
            options.vr = VrOptions {
                window_frames: vr.window_frames,
                inference_batch: vr.inference_batch,
                window_parallelism: vr.window_parallelism,
            };
        }
        if let Some(roformer) = self.roformer {
            options.roformer = RoformerOptions {
                time_batch: roformer.time_batch,
                frequency_batch: roformer.frequency_batch,
                window_parallelism: roformer.window_parallelism,
                linear_layout: match roformer.linear_layout {
                    Layout::Flattened => LinearLayout::Flattened,
                    Layout::Batched => LinearLayout::Batched,
                },
            };
        }
        options
            .effective_for(spec)
            .map_err(|error| error.to_string())
    }
}

fn default_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| RuntimeOptions::default().threads)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeDefaults {
    backends: Vec<Backend>,
    roformer_backend: Backend,
    threads: usize,
    vr: VrSettings,
    roformer: RoformerSettings,
}

pub fn defaults() -> RuntimeDefaults {
    let options = RuntimeOptions::for_model(ModelSpec::Roformer1296);
    let mut backends = vec![Backend::Burn];
    if runtime::openvino_available() {
        backends.push(Backend::OpenvinoCpu);
    }
    RuntimeDefaults {
        backends,
        roformer_backend: options.backend.into(),
        threads: default_threads(),
        vr: options.vr.into(),
        roformer: options.roformer.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gui_request_reaches_shared_runtime() {
        let request: RuntimeRequest = serde_json::from_str(r#"{
            "backend":"burn","threads":3,
            "roformer":{"timeBatch":7,"frequencyBatch":37,"windowParallelism":2,"linearLayout":"batched"}
        }"#).unwrap();
        let options = request.resolve(ModelSpec::Roformer1296).unwrap();
        assert_eq!(options.threads, 3);
        assert_eq!(options.backend, RuntimeBackend::Burn);
        assert_eq!(options.roformer.time_batch, 7);
        assert_eq!(options.roformer.frequency_batch, 37);
        assert_eq!(options.roformer.window_parallelism, 2);
        assert_eq!(options.roformer.linear_layout, LinearLayout::Batched);
    }

    #[test]
    fn vr_request_uses_model_specific_validation() {
        let parse = || {
            serde_json::from_str::<RuntimeRequest>(
                r#"{
            "backend":"burn","threads":1,
            "vr":{"windowFrames":144,"inferenceBatch":1,"windowParallelism":1}
        }"#,
            )
            .unwrap()
        };
        assert!(
            parse()
                .resolve(ModelSpec::from_key("deecho").unwrap())
                .is_ok()
        );
        assert!(
            parse()
                .resolve(ModelSpec::from_key("5hp").unwrap())
                .is_err()
        );
        assert!(parse().resolve(ModelSpec::Roformer1296).is_err());
    }

    #[test]
    fn malformed_and_inapplicable_settings_are_rejected() {
        for json in [
            r#"{"threads":-1}"#,
            r#"{"threads":1.5}"#,
            r#"{"backend":"gpu"}"#,
            r#"{"thread":8}"#,
        ] {
            assert!(serde_json::from_str::<RuntimeRequest>(json).is_err());
        }
        let request: RuntimeRequest = serde_json::from_str(r#"{"threads":0}"#).unwrap();
        assert!(
            request
                .resolve(ModelSpec::from_key("5hp").unwrap())
                .is_err()
        );
        let request: RuntimeRequest = serde_json::from_str(r#"{
            "backend":"openvino-cpu",
            "roformer":{"timeBatch":62,"frequencyBatch":301,"windowParallelism":1,"linearLayout":"flattened"}
        }"#).unwrap();
        assert!(request.resolve(ModelSpec::Roformer1296).is_err());
    }
}
