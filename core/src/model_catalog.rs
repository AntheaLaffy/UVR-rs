//! Identities of the four supported external weight files. No weights are embedded.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    pub key: &'static str,
    pub label: &'static str,
    pub filename: &'static str,
    pub size_bytes: u64,
    pub sha256: &'static str,
}

pub const MODELS: [ModelInfo; 4] = [
    ModelInfo {
        key: "1296",
        label: "BS-RoFormer 1296",
        filename: "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        size_bytes: 639_317_465,
        sha256: "f6c94864adfb73bbb0ca58ec14d58dd0b364549e9fb61433ae51916f3e2f8d0b",
    },
    ModelInfo {
        key: "5hp",
        label: "Karaoke 5-HP",
        filename: "5_HP-Karaoke-UVR.pth",
        size_bytes: 126_782_699,
        sha256: "fe00891defbb61f4261500af22f7624f1a3df8dc75fa3998d1aece02e6be4537",
    },
    ModelInfo {
        key: "6hp",
        label: "Karaoke 6-HP",
        filename: "6_HP-Karaoke-UVR.pth",
        size_bytes: 126_782_699,
        sha256: "4ce7eaaa9e56f09366b788aebf6d3a72aec8145692c56f1e090e4e7e2d7ce65f",
    },
    ModelInfo {
        key: "deecho",
        label: "DeEcho / DeReverb",
        filename: "UVR-DeEcho-DeReverb.pth",
        size_bytes: 223_650_277,
        sha256: "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7",
    },
];

impl ModelInfo {
    pub fn from_key(key: &str) -> Option<Self> {
        MODELS.into_iter().find(|model| model.key == key)
    }

    pub fn hugging_face_url(self) -> String {
        format!(
            "https://huggingface.co/Blane187/all_public_uvr_models/resolve/fddec39677560e41e3194f24a9e4c4cd32ef0e83/{}",
            self.filename
        )
    }

    pub fn github_url(self) -> String {
        format!(
            "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{}",
            self.filename
        )
    }
}

/// Caller applies the persisted user choice first. is_dir follows directory symlinks.
pub fn default_directory(executable: &Path, working_directory: &Path) -> PathBuf {
    let portable = executable
        .parent()
        .unwrap_or(working_directory)
        .join("models");
    if portable.is_dir() {
        return portable;
    }
    let development = working_directory.join("models");
    if MODELS
        .iter()
        .any(|model| development.join(model.filename).is_file())
    {
        development
    } else {
        portable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_matches_independently_verified_identities() {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../references/verified-weights.json")).unwrap();
        for model in MODELS {
            let verified = manifest["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["file"] == model.filename)
                .unwrap();
            assert_eq!(verified["sha256"], model.sha256);
            assert_eq!(verified["size_bytes"], model.size_bytes);
        }
    }
}
