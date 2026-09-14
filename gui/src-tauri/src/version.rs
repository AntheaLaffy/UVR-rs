use serde::Serialize;

use crate::locale::Locale;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateResult {
    pub current: String,
    pub latest: Option<String>,
    pub has_update: bool,
    pub release_url: Option<String>,
}

#[tauri::command]
pub async fn check_update(lang: Option<Locale>) -> Result<UpdateResult, String> {
    let _lang = lang.unwrap_or_default();
    uvr_core::version::check_latest()
        .await
        .map_err(|error| error.to_string())
        .map(|info| match info {
            Some(info) => UpdateResult {
                current: info.current,
                latest: Some(info.latest),
                has_update: info.has_update,
                release_url: Some(info.release_url),
            },
            None => UpdateResult {
                current: uvr_core::version::VERSION.into(),
                latest: None,
                has_update: false,
                release_url: None,
            },
        })
}
