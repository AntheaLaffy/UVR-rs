//! Application version and opt-in release update checks.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const RELEASES_URL: &str = "https://api.github.com/repos/IronHpc/UVR-rs/releases?per_page=20";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub release_url: String,
    pub has_update: bool,
}

#[cfg(feature = "update-check")]
pub async fn check_latest() -> anyhow::Result<Option<UpdateInfo>> {
    let current = semver::Version::parse(VERSION)?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .user_agent(concat!("UVR/", env!("CARGO_PKG_VERSION")))
        .build()?;
    let response: serde_json::Value = client
        .get(RELEASES_URL)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let releases = response
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("GitHub releases response was not an array"))?;
    let mut latest: Option<(semver::Version, String)> = None;
    for release in releases {
        if release
            .get("draft")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
            || release
                .get("prerelease")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(true)
        {
            continue;
        }
        let Some(tag) = release.get("tag_name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Some(url) = release.get("html_url").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Ok(version) = semver::Version::parse(tag.trim_start_matches('v')) else {
            continue;
        };
        if latest.as_ref().is_none_or(|(found, _)| version > *found) {
            latest = Some((version, url.to_owned()));
        }
    }
    let Some((latest, release_url)) = latest else {
        return Ok(None);
    };
    Ok(Some(UpdateInfo {
        current: current.to_string(),
        latest: latest.to_string(),
        release_url,
        has_update: latest > current,
    }))
}
