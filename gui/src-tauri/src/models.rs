use std::{
    collections::BTreeMap,
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use uvr_core::{
    model_catalog::{MODELS, ModelInfo},
    model_store::{
        self, Availability, DownloadOptions, DownloadProgress, DownloadSource, DownloadStage,
    },
    task::TaskCancelled,
};

use crate::locale::Locale;
use crate::tasks::TaskState;

#[derive(Default)]
pub struct LibraryState {
    generation: Arc<AtomicU64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    key: &'static str,
    name: &'static str,
    filename: &'static str,
    size_bytes: u64,
    status: &'static str,
    message: String,
}

#[tauri::command]
pub async fn inspect_models(
    state: State<'_, LibraryState>,
    directory: PathBuf,
    lang: Option<Locale>,
) -> Result<Vec<ModelEntry>, String> {
    let lang = lang.unwrap_or_default();
    if directory.as_os_str().is_empty() {
        return Err(lang
            .text(
                "请选择模型目录",
                "Choose a model directory",
                "モデルフォルダーを選択してください",
            )
            .into());
    }
    let generation = state.generation.clone();
    let id = generation.fetch_add(1, Ordering::Relaxed) + 1;
    tauri::async_runtime::spawn_blocking(move || {
        let mut entries = Vec::new();
        for info in MODELS {
            let result = model_store::inspect(info, &directory, |_| {
                if generation.load(Ordering::Relaxed) == id {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            });
            let (status, message) = match result {
                Ok(Availability::Ready) => (
                    "ready",
                    lang.text(
                        "已校验，可直接使用",
                        "Verified and ready",
                        "検証済み・使用可能",
                    )
                    .into(),
                ),
                Ok(Availability::Missing) => (
                    "missing",
                    lang.text("尚未下载", "Not downloaded", "未ダウンロード")
                        .into(),
                ),
                Ok(Availability::Invalid(reason)) => ("invalid", reason),
                Err(error) if error.is::<TaskCancelled>() => {
                    return Err(lang
                        .text(
                            "检测已由新目录取代",
                            "Scan replaced by a newer directory selection",
                            "フォルダーが変更されたため検出を中止しました",
                        )
                        .into());
                }
                Err(error) => ("error", format!("{error:#}")),
            };
            entries.push(ModelEntry {
                key: info.key,
                name: info.label,
                filename: info.filename,
                size_bytes: info.size_bytes,
                status,
                message,
            });
        }
        Ok(entries)
    })
    .await
    .map_err(|error| error.to_string())?
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DownloadRequest {
    id: String,
    model: String,
    directory: PathBuf,
    source: String,
    proxy: Option<String>,
    replace_invalid: bool,
    #[serde(default)]
    lang: Locale,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadEvent {
    id: String,
    model: &'static str,
    status: &'static str,
    phase: String,
    phase_key: &'static str,
    phase_args: BTreeMap<&'static str, String>,
    received: u64,
    total: u64,
    path: Option<String>,
}

#[tauri::command]
pub fn download_model(
    app: AppHandle,
    state: State<'_, TaskState>,
    request: DownloadRequest,
) -> Result<(), String> {
    let lang = request.lang;
    let info = ModelInfo::from_key(&request.model)
        .ok_or_else(|| lang.text("不支持的模型", "Unsupported model", "未対応のモデルです"))?;
    if request.directory.as_os_str().is_empty() {
        return Err(lang
            .text(
                "请选择模型目录",
                "Choose a model directory",
                "モデルフォルダーを選択してください",
            )
            .into());
    }
    let source = match request.source.as_str() {
        "huggingface" => DownloadSource::HuggingFace,
        "github" => DownloadSource::GitHub,
        _ => {
            return Err(lang
                .text(
                    "不支持的下载来源",
                    "Unsupported download source",
                    "未対応のダウンロード元です",
                )
                .into());
        }
    };
    let cancelled = state
        .begin(&request.id)
        .map_err(|error| lang.error(error))?;
    let worker_id = request.id.clone();
    let worker = std::thread::Builder::new()
        .name("uvr-model-download".into())
        .spawn(move || {
            let options = DownloadOptions {
                source,
                proxy: request.proxy,
                replace_invalid: request.replace_invalid,
            };
            let mut last_stage = None;
            let mut last_emit = Instant::now() - Duration::from_secs(1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tauri::async_runtime::block_on(model_store::download(
                    info,
                    &request.directory,
                    &options,
                    |p| {
                        if cancelled.load(Ordering::Relaxed) {
                            return ControlFlow::Break(());
                        }
                        if p.stage != DownloadStage::Complete
                            && (last_stage != Some(p.stage)
                                || last_emit.elapsed() >= Duration::from_millis(200))
                        {
                            let _ = app.emit(
                                "model-download",
                                DownloadEvent {
                                    id: request.id.clone(),
                                    model: info.key,
                                    status: "running",
                                    phase: describe(p, lang).into(),
                                    phase_key: stage_key(p.stage),
                                    phase_args: BTreeMap::new(),
                                    received: p.completed_bytes,
                                    total: p.total_bytes,
                                    path: None,
                                },
                            );
                            last_stage = Some(p.stage);
                            last_emit = Instant::now();
                        }
                        ControlFlow::Continue(())
                    },
                ))
            }));
            let mut phase_key = "download.phase.error";
            let (status, phase, path) = match result {
                Ok(Ok(output)) => (
                    "completed",
                    if output.downloaded {
                        phase_key = "download.phase.completed";
                        lang.text(
                            "下载完成，校验通过",
                            "Downloaded and verified",
                            "ダウンロードと検証が完了しました",
                        )
                    } else {
                        phase_key = "download.phase.existing";
                        lang.text(
                            "模型已存在，校验通过",
                            "Existing model verified",
                            "既存のモデルを検証しました",
                        )
                    }
                    .into(),
                    Some(output.path.to_string_lossy().into_owned()),
                ),
                Ok(Err(error)) if error.is::<TaskCancelled>() => {
                    phase_key = "download.phase.cancelled";
                    (
                        "cancelled",
                        lang.text(
                            "下载已取消，临时文件已清理",
                            "Download cancelled; temporary files removed",
                            "ダウンロードをキャンセルし、一時ファイルを削除しました",
                        )
                        .into(),
                        None,
                    )
                }
                Ok(Err(error)) => ("failed", format!("{error:#}"), None),
                Err(_) => (
                    "failed",
                    lang.text(
                        "下载线程出现异常，请查看终端日志",
                        "The download worker failed; check the terminal log",
                        "ダウンロードスレッドでエラーが発生しました。端末のログを確認してください",
                    )
                    .into(),
                    None,
                ),
            };
            let phase_args = if status == "failed" {
                BTreeMap::from([("error", phase.clone())])
            } else {
                BTreeMap::new()
            };
            let closing = app.state::<TaskState>().finish(&request.id);
            let _ = app.emit(
                "model-download",
                DownloadEvent {
                    id: request.id,
                    model: info.key,
                    status,
                    phase,
                    phase_key,
                    phase_args,
                    received: if status == "completed" {
                        info.size_bytes
                    } else {
                        0
                    },
                    total: info.size_bytes,
                    path,
                },
            );
            if closing && let Some(window) = app.get_webview_window("main") {
                let _ = window.close();
            }
        });
    if let Err(error) = worker {
        state.finish(&worker_id);
        return Err(format!(
            "{}: {error}",
            lang.text(
                "无法启动下载线程",
                "Cannot start the download worker",
                "ダウンロードスレッドを開始できません"
            )
        ));
    }
    Ok(())
}

fn stage_key(stage: DownloadStage) -> &'static str {
    match stage {
        DownloadStage::Checking => "download.phase.checking",
        DownloadStage::Connecting => "download.phase.connecting",
        DownloadStage::Receiving => "download.phase.receiving",
        DownloadStage::Verifying => "download.phase.verifying",
        DownloadStage::Complete => "download.phase.ready",
    }
}

fn describe(progress: DownloadProgress, lang: Locale) -> &'static str {
    match progress.stage {
        DownloadStage::Checking => lang.text(
            "检查已有模型",
            "Checking existing model",
            "既存のモデルを確認中",
        ),
        DownloadStage::Connecting => lang.text(
            "正在连接下载源",
            "Connecting to download source",
            "ダウンロード元に接続中",
        ),
        DownloadStage::Receiving => lang.text(
            "正在下载模型",
            "Downloading model",
            "モデルをダウンロード中",
        ),
        DownloadStage::Verifying => lang.text(
            "校验下载文件",
            "Verifying download",
            "ダウンロードしたファイルを検証中",
        ),
        DownloadStage::Complete => {
            lang.text("模型已就绪", "Model ready", "モデルの準備ができました")
        }
    }
}
