use std::{
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
) -> Result<Vec<ModelEntry>, String> {
    if directory.as_os_str().is_empty() {
        return Err("请选择模型目录".into());
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
                Ok(Availability::Ready) => ("ready", "已校验，可直接使用".into()),
                Ok(Availability::Missing) => ("missing", "尚未下载".into()),
                Ok(Availability::Invalid(reason)) => ("invalid", reason),
                Err(error) if error.is::<TaskCancelled>() => {
                    return Err("检测已由新目录取代".into());
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
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DownloadEvent {
    id: String,
    model: &'static str,
    status: &'static str,
    phase: String,
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
    let info = ModelInfo::from_key(&request.model).ok_or("不支持的模型")?;
    if request.directory.as_os_str().is_empty() {
        return Err("请选择模型目录".into());
    }
    let source = match request.source.as_str() {
        "huggingface" => DownloadSource::HuggingFace,
        "github" => DownloadSource::GitHub,
        _ => return Err("不支持的下载来源".into()),
    };
    let cancelled = state.begin(&request.id)?;
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
                                    phase: describe(p).into(),
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
            let (status, phase, path) = match result {
                Ok(Ok(output)) => (
                    "completed",
                    if output.downloaded {
                        "下载完成，校验通过"
                    } else {
                        "模型已存在，校验通过"
                    }
                    .into(),
                    Some(output.path.to_string_lossy().into_owned()),
                ),
                Ok(Err(error)) if error.is::<TaskCancelled>() => {
                    ("cancelled", "下载已取消，临时文件已清理".into(), None)
                }
                Ok(Err(error)) => ("failed", format!("{error:#}"), None),
                Err(_) => ("failed", "下载线程出现异常，请查看终端日志".into(), None),
            };
            let closing = app.state::<TaskState>().finish(&request.id);
            let _ = app.emit(
                "model-download",
                DownloadEvent {
                    id: request.id,
                    model: info.key,
                    status,
                    phase,
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
        return Err(format!("无法启动下载线程：{error}"));
    }
    Ok(())
}

fn describe(progress: DownloadProgress) -> &'static str {
    match progress.stage {
        DownloadStage::Checking => "检查已有模型",
        DownloadStage::Connecting => "正在连接下载源",
        DownloadStage::Receiving => "正在下载模型",
        DownloadStage::Verifying => "校验下载文件",
        DownloadStage::Complete => "模型已就绪",
    }
}
