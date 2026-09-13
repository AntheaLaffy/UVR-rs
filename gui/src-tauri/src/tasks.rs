use std::{
    collections::BTreeMap,
    ops::ControlFlow,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use crate::locale::Locale;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;
use uvr_core::{
    file_task::{self, FileProgress, FileStage, ModelSpec},
    task::TaskCancelled,
};

struct ActiveTask {
    id: String,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct TaskState {
    active: Mutex<Option<ActiveTask>>,
    closing: AtomicBool,
}

impl TaskState {
    pub fn begin(&self, id: &str) -> Result<Arc<AtomicBool>, String> {
        if id.is_empty()
            || id.len() > 80
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err("任务标识不正确".into());
        }
        let mut active = self.active.lock().map_err(|_| "任务状态不可用")?;
        if active.is_some() {
            return Err("已有任务正在处理，请先等待或取消".into());
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        self.closing.store(false, Ordering::Relaxed);
        *active = Some(ActiveTask {
            id: id.into(),
            cancelled: cancelled.clone(),
        });
        Ok(cancelled)
    }

    pub fn finish(&self, id: &str) -> bool {
        if let Ok(mut active) = self.active.lock()
            && active.as_ref().is_some_and(|task| task.id == id)
        {
            *active = None;
            return self.closing.load(Ordering::Relaxed);
        }
        false
    }

    pub fn cancel_for_close(&self) -> bool {
        if let Ok(active) = self.active.lock()
            && let Some(task) = active.as_ref()
        {
            self.closing.store(true, Ordering::Relaxed);
            task.cancelled.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TaskRequest {
    id: String,
    model: String,
    input: PathBuf,
    output_dir: PathBuf,
    models_dir: PathBuf,
    #[serde(default)]
    runtime: crate::runtime::RuntimeRequest,
    #[serde(default)]
    lang: Locale,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TaskEvent {
    id: String,
    status: &'static str,
    phase: String,
    phase_key: &'static str,
    phase_args: BTreeMap<&'static str, String>,
    fraction: Option<f64>,
    elapsed_seconds: f64,
    outputs: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Defaults {
    models_dir: Option<String>,
    runtime: crate::runtime::RuntimeDefaults,
}

#[tauri::command]
pub fn defaults(lang: Option<Locale>) -> Defaults {
    let _ = lang;
    let models_dir = std::env::current_exe()
        .ok()
        .zip(std::env::current_dir().ok())
        .map(|(exe, cwd)| uvr_core::model_catalog::default_directory(&exe, &cwd))
        .map(|p| p.to_string_lossy().into_owned());
    Defaults {
        models_dir,
        runtime: crate::runtime::defaults(),
    }
}

#[tauri::command]
pub async fn choose_path(
    app: AppHandle,
    kind: String,
    lang: Option<Locale>,
) -> Result<Option<String>, String> {
    let lang = lang.unwrap_or_default();
    if !matches!(kind.as_str(), "input" | "output" | "models") {
        return Err(lang
            .text("未知路径类型", "Unknown path type", "不明なパスの種類です")
            .into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = app.dialog().file();
        let result = if kind == "input" {
            dialog
                .add_filter(lang.text("音频", "Audio", "音声"), &["wav", "flac", "mp3"])
                .blocking_pick_file()
        } else {
            dialog.blocking_pick_folder()
        };
        result
            .map(|file| {
                file.into_path()
                    .map(|p| p.to_string_lossy().into_owned())
                    .map_err(|e| e.to_string())
            })
            .transpose()
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn cancel_task(
    state: State<'_, TaskState>,
    id: String,
    lang: Option<Locale>,
) -> Result<bool, String> {
    let lang = lang.unwrap_or_default();
    let active = state
        .active
        .lock()
        .map_err(|_| lang.error("任务状态不可用".into()))?;
    if let Some(task) = active.as_ref().filter(|task| task.id == id) {
        task.cancelled.store(true, Ordering::Relaxed);
        return Ok(true);
    }
    Ok(false)
}

#[tauri::command]
pub fn start_task(
    app: AppHandle,
    state: State<'_, TaskState>,
    request: TaskRequest,
) -> Result<(), String> {
    let lang = request.lang;
    if request.input.as_os_str().is_empty()
        || request.output_dir.as_os_str().is_empty()
        || request.models_dir.as_os_str().is_empty()
    {
        return Err(lang
            .text(
                "请选择音频、模型目录和输出目录",
                "Choose audio, model and output locations",
                "音声ファイル、モデルフォルダー、保存先を選択してください",
            )
            .into());
    }
    let spec = ModelSpec::from_key(&request.model)
        .ok_or_else(|| lang.text("不支持的模型", "Unsupported model", "未対応のモデルです"))?;
    let runtime = request
        .runtime
        .resolve(spec)
        .map_err(|error| lang.error(error))?;
    let cancelled = state
        .begin(&request.id)
        .map_err(|error| lang.error(error))?;
    let worker_id = request.id.clone();
    let worker_app = app.clone();
    let worker = std::thread::Builder::new()
        .name("uvr-separation".into())
        .spawn(move || {
            let start = Instant::now();
            let mut last_phase = None;
            let mut last_emit = Instant::now() - Duration::from_secs(1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                file_task::separate_file_with_options(
                    spec,
                    &request.models_dir.join(spec.weights_name()),
                    &request.input,
                    &request.output_dir,
                    runtime,
                    |p| {
                        if cancelled.load(Ordering::Relaxed) {
                            return ControlFlow::Break(());
                        }
                        if p.stage != FileStage::Complete
                            && (last_phase != Some(p.stage)
                                || last_emit.elapsed() >= Duration::from_millis(200))
                        {
                            let (phase, fraction) = describe_progress(spec, p, lang);
                            let _ = worker_app.emit(
                                "task-progress",
                                TaskEvent {
                                    id: request.id.clone(),
                                    status: "running",
                                    phase,
                                    phase_key: stage_key(p.stage),
                                    phase_args: BTreeMap::from([
                                        ("completed", p.windows_completed.to_string()),
                                        ("total", p.windows_total.to_string()),
                                    ]),
                                    fraction,
                                    elapsed_seconds: start.elapsed().as_secs_f64(),
                                    outputs: Vec::new(),
                                },
                            );
                            last_phase = Some(p.stage);
                            last_emit = Instant::now();
                        }
                        ControlFlow::Continue(())
                    },
                )
            }));
            let mut phase_args = BTreeMap::new();
            let (status, phase, outputs) = match result {
                Ok(Ok(output)) => {
                    phase_args.insert("sampleRate", output.sample_rate.to_string());
                    phase_args.insert("samples", output.samples_per_channel.to_string());
                    (
                        "completed",
                        format!(
                            "{} · {} Hz · {} {}",
                            lang.text("完成", "Complete", "完了"),
                            output.sample_rate,
                            output.samples_per_channel,
                            lang.text(
                                "个采样 / 声道",
                                "samples / channel",
                                "サンプル / チャンネル"
                            )
                        ),
                        output
                            .paths
                            .into_iter()
                            .map(|p| p.to_string_lossy().into_owned())
                            .collect(),
                    )
                }
                Ok(Err(error)) if error.is::<TaskCancelled>() => (
                    "cancelled",
                    lang.text(
                        "已取消，未完成的输出已清理",
                        "Cancelled; incomplete outputs removed",
                        "キャンセルしました。未完了の出力は削除されました",
                    )
                    .into(),
                    Vec::new(),
                ),
                Ok(Err(error)) => ("failed", format!("{error:#}"), Vec::new()),
                Err(_) => (
                    "failed",
                    lang.text(
                        "处理线程出现异常，请查看终端日志",
                        "The processing worker failed; check the terminal log",
                        "処理スレッドでエラーが発生しました。端末のログを確認してください",
                    )
                    .into(),
                    Vec::new(),
                ),
            };
            let phase_key = match status {
                "completed" => "task.phase.complete",
                "cancelled" => "task.phase.cancelled",
                _ => {
                    phase_args.insert("error", phase.clone());
                    "task.phase.error"
                }
            };
            let state = worker_app.state::<TaskState>();
            let closing = state.finish(&request.id);
            let _ = worker_app.emit(
                "task-progress",
                TaskEvent {
                    id: request.id,
                    status,
                    phase,
                    phase_key,
                    phase_args,
                    fraction: (status == "completed").then_some(1.0),
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                    outputs,
                },
            );
            if closing && let Some(window) = worker_app.get_webview_window("main") {
                let _ = window.close();
            }
        });
    if let Err(error) = worker {
        state.finish(&worker_id);
        return Err(format!(
            "{}: {error}",
            lang.text(
                "无法启动处理线程",
                "Cannot start the processing worker",
                "処理スレッドを開始できません"
            )
        ));
    }
    Ok(())
}

fn stage_key(stage: FileStage) -> &'static str {
    match stage {
        FileStage::Decode => "task.phase.decode",
        FileStage::LoadModel => "task.phase.loadModel",
        FileStage::Analysis => "task.phase.analysis",
        FileStage::Inference => "task.phase.inference",
        FileStage::Reconstruction => "task.phase.reconstruction",
        FileStage::Encode => "task.phase.encode",
        FileStage::Complete => "task.phase.complete",
    }
}

fn describe_progress(spec: ModelSpec, p: FileProgress, lang: Locale) -> (String, Option<f64>) {
    match p.stage {
        FileStage::Decode => (
            lang.text("读取音频", "Reading audio", "音声を読み込み中")
                .into(),
            None,
        ),
        FileStage::LoadModel => (
            lang.text(
                "核验并载入模型",
                "Verifying and loading model",
                "モデルを検証・読み込み中",
            )
            .into(),
            None,
        ),
        FileStage::Analysis => (
            lang.text("分析音频", "Analyzing audio", "音声を解析中")
                .into(),
            None,
        ),
        FileStage::Inference => {
            let part = if matches!(spec, ModelSpec::Roformer1296) && p.total > 0 {
                p.completed as f64 / p.total as f64
            } else {
                0.0
            };
            let fraction = (p.windows_total > 0).then(|| {
                ((p.windows_completed as f64 + part) / p.windows_total as f64).clamp(0.0, 1.0)
            });
            (
                format!(
                    "{} · {}/{}",
                    lang.text("分离音频", "Separating audio", "音声を分離中"),
                    p.windows_completed,
                    p.windows_total
                ),
                fraction,
            )
        }
        FileStage::Reconstruction => (
            lang.text(
                "重建音轨",
                "Reconstructing tracks",
                "音声トラックを再構成中",
            )
            .into(),
            None,
        ),
        FileStage::Encode => (
            lang.text("保存 WAV", "Saving WAV", "WAVを保存中").into(),
            None,
        ),
        FileStage::Complete => (lang.text("完成", "Complete", "完了").into(), Some(1.0)),
    }
}
