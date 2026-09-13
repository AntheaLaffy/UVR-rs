use std::{
    ops::ControlFlow,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::locale::Locale;
use anyhow::{Context, Result};
use uvr_core::{
    audio_io,
    file_task::{self, FileStage, ModelSpec},
    runtime::{LinearLayout, RuntimeBackend, RuntimeOptions},
};

static CANCELLED: AtomicBool = AtomicBool::new(false);
fn keep_going() -> bool {
    !CANCELLED.load(Ordering::Relaxed)
}

fn install_cancellation(locale: Locale) -> Result<()> {
    ctrlc::set_handler(|| CANCELLED.store(true, Ordering::Relaxed)).with_context(|| {
        tr!(
            locale,
            "无法安装 Ctrl-C 处理器",
            "Cannot install Ctrl-C handler",
            "Ctrl-C ハンドラーを設定できません"
        )
    })
}

pub fn inspect(path: &Path, locale: Locale) -> Result<()> {
    install_cancellation(locale)?;
    let audio = audio_io::decode(path, keep_going)?;
    println!(
        "sample_rate: {}\nchannels: {}\nsamples_per_channel: {}\nduration_seconds: {:.6}",
        audio.sample_rate,
        audio.channels.len(),
        audio.channels[0].len(),
        audio.channels[0].len() as f64 / f64::from(audio.sample_rate)
    );
    Ok(())
}

pub fn separate(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    runtime: RuntimeOptions,
    locale: Locale,
) -> Result<()> {
    let runtime = runtime.effective_for(spec)?;
    install_cancellation(locale)?;
    let openvino = runtime.backend == RuntimeBackend::OpenvinoCpu;
    let backend = match runtime.backend {
        RuntimeBackend::Burn => "Burn CPU",
        RuntimeBackend::OpenvinoCpu => "OpenVINO CPU",
    };
    eprintln!(
        "{}",
        tr!(
            locale,
            "运行时：{backend}，{} 线程",
            "Runtime: {backend}, {} threads",
            "ランタイム：{backend}、{} スレッド",
            runtime.threads
        )
    );
    match spec {
        ModelSpec::Vr { .. } => eprintln!(
            "{}",
            tr!(
                locale,
                "生效参数：窗口 {} 帧，推理 batch {}，窗口并发 {}",
                "Effective settings: {} frames/window, inference batch {}, concurrent windows {}",
                "実際の設定：窓 {} フレーム、推論バッチ {}、並列窓数 {}",
                runtime.vr.window_frames,
                runtime.vr.inference_batch,
                runtime.vr.window_parallelism,
            )
        ),
        ModelSpec::Roformer1296 if !openvino => eprintln!(
            "{}",
            tr!(
                locale,
                "生效参数：时间 batch {}，频率 batch {}，窗口并发 {}，线性布局 {}",
                "Effective settings: time batch {}, frequency batch {}, concurrent windows {}, linear layout {}",
                "実際の設定：時間バッチ {}、周波数バッチ {}、並列窓数 {}、線形層レイアウト {}",
                runtime.roformer.time_batch,
                runtime.roformer.frequency_batch,
                runtime.roformer.window_parallelism,
                match runtime.roformer.linear_layout {
                    LinearLayout::Flattened => "flattened",
                    LinearLayout::Batched => "batched",
                },
            )
        ),
        ModelSpec::Roformer1296 => (),
    }
    let mut last = None;
    let output = file_task::separate_file_with_options(
        spec,
        weights,
        input,
        directory,
        runtime,
        |progress| {
            if !keep_going() {
                return ControlFlow::Break(());
            }
            let changed_stage = last
                .is_none_or(|previous: file_task::FileProgress| previous.stage != progress.stage);
            if last != Some(progress) {
                match progress.stage {
                    FileStage::Decode if changed_stage => eprintln!(
                        "{}",
                        tr!(
                            locale,
                            "解码：{}",
                            "Decoding: {}",
                            "デコード：{}",
                            input.display()
                        )
                    ),
                    FileStage::LoadModel if changed_stage => {
                        eprintln!(
                            "{}",
                            tr!(
                                locale,
                                "载入模型：{}",
                                "Loading model: {}",
                                "モデルの読み込み：{}",
                                spec.key()
                            )
                        )
                    }
                    FileStage::Analysis if changed_stage => eprintln!(
                        "{}",
                        tr!(locale, "分析音频…", "Analyzing audio…", "音声を解析中…")
                    ),
                    FileStage::Inference
                        if !matches!(spec, ModelSpec::Roformer1296)
                            || openvino
                            || progress.completed % 32 == 0 =>
                    {
                        if matches!(spec, ModelSpec::Roformer1296) && progress.total > 0 {
                            eprintln!(
                                "{}",
                                tr!(
                                    locale,
                                    "推理：{}/{} 窗口，窗内 {}/{}",
                                    "Inference: {}/{} windows, within window {}/{}",
                                    "推論：{}/{} ウィンドウ、窓内 {}/{}",
                                    progress.windows_completed,
                                    progress.windows_total,
                                    progress.completed,
                                    progress.total
                                )
                            );
                        } else {
                            eprintln!(
                                "{}",
                                tr!(
                                    locale,
                                    "推理：{}/{} 窗口",
                                    "Inference: {}/{} windows",
                                    "推論：{}/{} ウィンドウ",
                                    progress.windows_completed,
                                    progress.windows_total
                                )
                            );
                        }
                    }
                    FileStage::Reconstruction if changed_stage => eprintln!(
                        "{}",
                        tr!(
                            locale,
                            "重建音轨…",
                            "Reconstructing tracks…",
                            "トラックを再構成中…"
                        )
                    ),
                    FileStage::Encode if changed_stage => eprintln!(
                        "{}",
                        tr!(
                            locale,
                            "写入 WAV…",
                            "Writing WAV files…",
                            "WAV を書き込み中…"
                        )
                    ),
                    _ => (),
                }
            }
            last = Some(progress);
            ControlFlow::Continue(())
        },
    )?;
    for (kind, path) in output.kinds.iter().zip(&output.paths) {
        println!("{kind}: {}", path.display());
    }
    println!(
        "sample_rate: {}\nsamples_per_channel: {}",
        output.sample_rate, output.samples_per_channel
    );
    eprintln!(
        "{}",
        tr!(
            locale,
            "完成：{:.3}s（解码 {:.3}，模型 {:.3}，预处理 {:.3}，网络 {:.3}，重建 {:.3}，编码 {:.3}）",
            "Completed: {:.3}s (decode {:.3}, model {:.3}, preprocessing {:.3}, network {:.3}, reconstruction {:.3}, encode {:.3})",
            "完了：{:.3}s（デコード {:.3}、モデル {:.3}、前処理 {:.3}、推論 {:.3}、再構成 {:.3}、エンコード {:.3}）",
            output.timings.total_seconds,
            output.timings.decode_seconds,
            output.timings.load_seconds,
            output.timings.analysis_seconds,
            output.timings.network_seconds,
            output.timings.reconstruction_seconds,
            output.timings.encode_seconds
        )
    );
    Ok(())
}
