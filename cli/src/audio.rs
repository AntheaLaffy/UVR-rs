use std::{
    ops::ControlFlow,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result};
use uvr_core::{
    audio_io,
    file_task::{self, FileStage, ModelSpec, RoformerBackend},
    vr::VrOptions,
    vr_dsp::VrVariant,
};

static CANCELLED: AtomicBool = AtomicBool::new(false);
fn keep_going() -> bool {
    !CANCELLED.load(Ordering::Relaxed)
}

fn install_cancellation() -> Result<()> {
    ctrlc::set_handler(|| CANCELLED.store(true, Ordering::Relaxed))
        .context("无法安装 Ctrl-C 处理器")
}

pub fn inspect(path: &Path) -> Result<()> {
    install_cancellation()?;
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
    _variant_name: &str,
    variant: VrVariant,
    weights: &Path,
    input: &Path,
    directory: &Path,
    options: VrOptions,
) -> Result<()> {
    run(
        ModelSpec::Vr { variant, options },
        weights,
        input,
        directory,
        false,
    )
}

pub fn separate_roformer(
    weights: &Path,
    input: &Path,
    directory: &Path,
    openvino: bool,
) -> Result<()> {
    run(ModelSpec::Roformer1296, weights, input, directory, openvino)
}

fn run(
    spec: ModelSpec,
    weights: &Path,
    input: &Path,
    directory: &Path,
    openvino: bool,
) -> Result<()> {
    install_cancellation()?;
    let threads: usize = std::env::var("RAYON_NUM_THREADS")
        .map_or(Ok(2), |s| s.parse())
        .context("RAYON_NUM_THREADS 必须为正整数")?;
    anyhow::ensure!(threads > 0, "RAYON_NUM_THREADS 必须为正整数");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let mut last = None;
    let backend = if openvino {
        RoformerBackend::OpenvinoCpu { threads }
    } else {
        RoformerBackend::Burn
    };
    let output = file_task::separate_file_with_backend(
        spec,
        weights,
        input,
        directory,
        backend,
        |progress| {
            if !keep_going() {
                return ControlFlow::Break(());
            }
            let changed_stage = last
                .is_none_or(|previous: file_task::FileProgress| previous.stage != progress.stage);
            if last != Some(progress) {
                match progress.stage {
                    FileStage::Decode if changed_stage => eprintln!("解码：{}", input.display()),
                    FileStage::LoadModel if changed_stage => {
                        let backend = if openvino { "OpenVINO CPU" } else { "Burn CPU" };
                        eprintln!("载入模型：{}（{backend}，{threads} 线程）", spec.key())
                    }
                    FileStage::Analysis if changed_stage => eprintln!("分析音频…"),
                    FileStage::Inference
                        if !matches!(spec, ModelSpec::Roformer1296)
                            || openvino
                            || progress.completed % 32 == 0 =>
                    {
                        if matches!(spec, ModelSpec::Roformer1296) && progress.total > 0 {
                            eprintln!(
                                "推理：{}/{} 窗口，窗内 {}/{}",
                                progress.windows_completed,
                                progress.windows_total,
                                progress.completed,
                                progress.total
                            );
                        } else {
                            eprintln!(
                                "推理：{}/{} 窗口",
                                progress.windows_completed, progress.windows_total
                            );
                        }
                    }
                    FileStage::Reconstruction if changed_stage => eprintln!("重建音轨…"),
                    FileStage::Encode if changed_stage => eprintln!("写入 WAV…"),
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
        "完成：{:.3}s（解码 {:.3}，模型 {:.3}，预处理 {:.3}，网络 {:.3}，重建 {:.3}，编码 {:.3}）",
        output.timings.total_seconds,
        output.timings.decode_seconds,
        output.timings.load_seconds,
        output.timings.analysis_seconds,
        output.timings.network_seconds,
        output.timings.reconstruction_seconds,
        output.timings.encode_seconds
    );
    Ok(())
}
