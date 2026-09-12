use std::{path::Path, process::ExitCode};

use uvr_core::{task::TaskCancelled, vr::VrOptions, vr_dsp::VrVariant};

mod audio;

fn finish(result: anyhow::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.is::<TaskCancelled>() => {
            eprintln!("处理已取消。");
            ExitCode::from(130)
        }
        Err(error) => {
            eprintln!("处理失败：{error:#}");
            ExitCode::FAILURE
        }
    }
}

fn usage_error() -> ExitCode {
    eprintln!("命令或参数不正确。使用 uvr --help 查看当前功能。");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        println!(concat!(
            "UVR — 音频分离\n\n用法：uvr [--help | --version]\n",
            "      uvr inspect-weights <文件>\n      uvr inspect-audio <音频>\n",
            "      uvr separate-vr <5hp|6hp|deecho> <权重> <音频> <输出目录> [--window-frames <帧数>]\n\n",
            "      uvr separate-1296 <权重> <音频> <输出目录> [--backend <burn|openvino-cpu>]\n\n",
            "inspect-weights 输出文件大小、整文件 SHA-256 和 UVR 元数据 MD5 标识。\n",
            "inspect-audio 解码 WAV／FLAC／MP3 并报告声道、采样率和精确长度。\n",
            "separate-vr 为 CPU 验证版，保存模型主输出与互补输出两条 44.1 kHz 双声道浮点 WAV。\n",
            "默认 512 帧、2 线程，TTA／额外掩码后处理关闭；帧数可为 16 的倍数且不超过 2048。\n",
            "RAYON_NUM_THREADS 可设线程数；Ctrl-C 在阶段和窗口之间取消；已有输出不会覆盖。\n",
            "separate-1296 使用工程基线 v1.1：8 秒窗、重叠数 4、64 位正向频谱、FP32 网络。\n",
            "1296 输出人声／伴奏两轨，支持窗内取消；当前为 CPU 验证版，整曲验收与处理链仍在开发。\n",
            "OpenVINO CPU 是可选实验后端，需要相应构建和原生运行时。"
        ));
        ExitCode::SUCCESS
    } else if args.len() == 1 && (args[0] == "--version" || args[0] == "-V") {
        println!("uvr {}", env!("CARGO_PKG_VERSION"));
        ExitCode::SUCCESS
    } else if args.len() == 2 && args[0] == "inspect-weights" {
        match uvr_core::weights::fingerprint(std::path::Path::new(&args[1])) {
            Ok(result) => {
                println!(
                    "size_bytes: {}\nsha256: {}\nuvr_md5: {}",
                    result.size_bytes, result.sha256, result.uvr_md5
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("无法核验权重文件：{error}");
                ExitCode::FAILURE
            }
        }
    } else if args.len() == 2 && args[0] == "inspect-audio" {
        finish(audio::inspect(Path::new(&args[1])))
    } else if (args.len() == 4 || args.len() == 6) && args[0] == "separate-1296" {
        let openvino = if args.len() == 6 {
            if args[4] != "--backend" {
                return usage_error();
            }
            match args[5].to_str() {
                Some("burn") => false,
                Some("openvino-cpu") => true,
                _ => return usage_error(),
            }
        } else {
            false
        };
        finish(audio::separate_roformer(
            Path::new(&args[1]),
            Path::new(&args[2]),
            Path::new(&args[3]),
            openvino,
        ))
    } else if (args.len() == 5 || args.len() == 7) && args[0] == "separate-vr" {
        let (name, variant) = match args[1].to_str() {
            Some("5hp") => ("5hp", VrVariant::HpFive),
            Some("6hp") => ("6hp", VrVariant::HpSix),
            Some("deecho") => ("deecho", VrVariant::DeEcho),
            _ => return usage_error(),
        };
        let mut options = VrOptions::default();
        if args.len() == 7 {
            if args[5] != "--window-frames" {
                return usage_error();
            }
            let Some(frames) = args[6].to_str().and_then(|v| v.parse().ok()) else {
                return usage_error();
            };
            options.window_frames = frames;
        }
        if options.window_frames <= 2 * variant.offset()
            || options.window_frames > 2048
            || !options.window_frames.is_multiple_of(16)
        {
            return usage_error();
        }
        finish(audio::separate(
            name,
            variant,
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            options,
        ))
    } else {
        usage_error()
    }
}
