use std::{path::Path, process::ExitCode};

use uvr_core::{runtime::RuntimeOptions, task::TaskCancelled};

#[macro_use]
mod locale;
mod args;
mod audio;

use locale::Locale;

fn finish(result: anyhow::Result<()>, locale: Locale) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.is::<TaskCancelled>() => {
            eprintln!(
                "{}",
                tr!(
                    locale,
                    "处理已取消。",
                    "Processing cancelled.",
                    "処理をキャンセルしました。"
                )
            );
            ExitCode::from(130)
        }
        Err(error) => {
            eprintln!(
                "{}",
                tr!(
                    locale,
                    "处理失败：{error:#}",
                    "Processing failed: {error:#}",
                    "処理に失敗しました：{error:#}"
                )
            );
            ExitCode::FAILURE
        }
    }
}

fn usage_error(message: &str, locale: Locale) -> ExitCode {
    eprintln!(
        "{}",
        tr!(
            locale,
            "参数错误：{message}。使用 uvr --help 查看当前功能。",
            "Argument error: {message}. Run uvr --lang en --help for usage.",
            "引数エラー：{message}。uvr --lang ja --help で使い方を確認できます。"
        )
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let (locale, args) = match Locale::parse(&arguments) {
        Ok(parsed) => parsed,
        Err((locale, error)) => return usage_error(&error, locale),
    };
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        print!("{}", locale.help());
        println!(
            "{}",
            tr!(
                locale,
                "本机默认计算线程数：{}（最多 8 个可用逻辑线程）。",
                "Default compute threads on this machine: {} (up to 8 available logical CPUs).",
                "このマシンの既定スレッド数：{}（利用可能な論理 CPU の最大 8 個）。",
                RuntimeOptions::default().threads
            )
        );
        ExitCode::SUCCESS
    } else if args.len() == 1 && (args[0] == "--version" || args[0] == "-V") {
        println!("uvr {}", env!("CARGO_PKG_VERSION"));
        ExitCode::SUCCESS
    } else if args.len() == 2 && args[0] == "inspect-weights" {
        match uvr_core::weights::fingerprint(Path::new(&args[1])) {
            Ok(result) => {
                println!(
                    "size_bytes: {}\nsha256: {}\nuvr_md5: {}",
                    result.size_bytes, result.sha256, result.uvr_md5
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!(
                    "{}",
                    tr!(
                        locale,
                        "无法核验权重文件：{error}",
                        "Cannot inspect weights: {error}",
                        "重みファイルを確認できません：{error}"
                    )
                );
                ExitCode::FAILURE
            }
        }
    } else if args.len() == 2 && args[0] == "inspect-audio" {
        finish(audio::inspect(Path::new(&args[1]), locale), locale)
    } else if args[0] == "separate-1296" || args[0] == "separate-vr" {
        let command = match args::separation(args, locale) {
            Ok(command) => command,
            Err(error) => return usage_error(&error, locale),
        };
        finish(
            audio::separate(
                command.spec,
                &command.weights,
                &command.input,
                &command.directory,
                command.runtime,
                locale,
            ),
            locale,
        )
    } else {
        usage_error(
            &tr!(
                locale,
                "命令或参数不正确",
                "Unknown command or incorrect arguments",
                "コマンドまたは引数が正しくありません"
            ),
            locale,
        )
    }
}
