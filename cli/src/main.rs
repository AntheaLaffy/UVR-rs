use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || (args.len() == 1 && (args[0] == "--help" || args[0] == "-h")) {
        println!("UVR — 音频分离\n\n用法：uvr [--help | --version]\n\n推理功能尚未实现。");
        ExitCode::SUCCESS
    } else if args.len() == 1 && (args[0] == "--version" || args[0] == "-V") {
        println!("uvr {}", env!("CARGO_PKG_VERSION"));
        ExitCode::SUCCESS
    } else {
        eprintln!("暂不支持此命令。使用 uvr --help 查看当前功能。");
        ExitCode::from(2)
    }
}
