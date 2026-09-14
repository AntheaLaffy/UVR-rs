<a href="https://github.com/IronHpc"><img src="https://avatars.githubusercontent.com/u/328778207?v=4&amp;s=128" alt="IronHPC" width="64" height="64"></a>

# UVR Rust

[![CI](https://github.com/IronHpc/UVR-rs/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/IronHpc/UVR-rs/actions/workflows/build.yml)
[![crates.io](https://img.shields.io/crates/v/uvr-core?logo=rust&label=crates.io)](https://crates.io/crates/uvr-core)
[![Downloads](https://img.shields.io/crates/d/uvr-core?logo=rust&label=downloads)](https://crates.io/crates/uvr-core)
[![docs.rs](https://docs.rs/uvr-core/badge.svg)](https://docs.rs/uvr-core)
[![License](https://img.shields.io/crates/l/uvr-core?label=license)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.88%2B-orange?logo=rust)](CONTRIBUTING.md)

[English](README.md) · 简体中文 · [日本語](README.ja.md)

**VR CPU 推理约为 Python 参考的 3 倍速度，原生 CLI 仅 6.94 MB。**

面向希望 CPU 处理更快、软件更小的 UVR 用户。UVR Rust 直接加载原始 UVR 权重，提供桌面应用与可脚本化的 CLI，处理音频无需携带 Python 或 PyTorch 运行时。

| 你关心的事 | UVR Rust |
| --- | --- |
| VR CPU 推理 | RTF 约 2，对比 Python 约 6：约 3 倍速度 |
| 当前原生程序体积 | CLI 6.94 MB；桌面应用 18.91 MB，不含权重与可选库 |
| 可选的 1296 加速库 | OpenVINO 另加 86.80 MB；CLI + 库共 93.74 MB，GUI + 库共 105.71 MB |
| 原始权重 | 直接加载 `.pth`／`.ckpt`，无需转换 |
| 日常处理 | 本地运行、GUI 与 CLI、进度、取消及已有输出保护 |

速度来自已记录的本机 5-HP 配置对比，两侧素材与线程数不同，不是同资源基准。RTF 2 表示处理 10 秒音频约需 20 秒。体积对应 9 月 13 日移除符号的 Linux native release，所有合计均不含模型权重与系统库。GUI 已包含推理，不需要再附加 CLI。准确字节数、摘要、历史构建与测量条件见[性能与体积证据](docs/performance-summary.zh-CN.md)。

## 选择模型

| 你想要… | 模型／CLI 标识 | 原始权重文件 | 输出音轨 |
| --- | --- | --- | --- |
| 分离人声与伴奏 | BS-RoFormer 1296／`1296` | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | `vocals`、`instrumental` |
| 尝试 Karaoke 分离 | 5-HP／`5hp` | `5_HP-Karaoke-UVR.pth` | `primary`、`residual` |
| 比较另一套 Karaoke 模型 | 6-HP／`6hp` | `6_HP-Karaoke-UVR.pth` | `primary`、`residual` |
| 减少回声与混响 | DeEcho／`deecho` | `UVR-DeEcho-DeReverb.pth` | `primary`、`residual` |

Karaoke 与 DeEcho 的 `primary` 是模型主输出，`residual` 由互补掩码重建。请用自己的素材试听两轨；这些标签不承诺完美的主唱／和声或干声／混响分离。权重单独下载，不捆绑在应用中；来源及准确文件名见[模型清单](references/targets.json)。

CLI 与桌面已支持单模型处理；自动多模型处理链、更多平台及整曲质量／性能仍在验收中。界面与用户文档支持简体中文、英语和日语。本项目支持上述四套权重，不代表兼容完整 UVR 模型目录。

## 开始使用

### 桌面应用

启动 `uvr-gui`，选择输入音频、模型，以及模型和输出目录。**模型管理**会检查已有权重，也能从界面提供的来源下载缺少的模型。权重就绪后，分离任务可离线执行。

**推理运行时**提供当前可用的后端和计算线程数；**高级推理参数**提供对应模型的窗口、批量及布局选项。设置会在本机保存，**恢复默认**会恢复当前模型的推荐参数。任务记录包含实际生效设置，便于在 CLI 中复现配置；两个入口共用 Rust 推理实现与参数规则。

可以切换简体中文、英语、日语，选择跟随系统／浅色／深色模式及主题色；语言与外观偏好保存在本机。

### 命令行

按下文完成通用构建后，在仓库根目录运行：

```sh
./target/release/uvr separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296

./target/release/uvr separate-vr deecho \
  models/UVR-DeEcho-DeReverb.pth input.wav outputs/deecho

./target/release/uvr --help
```

如果使用本机构建，将程序路径换成 `target/native/release/uvr`。两类模型都支持 `--threads`；VR 提供 `--window-frames`、`--inference-batch` 和 `--parallel-windows`。1296 可选择 `--backend`，其 Burn 路径还提供注意力批量、窗口并发及线性层布局。完整选项、边界和示例见[运行时指南](docs/runtime.zh-CN.md)。

在命令前添加 `--lang zh-CN`、`--lang en` 或 `--lang ja`，也可设置 `UVR_LANG`；显式选择优先，默认中文。stdout 机器可读字段名不随语言变化，底层技术错误保留原文。

输入支持单／双声道 WAV、FLAC、MP3。输出为两条 44.1 kHz 双声道 32 位浮点 WAV，命名为 `<输入名>_<模型>_<音轨>.wav`，不归一化、不削波。单声道会复制为双声道，其他采样率先重采样。已有输出会保留。可按 Ctrl-C 或桌面取消按钮停止任务，响应时间取决于当前计算片段。CLI 退出码：`0` 成功，`1` 处理失败，`2` 参数错误，`130` 取消。

`inspect-audio <文件>` 报告解码后的音频信息；`inspect-weights <文件>` 报告字节数、SHA-256 与 UVR 元数据 MD5。仅计算文件摘要，不代表权重已通过模型支持检查。

## 构建与安装

在 Linux x86_64 上构建本机优化版本前，请先安装 `make`，并阅读[运行时指南](docs/runtime.zh-CN.md)准备依赖，然后运行：

```sh
pnpm install --frozen-lockfile
make
```

优化产物位于 `target/native/release/`；启用 OpenVINO 时移动程序请保留旁边的 `lib/`。只使用 Burn 时运行 `make native-burn`。通用构建、Windows、CI 产物和依赖细节见[运行时指南](docs/runtime.zh-CN.md)。开发构建也可从 [GitHub Actions](https://github.com/AntheaLaffy/UVR-rs/actions/workflows/build.yml)获取。

如果系统没有 `make`，请直接查看[运行时指南](docs/runtime.zh-CN.md)中的底层构建命令。

## 复用 Rust crates

信号处理使用 [`uvr-dsp`](https://crates.io/crates/uvr-dsp)，模型识别／下载使用 [`uvr-models`](https://crates.io/crates/uvr-models)，只需要一种推理网络时选择 [`uvr-vr`](https://crates.io/crates/uvr-vr) 或 [`uvr-roformer`](https://crates.io/crates/uvr-roformer)，完整文件任务使用 [`uvr-runtime`](https://crates.io/crates/uvr-runtime)。这些聚焦 API 共用 [`uvr-core`](https://crates.io/crates/uvr-core) 实现。

## 参与与深入了解

[贡献指南](CONTRIBUTING.zh-CN.md)说明架构、检查命令、问题报告和证据要求；[运行时指南](docs/runtime.zh-CN.md)解释参数与部署；[基准记录](benchmarks/README.md)保存实测结果和未完成范围。

研究资料目前主要使用中文：[任务与验收](docs/tasks.md)、[工程基线](docs/baseline.md)、[性能协议](docs/performance.md)、[后验知识](docs/posterior-knowledge.md)和[上游参考](docs/references.md)。本项目基于所引用的模型与算法工作。

## 许可证

UVR Rust 的原创代码与文档采用 [MIT 许可证](LICENSE)。模型权重与其他第三方材料仍遵循各自条款，详见[第三方说明](THIRD_PARTY_NOTICES.md)。
