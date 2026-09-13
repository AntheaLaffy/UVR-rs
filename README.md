# UVR Rust

面向指定 UVR 模型的本地音频分离项目，目标是提取伴奏、分离主唱与和声、去除回声与混响。

CLI 已接通 1296、5-HP、6-HP、DeEcho 四套模型，并通过真实片段的独立波形对照。桌面单模型界面已接入共享 core，正在验证原生交互；跨窗、多模型处理链、音质和端到端性能仍在验收中。推理由 Rust 执行，GUI 使用 Tauri + TypeScript；运行时不依赖 Python，独立验证工具可以使用 Python。

## Workspace

| 路径 | 职责 |
| --- | --- |
| `core/` | `uvr-core`，CLI 与 GUI 共用的 Rust 库 |
| `cli/` | `uvr-cli` 包，生成 `uvr` 命令 |
| `gui/` | `@uvr/gui`，TypeScript + Vite 前端 |
| `gui/src-tauri/` | `uvr-gui`，Tauri Rust 宿主 |
| `upstream/` | 固定版本的 UVR 参考子模块 |
| `agent/deepseek-harness/` | 固定版本的未来 AI 集成参考，保留独立 workspace |

根 Cargo workspace 管理三个 Rust 包，默认构建 core 与 CLI，避免命令行开发依赖桌面系统库。根 pnpm workspace 管理 GUI；参考子模块不参与本项目构建。

## 使用与开发

```sh
cargo run -p uvr-cli -- --help
cargo run --locked -p uvr-cli -- inspect-weights models/5_HP-Karaoke-UVR.pth
cargo run --release --locked -p uvr-cli -- inspect-audio input.mp3
cargo run --release --locked -p uvr-cli -- separate-1296 models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296
cargo run --release --locked -p uvr-cli -- separate-vr deecho models/UVR-DeEcho-DeReverb.pth input.wav outputs/deecho
pnpm gui:dev
```

`inspect-weights` 输出文件字节数、整文件 SHA-256 和 UVR 元数据查询用的 MD5 标识，供实验清单记录。MD5 按参考程序只覆盖末尾 10,240,000 字节（短文件覆盖全部），不能替代整文件校验。命令不反序列化权重，也不确认模型来源、张量结构或推理可用性；空文件同样可以计算摘要。检查期间请勿修改文件。读取失败返回 1，用法错误返回 2。

`inspect-audio` 解码 WAV、FLAC 或 MP3，报告原始声道、采样率及有效采样数。`separate-vr` 接受 `5hp`、`6hp`、`deecho`，保存 `<输入名>_<模型>_primary.wav` 与 `residual.wav`；前者为模型主输出，后者为互补掩码重建。两轨都是 44.1 kHz 双声道浮点 WAV，单声道复制为双声道，不归一化或削波。Karaoke 在混音与独立人声上的音轨语义、实际效果仍需分别验证。

当前预设是 FP32、窗口 512 帧、TTA／额外掩码后处理关闭。可用 `--window-frames` 调整窗口，须为 16 的倍数、大于双侧上下文总长且不超过 2048；默认使用 2 个 CPU 线程，可通过 `RAYON_NUM_THREADS` 设置。Ctrl-C 在阶段及窗口之间取消，推理窗口内部暂不支持即时中断。已有输出返回错误；两轨先完成临时编码再发布，若第二轨发布失败会明确报告已保存的第一轨。

### VR Burn CPU 性能存档

当前保存的 VR 性能候选针对本机 native 构建：`TILE_BATCH=512`、HP 窗口并发 4、`RAYON_NUM_THREADS=8`，Winograd 使用直接 GEMM、列主序权重、forward 内 scratch 复用，以及 tile 坐标和 product 偏移预计算。它们的共同目的，是减少 Burn 张量包装、重复分配和缓存搬运；这些改动保留是因为曾在完整 5-HP 音频任务中给出端到端收益，而不是只改善局部算子计时。

复现实验（10 秒、4 窗口、5-HP）可以使用：

```sh
RUSTFLAGS='-C target-cpu=native' \
CARGO_TARGET_DIR=/tmp/uvr-native \
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --bin probe-vr-audio

env PATH=/nonexistent RAYON_NUM_THREADS=8 \
/tmp/uvr-native/release/probe-vr-audio \
  --measure --runs 7 --case local_audio --parallel-windows 4 \
  models/5_HP-Karaoke-UVR.pth \
  benchmarks/artifacts/2026-09-13-vr-hpc/5hp-long10 \
  /tmp/vr-archive-measure.json
```

本次归档版本的 5 个 warm 样本为 `RTF 1.992/1.981/1.991/2.052/2.008`，中位数 `1.992`；单次结果曾越过 `RTF <= 2.0`，但完整 7-run 的历史样本也出现过 `2.023` 中位数，因此不能把它描述为所有机器、所有运行都稳定达到 1:2。每次双轨质量检查通过，最大绝对误差 `7.15e-7`、RMSE 约 `3.6e-8`。构建和实验细节见 [VR CPU 优化实验](benchmarks/2026-09-11-vr-cpu-optimization.md) 与 [性能协议](docs/performance.md)；RoFormer 1296 网络不属于这份 VR 存档。

`separate-1296` 使用 8 秒窗口、重叠数 4、float64 正向频谱与 FP32 网络，输出 `_1296_vocals.wav` 和 `_1296_instrumental.wav`；支持窗内取消。短音频按完整 hop 补齐后裁回精确长度，伴奏为重采样后的输入减人声。精度选择与验证证据见 [1296 数值记录](benchmarks/2026-09-11-roformer-numerics.md)。

CLI 和 GUI 使用 core 的 `file_task` 接口，直接读取已核验的原始 `.pth`／`.ckpt`，无需 Python 转换。`VrSeparator`／`RoformerModel` 接受 PCM，文件入口使用 Symphonia 解码、Hound 编码；core 的无特性构建仍可独立使用 DSP／权重检查。参考、预设及验收预算由 [工程基线](docs/baseline.md) 固定；VR 完整 PCM 对照见 [音频记录](benchmarks/2026-09-10-vr-cpu-audio.md)。

开发环境需要支持 Rust 2024 edition 的 Rust 工具链、Node.js 24 与 pnpm 12.1.0。桌面开发另需 [Tauri 系统依赖](https://v2.tauri.app/start/prerequisites/)。初始化实际验证环境见 [后验知识](docs/posterior-knowledge.md)。

```sh
pnpm install --frozen-lockfile
cargo check --locked
pnpm build
cargo check --workspace --all-targets --locked
```

`pnpm build` 只构建网页；`pnpm gui:build` 构建桌面程序。当前关闭安装包打包，发布标识与安装包留待发布阶段配置。GUI 已使用 ATRI 风格 Logo，源图与重新导出方式见 [开发约定](CONTRIBUTING.md)。

阅读上游时按需执行 `git submodule update --init`；普通构建不要求下载参考仓库。来源、版本及恢复方法见 [参考资料](docs/references.md)。

## 实现前文档

- [任务](docs/tasks.md)：范围、阶段与验收条件。
- [工程基线](docs/baseline.md)：已决定的参考版本、运行边界、音频参数与验收门限。
- [先验知识](docs/prior-knowledge.md)：客户反馈、假设与尚未决定的问题。
- [推理先验方案](docs/inference-priors.md)：算法清单、网络架构、技术栈候选与验证顺序。
- [框架与专用实现先验](docs/backend-priors.md)：Burn、Candle、ONNX 路线与局部手写算子的比较。
- [后验知识](docs/posterior-knowledge.md)：源码证据与实际验证结果。
- [性能](docs/performance.md)：基准设计、指标与记录规范。
- [开发约定](CONTRIBUTING.md)：检查命令与知识更新方式。
