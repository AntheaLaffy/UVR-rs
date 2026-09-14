# 参与 UVR Rust

[English](CONTRIBUTING.md) · 简体中文 · [日本語](CONTRIBUTING.ja.md)

帮助我们让本地音频分离更好用，也让结果更容易验证。先从[使用指南](README.zh-CN.md)和[运行时说明](docs/runtime.zh-CN.md)了解产品。推理改动必须保留约定的音频语义与质量门限；更快的内核只有在不破坏这些保证、并改善实际任务时才有价值。

## 找到合适的改动位置

| 位置 | 职责 |
| --- | --- |
| `core/` | 音频解码／DSP、原始权重加载、推理与文件任务 |
| `crates/` | 面向 crates.io 的聚焦入口；保持轻量，由 `uvr-core` 作为唯一实现来源 |
| `cli/` | 参数解析、环境变量兼容、终端进度与退出码 |
| `gui/src/` | 桌面交互、校验、偏好设置与进度展示 |
| `gui/src-tauri/` | 原生桌面命令、模型管理与任务协调 |
| `tools/reference/` | 独立参考生成与验证，不参与产品运行时 |
| `benchmarks/` | 可复现的正确性与性能实验 |
| `upstream/`、`agent/deepseek-harness/` | 产品 workspace 外的固定版本参考 |

Cargo workspace 默认构建 core 与 CLI，让命令行开发不依赖桌面系统库；pnpm workspace 管理 GUI。普通构建不需要参考子模块，需要阅读上游源码时再按[参考资料](docs/references.md)初始化。

音频与推理行为放在 core。CLI 和桌面共用 `RuntimeOptions` 与 `separate_file_with_options`，包括参数校验、实际调度设置和每任务线程池。仅展示当前模型／后端已经实现的选项。不要修改进程级环境变量来传递设置：CLI 将兼容的环境变量解析为显式任务参数，桌面发送用户选中的设置。

core 的 `burn-cpu` 特性提供推理与 PCM 任务，`audio-io` 增加编解码和文件任务，CLI 默认启用两者。`openvino` 为可选特性，需要原生 CPU 运行时可用。Python／PyTorch 仅用于独立验证，普通构建、应用启动和分离任务不能调用它；固定样本保证普通 Cargo 测试无需该环境。

## 检查改动

使用支持 Rust 2024 的工具链。桌面开发还需 Node.js 24、pnpm 12.1.0 和 [Tauri 系统依赖](https://v2.tauri.app/start/prerequisites/)，然后运行 `pnpm install --frozen-lockfile`。

迭代时运行相关检查，交接前完成适用的 workspace 检查：

```sh
cargo fmt --all -- --check
cargo test --locked -p uvr-core -p uvr-cli
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/burn-cpu
cargo clippy --locked -p uvr-core --all-targets --features burn-cpu -- -D warnings
pnpm build
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --locked -p uvr-cli -- --help
cargo run --locked -p uvr-cli -- --version
git diff --check
```

后端或运行时改动还需要检查可选构建：

```sh
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/openvino,uvr-cli/openvino
cargo check --workspace --all-targets --locked --all-features
```

核心开发可先运行 `cargo check --locked`。普通测试覆盖文件摘要、DSP／编解码、取消、输出保护、运行时校验及 CLI 失败场景。缺少本地大型权重时通过这些检查，**不能**说明完整模型已经验收。模型与波形对照使用[后端探针](benchmarks/backend-probe/README.md)和[独立验证工具](tools/reference/README.md)，交接时写清实际检查与未验证范围。

GUI 改动同时验证原生应用和网页预览；受影响时检查模型／后端切换、非法参数、设置保存、取消及输出处理。网页构建成功不能证明原生命令与推理链行为正确。

Windows 工作可由 Linux 开发配合 Windows 协作者本机编译、运行验证，Windows 验收仍未完成。比较性能时先核对 release 优化、CPU 指令目标、工具链、后端设置与线程调度，不把速度差异直接归因于交叉编译。`tools/build-native.mjs` 仅支持 Linux；不要把构建机的 `target-cpu=native` 带入面向其他电脑的 Windows 发行版。

## 报告问题与提出改动

问题报告从用户可见的现象出发，给出尽可能短的复现：应用版本与构建命令、系统／CPU、模型、实际运行参数、输入格式／长度、预期行为及相关错误或任务记录。性能问题还需输入与权重摘要、计时边界和重复测量。不要附带私人音频或大型权重；能复现相同问题时，优先使用可分享的合成样本。

让每个改动围绕一个可独立评审的问题。PR 说明交代动机、最终行为、重要取舍和验证限制。共享参数变化时，同步更新 CLI 帮助、GUI 控件及运行时文档。界面文字和用户文档的中、英、日版本保持一致，命令名、路径和选项值无需翻译。修改可见文字或样式时，检查语言切换与浅色／深色模式。

依赖变化时同步提交 `Cargo.lock` 和 `pnpm-lock.yaml`。权重、客户音频、生成分轨和大型基准产物放在被忽略的目录；可共享清单与摘要需记录输入校验和。参考子模块更新须写明新提交号与更新原因。

## 保留证据与判断边界

研究资料目前主要为中文。修改推理假设前阅读[任务](docs/tasks.md)、[工程基线](docs/baseline.md)和[先验知识](docs/prior-knowledge.md)。向[后验知识](docs/posterior-knowledge.md)补充记录时，写明日期、代码版本、证据位置、结果与适用范围。

- 客户要求直接标注为要求，不伪装成实验结论。
- 未验证判断保留在先验文档，并给出验证方法。
- 遵循[性能协议](docs/performance.md)，保证模型、输入、质量门限与计时边界可比较；局部算子提速或构建耗时不能代表端到端推理收益。
- 保留失败和被修正的结论，并解释失效原因；不放宽质量门限来掩盖回归。

详细实验历史放在[基准记录](benchmarks/README.md)。README 帮助新用户判断是否适合自己、如何运行，避免承载整份实验存档。

## 共用品牌资源

桌面图标、界面 Logo 与 favicon 共用原始的 `gui/src-tauri/icons/source.svg`：深蓝绿背景上的浅绿声波。保持这一份源图，避免桌面与网页标志不一致。桌面导出资源位于 `gui/src-tauri/icons/uvr/`，网页资源为 `gui/public/branding/uvr.png` 与 `uvr-32.png`。

更新源图后，用 Tauri 重新导出，只同步本项目使用的平台资源：

```sh
pnpm --filter @uvr/gui tauri icon src-tauri/icons/source.svg --output /tmp/uvr-icons
cp /tmp/uvr-icons/{32x32.png,64x64.png,128x128.png,128x128@2x.png,icon.png,icon.ico,icon.icns} gui/src-tauri/icons/uvr/
cp /tmp/uvr-icons/128x128.png gui/public/branding/uvr.png
cp /tmp/uvr-icons/32x32.png gui/public/branding/uvr-32.png
```
