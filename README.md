# UVR Rust

面向指定 UVR 模型的本地音频分离项目，目标是提取伴奏、分离主唱与和声、去除回声与混响。

当前是实现前的仓库基线：CLI 支持帮助和版本，GUI 只有启动页面，尚不能处理音频。推理优先纯 Rust，GUI 使用 Tauri + TypeScript；具体推理方案将在初始化后讨论。

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
pnpm gui:dev
```

开发环境需要支持 Rust 2024 edition 的 Rust 工具链、Node.js 24 与 pnpm 12.1.0。桌面开发另需 [Tauri 系统依赖](https://v2.tauri.app/start/prerequisites/)。初始化实际验证环境见 [后验知识](docs/posterior-knowledge.md)。

```sh
pnpm install --frozen-lockfile
cargo check --locked
pnpm build
cargo check --workspace --all-targets --locked
```

`pnpm build` 只构建网页；`pnpm gui:build` 构建桌面程序。当前关闭安装包打包，发布标识、图标和安装包留待发布阶段配置。

阅读上游时按需执行 `git submodule update --init`；普通构建不要求下载参考仓库。来源、版本及恢复方法见 [参考资料](docs/references.md)。

## 实现前文档

- [任务](docs/tasks.md)：范围、阶段与验收条件。
- [先验知识](docs/prior-knowledge.md)：客户反馈、假设与尚未决定的问题。
- [推理先验方案](docs/inference-priors.md)：算法清单、网络架构、技术栈候选与验证顺序。
- [框架与专用实现先验](docs/backend-priors.md)：Burn、Candle、ONNX 路线与局部手写算子的比较。
- [后验知识](docs/posterior-knowledge.md)：源码证据与实际验证结果。
- [性能](docs/performance.md)：基准设计、指标与记录规范。
- [开发约定](CONTRIBUTING.md)：检查命令与知识更新方式。
