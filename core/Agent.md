# Core

共享的 Rust 推理库，供 `cli/` 与 `gui/src-tauri/` 使用。原计划放在此处的 CLI 已拆到 `cli/`，避免 GUI 依赖命令行进程。

优先纯 Rust；推理后端、算子与权重格式待讨论。任务与证据分别记录在 `../docs/tasks.md`、`../docs/prior-knowledge.md` 和 `../docs/posterior-knowledge.md`。
