# GUI

使用 Tauri 与 TypeScript；前端暂用最小 Vite 页面，Rust 宿主位于 `src-tauri/`。GUI 是纯 Rust 偏好的例外，音频推理归 `../core/`。

未来可能接入 TypeScript 的 DeepSeek Harness、MCP 和知识技能，辅助问答与操作。首版不实现 AI 功能，也不将其依赖加入本项目 workspace。
