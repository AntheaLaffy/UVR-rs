# 参考来源与版本

两个既有参考仓库以 Git 子模块记录，父仓库提交中的 gitlink 是版本依据。它们不属于 UVR Rust 的 Cargo／pnpm workspace；DeepSeek Harness 内部的 workspace 由上游维护，首版不安装或运行。

| 路径 | 来源 | 初始化固定提交 |
| --- | --- | --- |
| `upstream/` | [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) | `5517e0cf0d1acd16a1618eeedec596957523f9e1` |
| `agent/deepseek-harness/` | [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | `5dda764ed3aa172535a7967b06ff95d9cbfe536a` |

新 clone 在需要参考源码时执行：

```sh
git submodule update --init
git submodule status
```

初始化使用已有本地 clone，因此其 Git 管理目录可能仍位于子目录内；新 clone 由 Git 按标准子模块方式管理。无需为布局差异改写上游历史。

## 已查阅但尚未固定的远程资料

- [UVR 模型清单](https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json)。
- [1296 配置](https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml)。
- [python-audio-separator 的 BS-RoFormer](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py)。
- [Tauri 项目初始化](https://v2.tauri.app/start/create-project/)。GUI 使用最小 TypeScript／Vite 页面，不绑定额外 UI 框架。

远程模型资料的具体版本与权重校验和必须在开始参考实验前补齐。第三方源码及权重的许可分别依其来源；本次未给第三方内容重新授予许可，项目自身发布许可也尚未指定。
