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

## 模型实验来源

`references/targets.json` 固定了四套目标权重的 release asset ID、字节数和下载 URL，以及 `TRvlvr/application_data` 提交 `3826b05b570dbd4fbedbc807758803b35348ba1b` 下模型清单、VR 元数据和 1296 配置的 SHA-256。元数据本地文件放在 Git 忽略目录，按清单 URL 可重新获取；部分下载使用 `.part` 后缀。此清单不代表权重已验证或获准随产品分发。

## 其他远程资料

- [UVR 模型清单](https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json)。
- [1296 配置](https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml)。
- [python-audio-separator 的 BS-RoFormer](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py)。
- [Tauri 项目初始化](https://v2.tauri.app/start/create-project/)。GUI 使用最小 TypeScript／Vite 页面，不绑定额外 UI 框架。

1296 的网络参考固定为 `nomadkaraoke/python-audio-separator` 提交 `bf1164aa0f1ee1d1d0ef0f09b315f7659fc06bab`，源码文件与摘要见 [源码清单](../references/roformer-source.json)。四套权重的实际整文件校验和与已验证范围见 [权重记录](../references/verified-weights.json)。第三方源码及权重的许可分别依其来源；本次未给第三方内容重新授予许可，项目自身发布许可也尚未指定。
