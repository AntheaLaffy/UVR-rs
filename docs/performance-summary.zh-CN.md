# 更快分离，更小的运行时

[English](performance-summary.md) | 简体中文 | [日本語](performance-summary.ja.md)

UVR Rust 的重点是 CPU 推理速度与原生程序体积。当前本机 VR 观察结果为 **RTF 约 2，对比 Python 参考约 6，处理速度约 3 倍**。当前 Linux native CLI 为 **6.94 MB**、桌面程序为 **18.91 MB**，无需捆绑 Python 或 PyTorch。可选的 OpenVINO 加速库另加 **86.80 MB**；GUI 与这些库合计 **105.71 MB**，不含模型权重和系统库。

## 速度数字如何理解

RTF = 处理耗时 ÷ 音频时长，越小越快。RTF 2 表示处理 10 秒音频约需 20 秒，不是实时播放速度；实时处理要求 RTF ≤1。

| 归档观察 | Rust native | Python 参考 |
| --- | --- | --- |
| 模型 | 5-HP | 5-HP |
| 机器 | Intel i5-13420H，Linux | 同一机器 |
| 音频时长 | 10 秒 | 3 秒 |
| CPU 线程 | 8 | 2 |
| 窗口／批次／窗口并发 | 512 / 1 / 4 | 512 / 1 / 1 |
| 五次热运行 RTF 中位数 | 1.9917 | 6.0809 |

这组数据展示本项目现有配置、CPU 调度与优化内核带来的本机使用体验。两份记录的音频时长和线程预算不同，因此「约 3 倍」是这些观察值的比较，不是同资源控制实验，也不代表所有 UVR 模型。计时包括 PCM 分析、推理、重建，不包括载入权重及文件解码、编码。Rust 五次热运行的双轨输出均通过已有波形误差门限。

[纳入版本控制的数据摘要](../benchmarks/2026-09-13-runtime-summary.json) 保存了每次样本、配置与原始记录摘要。[VR 实验记录](../benchmarks/2026-09-11-vr-cpu-optimization.md) 另有历史同资源测试；发布结果时应区分这两种比较口径。

DeEcho 的批次／窗口并发调优和 BS-RoFormer 1296 的性能调优均未完成。DeEcho 当前强制批次 1、窗口并发 1，尚未完成并发 1／2／4 的完整音频对照；RoFormer 的 Burn／OpenVINO 测量仍是阶段性结果，整曲和多模型处理链尚待验收。上面的 5-HP 速度数字不代表这两个模型。当前限制与待办见[运行时指南](runtime.zh-CN.md#尚未完成的调优)。

## 软件体积包含什么

9 月 13 日构建于 UTC 10:06:28 在 Intel i5-13420H 上完成，采用 Linux x86_64 release 优化、`-C target-cpu=native`、移除符号，并启用 OpenVINO 特性。`target/native/release/build-info.json` 记录可执行文件字节数、SHA-256 与构建设置；已将摘要与最终文件核对，并连同每个暂存原生库的体积和摘要写入[受版本控制的数据摘要](../benchmarks/2026-09-13-runtime-summary.json)。

| 当前产物 | 字节数 | MB |
| --- | ---: | ---: |
| CLI 可执行文件 `uvr` | 6,943,112 | 6.94 |
| 桌面可执行文件 `uvr-gui` | 18,905,752 | 18.91 |
| 可选 OpenVINO CPU 库 `lib/` | 86,800,978 | 86.80 |
| CLI + OpenVINO 库 | 93,744,090 | 93.74 |
| GUI + OpenVINO 库 | 105,706,730 | 105.71 |

MB 按 1,000,000 字节计算。可选库总量为暂存目录中 10 个文件的长度之和，包括复制的别名文件，不计算压缩或文件系统块取整。所有合计均不含模型权重、系统库及元数据文件。GUI 已包含推理代码，无需再附加 CLI。Burn-only 构建输出到 `target/native-burn/release/`，不需要这些 OpenVINO 库；其可执行文件体积不能从本次启用 OpenVINO 的构建推算。

9 月 12 日历史 release 仍单独保留：`uvr` 为 9,122,024 字节（9.12 MB），`uvr-gui` 为 25,811,240 字节（25.81 MB），早于当前界面与运行时更新。体积下降对应本次构建，包括移除符号的效果，与上文归档的推理计时分别记录。

Burn 运行时不携带 Python、PyTorch。Tauri 使用平台 WebView，不额外捆绑 Chromium 浏览器。用于加速 1296 的可选 OpenVINO 会增加上表单列的原生库。所有模型仍需原始权重，Linux GUI 仍需 GTK／WebKit 系统库。

原版 UVR 的 Windows／macOS 安装包已经包含 Python 依赖；本项目优势是分发时无需携带这套运行环境，不是声称所有 UVR 用户都得自行安装 Python。原版说明见[固定上游版本](../upstream/README.md)。

## 复现与继续优化

使用 `pnpm build:native` 构建，并按[运行时指南](runtime.zh-CN.md)选择对应配置。正式比较时固定音频、权重摘要、预设、计时范围和硬件，记录线程数、原生库版本与全部热运行样本；采用任何提速方案前都要通过波形检查。

完整方法见[性能协议](performance.md)。后续结果应增加证据，不覆盖此前数字的来源和适用条件。
