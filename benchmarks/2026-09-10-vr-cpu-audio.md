# VR 完整 PCM 音频对照

日期：2026-09-10。三套实际权重分别通过 6 组完整音频对照，共 18 组。Rust 从 PCM 执行 polyphase 重采样、多频带 STFT、全段幅度归一化、完整网络窗口调度、互补掩码重建及长度裁剪。所有输出的声道、采样数及有限值符合参考；静音严格为零。

固定 [工程基线 v1](../docs/baseline.md)：CPU FP32、2 线程、batch=1、窗口 512，TTA、aggressiveness 与其他可选后处理关闭。两侧使用相同原始 checkpoint、UVR 提交、librosa 0.9.2／SciPy 1.16.3 参考参数；重建未覆盖频点填零，静音不执行除零归一化。

| 模型 | 通过样本 | 全部样本／音轨中的最大绝对误差 | 最大 RMSE |
| --- | --- | --- | --- |
| 5-HP | 6 / 6 | 2.3842e-7 | 3.2457e-8 |
| 6-HP | 6 / 6 | 2.6822e-7 | 3.4306e-8 |
| DeEcho | 6 / 6 | 8.8290e-7 | 7.5093e-8 |

每轨同时满足逐样本 `2e-4 + 2e-3 × abs(reference)` 及 `RMSE ≤ 1e-3 × max(reference RMS, 1e-4)`。没有分别归一化或移动波形来消除差异。最大误差与最大 RMSE 可能来自不同样本；完整逐轨数据保存在原始报告。

样本包括 4097 个采样的双声道合成音、首尾脉冲、7 个 22.05 kHz 单声道采样（输出 14 个）、503 个 48 kHz 静音采样（输出 463 个）、恰好 ROI 帧及非整数 ROI 长度。后两项检查跨窗拼接；恰好 ROI 时保留参考额外执行一窗的行为。进度中的窗口数必须匹配参考，取消前向的错误类型也经过检查。

产物与复现：

- 原始报告：`benchmarks/artifacts/backend-probe/{5hp,6hp,deecho}-audio.json`；样本在对应的 `*-audio/` 目录。
- 三个样本清单 SHA-256 依次为 `b4dc91ce622ecebbb70546dfa055daf2cd124335d26b05fc60067409000d5b42`、`17773c1da54ad110e86da5c58e5262511cb88ec379bdbafe83d733ef415e74b2`、`15f44ae9d7430bd68209c1f5b90b48e22881fa1ffa5fbabda6db88d2ce4e0519`。
- 探针的音频实现基线 `core/src/vr/audio.rs` SHA-256 为 `4bb66bc338a253659d322a3a5c219c7deec59daeb27fe9cc4d9065529b5ee4c6`；随后取消错误移至共享 `task` 模块，数值调度未修改。重采样源码 SHA-256 为 `a24cea48ab719ff6c7a449c08e969b150af3ba95a68dc61f8aea9c9db3f9e80d`；多频带 DSP 为 `9b9ca9e165f64f9faa375bb81fec6a0e101fd9e05ab80c9285939dda5079a62c`。
- 数值实现基于 `b5418ed` 加本轮未提交变更。网络／权重版本见此前 [HP](2026-09-10-hp-cpu-network.md) 与 [DeEcho](2026-09-10-deecho-cpu-network.md) 记录；新增文件 I/O 依赖前的根锁文件保存在忽略目录 `vr-audio-Cargo.lock`。
- 参考 uv.lock SHA-256 为 `ab9f59b5bd7ffa40dc8983c4b924aa28c2138d84c185d641c626562113e40266`；探针 Cargo.lock 为 `9395d37ddc7d88273918d7942bd6d2d79a454b8258a47e73169b4bb7b6a1a4b4`。实际硬件快照仅在本机产物中记录。
- 运行方法见 [探针说明](backend-probe/README.md)。Rust 进程设置 `PATH=/nonexistent`，没有 Python 运行依赖。

这些是 `verification_only` 运行，部分检查期间存在编译任务。JSON 中的单次分阶段时间仅帮助发现开销，不能作为热运行、RTF 或与 Python 的速度比较。尚未完成协议规定的性能测量，也未验证模型听感、长音频内存、完整处理链或 GUI。
