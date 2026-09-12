# DeEcho 完整 CPU 掩码网络

日期：2026-09-10。core 的 `burn-cpu` 实现直接读取原始 `UVR-DeEcho-DeReverb.pth`，严格校验全部 689 个张量，执行完整 CascadedNet，包括各向异性空洞卷积、五个子网络、双向 LSTM、线性投影与 BatchNorm。权重身份见 [权重记录](../references/verified-weights.json)。

三组样本的全部重复执行均通过既定逐元素门限 `1e-3 + 1e-4 × abs(reference)`。输入是合成幅度窗口，输出已复制 Nyquist 频点并裁去左右各 64 帧。

| 样本／输入帧数 | 输出形状 | 首次执行 s | 热运行中位数（范围）s | 最大绝对误差 | RMSE |
| --- | --- | --- | --- | --- | --- |
| 普通幅度谱／320 | `[1,2,673,192]` | 10.096 | 9.811（9.737–10.644） | 2.8014e-6 | 1.4961e-8 |
| 静音／144 | `[1,2,673,16]` | 4.368 | 4.249（3.959–4.545） | 2.6453e-4 | 5.6872e-6 |
| 脉冲／176 | `[1,2,673,48]` | 5.527 | 5.146（4.989–5.354） | 8.0246e-4 | 1.8479e-5 |

静音和稀疏脉冲的最大掩码误差明显高于普通输入，脉冲已接近绝对门限。当前证据不能确定误差来自哪个阶段，也不能据掩码误差推断音频听感；完整音频对照继续使用预先固定的独立门限。

模型加载约 1.056 秒，包含摘要、checkpoint 读取、校验及权重准备，未控制文件系统缓存。整个探针进程峰值 RSS 为 673056 KiB，包含驻留参考样本；墙钟 2:17.45，不是歌曲处理时间。

复现证据：

- 源码基线为 `b5418ed` 加本轮未提交改动；`core/src/vr.rs` SHA-256 为 `faba07e36fe6be2d33962dd7d733b05d833f70850cfbf091208a2227917b4c3c`，`core/src/vr/deecho.rs` 为 `b6be68d602fd9c201dc085f656044d25c409f9a68d96a83e4635b67c15207fef`。
- rustc 1.98.1、默认 release、RUSTFLAGS 未设置、Burn Flex／burn-store 0.21.0、CPU FP32、batch=1、2 线程。本机硬件快照保存在忽略目录。
- 根 Cargo.lock SHA-256：`3a343be48b21c00b321dd6173ca775f38c985c75016c394cfa9469c57294c690`；探针 Cargo.lock：`9395d37ddc7d88273918d7942bd6d2d79a454b8258a47e73169b4bb7b6a1a4b4`。
- UVR 参考 `5517e0cf0d1acd16a1618eeedec596957523f9e1`、PyTorch `2.13.0+cpu`；生成时 uv.lock SHA-256：`2590ac3d78e0974aab908c11ba8a1a6bc4ab78a0ed38902e1382576a4edddc04`。
- 样本清单 SHA-256：`c577e804e66ce6ab7e9c791b228fe9ddefe03fd7849f0038be40bcf9614e9926`；原始报告为 `benchmarks/artifacts/backend-probe/deecho-model.json`，进程资源为同目录 `deecho-model-time.txt`。
- 首次执行后额外预热 1 次，再计时 5 次；计时包含输入构建、完整网络和输出物化，排除文件读取、模型载入和误差计算。测量进程设置 `PATH=/nonexistent`，模型加载及推理无需 Python。
- 生成与运行方法见 [探针说明](backend-probe/README.md)。接口另外验证错误尺寸、非法窗口、NaN、负幅度返回错误；core 与探针 Clippy、release 构建通过。

这完成了 DeEcho 的 Rust 网络合成对照，尚未覆盖多频带音频重建、去混响质量、整曲 RTF 或其他平台。
