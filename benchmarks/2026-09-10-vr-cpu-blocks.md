# 5-HP CPU 模块比较

日期：2026-09-10。问题：Candle CPU 与 Burn Flex 能否匹配实际 5-HP 模块，以及这些形状的局部成本有何差异。结果只覆盖下表，未决定整模型后端。

两个后端均通过全部输出形状和逐元素误差检查。卷积块最大绝对误差为 `3.8147e-5`，上采样最大绝对误差不超过 `3.3379e-6`。验收使用 `1e-3 + 1e-4 * abs(reference)`，所有预热与计时运行均检查，误差计算在计时外。

| 模块／输入形状 | Candle 热运行中位数（范围）ms | Burn Flex 热运行中位数（范围）ms |
| --- | --- | --- |
| 第一卷积块 `[1,2,336,320]` | 43.517（43.005–44.089） | 17.089（16.928–20.038） |
| stride=2 编码块 `[1,32,336,320]` | 18.411（17.902–18.641） | 9.645（9.563–9.746） |
| ASPP 空洞 depthwise／pointwise 块 `[1,256,21,20]` | 3.200（3.115–3.275） | 0.516（0.460–0.520） |
| 双线性上采样、align_corners=true `[1,512,21,20]` | 2.260（2.223–2.298） | 3.807（3.593–3.869） |
| 双线性上采样、align_corners=false `[1,512,21,20]` | 2.240（2.228–2.412） | 3.785（3.613–3.821） |

首次模块执行分别为 Candle `47.735 / 19.104 / 3.488 / 2.574 / 2.389 ms`、Burn `20.195 / 12.036 / 0.733 / 3.895 / 3.742 ms`。它们不含进程启动与文件载入，因此不能当作完整冷启动。

`/usr/bin/time -v` 记录的整个探针进程峰值 RSS：Candle 153612 KiB、Burn 114576 KiB，包含所有输入、参考输出和张量准备，不能视为完整模型或单模块的峰值额外内存。

构建与数据：

- 项目基线 `b5418ed` 加本轮未提交实验；独立 crate 位于 `benchmarks/backend-probe/`。
- rustc 1.98.1，x86_64 Linux，默认 release 构建，未设置 RUSTFLAGS。Candle 0.10.2 CPU（default-features=false）、Burn Flex 0.21.0（std、SIMD、Rayon）。
- 实验 Cargo.lock SHA-256：`c70ca0c1ad53dbbc6e696fe93cc9263ceb68010bbce08a64ff4f64abdc9d0c89`。
- 验证 uv.lock SHA-256：`2590ac3d78e0974aab908c11ba8a1a6bc4ab78a0ed38902e1382576a4edddc04`；PyTorch `2.13.0+cpu`。
- 5-HP 权重 SHA-256：`fe00891defbb61f4261500af22f7624f1a3df8dc75fa3998d1aece02e6be4537`；原网络参考 `5517e0cf0d1acd16a1618eeedec596957523f9e1`。
- 探针 manifest SHA-256：`e15b4c718dba74621af1e5f9bc8e8fc04bd42b6afdb3577b420337ad10f19672`。清单含每个输入、权重、预期输出的尺寸和 SHA-256；数据源于严格加载后原模型的合成前向中间激活。
- FP32、batch=1、RAYON_NUM_THREADS=2，两个程序顺序执行。计时包括模块运算、临时张量与输出物化；排除 I/O 和初始张量准备。每个模块首次执行后另预热 1 次，再测 5 次，保留全部样本。
- 本机 CPU、内存、系统、governor、工具链快照只保存在忽略目录 `benchmarks/artifacts/backend-probe/`，遵守硬件信息的本地记录约定。未控制系统其他应用，也未控制文件系统缓存；不据此报告进程冷启动指标。

复现命令见 [探针说明](backend-probe/README.md)。原始全部样本和误差位于 `benchmarks/artifacts/backend-probe/{candle,burn}.json`，进程资源位于同目录 `*-time.txt`。

结论：这轮支持继续评估两个后端。Burn Flex 在这些卷积块上耗时更低，Candle 在这两种双线性上采样上耗时更低；不能将局部耗时相加推算整模型 RTF。还需覆盖双向 LSTM、RoFormer、真实窗口下的内存和一个完整模型，才能作主后端选择。未进行音频质量、Windows 或客户端硬件验收。
