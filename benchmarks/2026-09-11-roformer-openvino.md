# 1296 完整频谱网络的 OpenVINO 实验

日期：2026-09-11～12。目标是把已测到的模块收益推进到完整 PCM、Rust 原权重加载与产品调用。网络采用原始 1296 的全部分频投影、24 个时间／频率 Transformer、最终归一化与掩码头；保留 [工程基线 v1.3](../docs/baseline.md) 的权重身份、float64 正向 STFT、FP32 网络／ISTFT 和完整波形门限。主要性能输入是同一份本地 3 秒真实片段，单窗 132300 采样、301 帧；另完成 8 秒合成窗口的 GPU 单次正确性检查。跨窗、整曲及处理链仍需验证。

## 环境与安装

本机 i5-13420H 的 Intel UHD 核显此前缺少官方 OpenCL 运行时，不能因此排除 GPU。按用户明确要求，安装 `intel-compute-runtime 26.31.39395.13-1`、`intel-graphics-compiler 1:2.40.13-1` 和 `clinfo 3.0.25.02.14-1`；已有 `ocl-icd 2.3.5-1`。`clinfo -l` 在沙箱外枚举到官方 Intel 设备，OpenVINO 枚举到 `CPU` 和 `GPU`，后者名称为 `Intel(R) UHD Graphics (iGPU)`，设备 ID 为 `0xa7a8`。

Python 实验固定 OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`。Rust 探针新增独立可选依赖 `openvino 0.11.0`，通过 C API 动态加载相同版本的原生库；没有新增 unsafe。库从固定 wheel 复制到 `.local/openvino-2026.3.1/lib/`，逐文件摘要见 `openvino-native-libraries.json`，执行时 `PATH=/nonexistent`。这证明 Rust 执行路径；开发期 IR 仍由原始 PyTorch 模型转换，尚未实现产品直接从原始 checkpoint 构图。

## Python CPU 完整 PCM

`benchmark_roformer_openvino_audio.py` 先核对原模型包装的完整双轨波形，再转换固定 shape。CPU 为 FP32、8 线程、1 stream；实际不绑核、不启用超线程、允许所有核心类型。首次、额外预热和五次热运行的每次双轨输出均通过基线门限。

| 指标 | 结果 |
| --- | --- |
| 首次／额外预热 | 37.295／39.842 s |
| 五次热运行 | 38.853、34.096、34.542、34.600、36.757 s |
| 热运行中位数／范围 | 34.600 s／34.096～38.853 s |
| 中位数 RTF | 11.534 |
| 双轨最大绝对误差 | `5.2154e-8` |
| 人声／伴奏 RMSE | `8.1058e-9`／`8.1787e-9` |
| 原模型载入／转换／编译 | 2.091／278.076／3.551 s |

计时包含规定的 STFT、同步网络调用、输入／输出复制、ISTFT 和伴奏残差，排除载入／转换／编译、参考 I/O、文件编码和验证。网络占热运行约 34.071～38.822 秒；运行图的浮点内核保持 FP32，包含 AVX2 子图、196 个 `gemm_mlas_f32` 与 110 个 `brgemm_avx2_f32` 全连接节点。进程峰值内存未在这次 Python 运行中采集，不能用后续 Rust 进程数据补填。

这组结果支持推进完整后端。现有 Rust Burn 的同片段 8 线程记录为一次 81.450 秒网络运行，不能据它与本组中位数宣称经过完整对照的加速比；同协议的 Rust 后端重复测量继续补齐。

## Rust CPU 完整 PCM

同一 IR 通过 `openvino-rs 0.11.0` 执行，DSP 来自 core，运行时 `PATH=/nonexistent`。实际 CPU 设置同上：FP32、8 线程、1 stream、无绑核／超线程、ANY_CORE。首次、额外预热、五次热运行全部通过双轨门限。

| 指标 | 结果 |
| --- | --- |
| 首次／额外预热 | 26.819／25.010 s |
| 五次热运行 | 35.664、32.638、35.001、36.955、34.732 s |
| 热运行中位数／范围 | 35.001 s／32.638～36.955 s |
| 中位数 RTF | 11.667 |
| 双轨最大绝对误差 | `4.4703e-8` |
| 人声／伴奏 RMSE | `7.9626e-9`／`8.0381e-9` |
| IR 载入／编译 | 0.228／3.548 s |
| 完整进程墙钟／user／sys | 232.09／1334.12／9.91 s |
| 峰值 RSS | 1846208 KiB，约 1.76 GiB |

计时包含 Rust STFT、写入预分配输入 Tensor、同步网络执行、输出复制、Rust ISTFT 与残差；每次新建 FFT 计划，跟当前产品窗口边界一致。Rust 与 Python OpenVINO 的热运行范围重叠，不据此声称语言包装层带来加速。原始权重加载到该后端的实现与完整 Burn／原 PyTorch 同协议对照仍须推进；这不是产品性能验收。

## GPU 编译崩溃与对照

第一次 Rust 核显执行在 14.76 秒时收到 SIGSEGV；仅加载官方 Intel ICD 后再次在 12.33 秒崩溃。两次都尚未产生网络输出。崩溃栈位于 `libopencl-clang.so.17`，调用链包含 `libigdfcl.so.2`、`libigdrcl.so`、`clBuildProgram` 和 OpenVINO GPU 插件；不是波形门限失败。完整栈和资源记录保留，不能把 `/usr/bin/time` 的末尾 `Exit status: 0` 当成成功，进程状态实际为 139 且记录了 signal 11。

设备查询显示默认 `COMPILATION_NUM_THREADS=12`。单独用 `UVR_OV_COMPILATION_THREADS=1` 限制 OpenVINO 编译并发后，同一 IR 编译成功，耗时 35.104 秒，开始通过完整波形检查。这个结果支持继续检验编译并发相关问题，不等于已经修复编译器根因。此设置只控制图编译，网络仍用 GPU FP32；热运行结果继续记录。

该轮首次／额外预热分别为 9.231／16.901 秒且通过波形门限，第三次则停在驱动事件等待。GDB 栈为 `clWaitForEvents` → `libigdrcl.so` → `sched_yield`，内核同时记录该 PID 的 i915 fence 超时。按用户依赖安装原则补装 `intel-gpu-tools 2.5-1`；六个监控样本中，目标进程的 Render/3D 与 Blitter 活跃度均为 0，单次末尾设备分配约 1.883 GB。统一内存的设备分配与 RSS 不直接相加。

保存诊断后发送 SIGINT，确认进程以 130 退出；总墙钟 416.85 秒、user／sys 379.53／34.97 秒、峰值 RSS 669896 KiB。该轮没有完成五次热运行，前两次成功不能作为稳定性能结果。随后依据同版本 compute-runtime 的 `EnableDirectSubmission` 开关，继续检验驱动提交路径；新探针逐次将时间和误差刷新到 JSONL，外部超时保护避免故障运行无限等待。

设置 `NEOReadDebugKeys=1 EnableDirectSubmission=0` 后，前两次为 8.760／12.645 秒，波形门限通过，第三次再次出现相同 fence 超时，实验进程已发送 SIGINT 并确认以 130 退出。这个配置没有消除故障；保留的首轮／预热误差和耗时不作为稳定热运行结果。

Python OpenVINO 使用相同版本与 IR，先通过原模型包装的双轨检查，再在 GPU 完成一次 13.497 秒推理，第二次同样停在驱动等待，内核记录 Python PID 的 fence 超时。进程已发送 SIGTERM 并确认以 143 退出，只有部分 JSONL，没有完成报告。这将问题范围缩小到两种绑定共有的原生执行路径。另一个独立编译错误来自 `NUM_STREAMS` 的 Python 类型：整数 `1` 在该 GPU 插件的图编译入口被拒绝，小型 ReLU 图证实 `streams.Num(1)` 与字符串 `"1"` 可以编译；两个 Python 探针已改用正式的 `streams.Num(1)`。

## GPU 队列顺序实验

固定版本 OpenVINO 源码显示本机的非 oneDNN GPU 路径使用乱序队列，但发行包不接受内部 `GPU_QUEUE_TYPE` 属性。按依赖安装原则，将 Intel 官方 OpenCL Intercept Layer 3.0.6（提交 `1e888b29c2a30379946dee4b6c316639299e780e`）构建安装到 `.local/clintercept-3.0.6/`，通过 `CLI_InOrderQueue=1` 检验顺序队列。诊断层实际日志显示 `Queue properties: (None)`，确认没有 `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE`。

首个顺序队列进程正常退出，7 次完整双轨检查全部通过：首次 8.724 秒、额外预热 11.412 秒；五次热运行 17.776、17.429、16.548、16.820、17.062 秒，中位数 17.062 秒、RTF 5.687。双轨最大误差均为 `5.2154e-8`，RMSE 分别为 `7.6984e-9`／`7.7730e-9`。IR 载入／编译为 0.191／10.701 秒；完整进程墙钟 118.05 秒、user／sys 8.19／5.09 秒、峰值 RSS 485724 KiB。

单个进程成功不足以证明稳定性。随后保留诊断层但设置 `CLI_InOrderQueue=0`，日志确认恢复乱序队列；第一轮 10.304 秒通过，第二轮触发 fence 超时，300 秒保护以 124 退出。后续两轮顺序队列重复由同一个脚本串行执行，固定 V3 二进制、原生库、IR 与所有数值门限，均正常退出。

| 顺序队列重复 | 首次／额外预热 | 五次热运行（s） | 热运行中位数 | 峰值 RSS |
| --- | --- | --- | --- | --- |
| 第二个独立进程 | 10.799／20.490 s | 18.559、18.679、18.707、17.998、19.529 | 18.679 s | 487588 KiB |
| 第三个独立进程 | 22.646／21.125 s | 20.974、19.474、20.191、20.974、19.672 | 20.191 s | 487276 KiB |

三次进程共 21 个完整 PCM 检查全部通过，最大波形误差保持 `5.2154e-8`；两轮重复区间没有记录到所筛选的 i915／fence 内核故障。实测队列顺序与故障重现存在一致关系，支持继续排查提交／依赖路径，但没有定位驱动根因。不同进程的热运行范围漂移明显；不挑最快一轮代表产品性能，也不把诊断层当成可直接分发的修复。仍需不同 shape、完整窗口／整曲、取消与部署验证，以及受支持的运行时解决方法。

## Rust 直接从原始权重构图

2026-09-12 新增 Rust IR 构图器，严格核对原 `.ckpt` 的 SHA-256、全部 699 个张量的名称、shape、FP32 dtype 与有限值，不读取 Python 生成的 IR。保留精确 GELU、L2 范数下限 `1e-12`、全部注意力序列与掩码计算；每个固定 shape 生成 5989 个 IR 节点，含常量。301 帧版本构图和写文件耗时 4.914 秒；这不含后端编译，不能与热运行时间混合。生成过程和 Rust 推理都在 `PATH=/nonexistent` 下执行。

| Rust 构图实验 | 首次／额外预热 | 五次热运行（s） | 中位数 | 峰值 RSS |
| --- | --- | --- | --- | --- |
| CPU FP32，8 线程 | 29.355／29.764 s | 36.662、37.716、39.030、41.627、34.376 | 37.716 s | 9164888 KiB，约 8.74 GiB |
| GPU FP32，顺序队列 | 14.252／18.027 s | 19.394、16.034、16.343、16.110、16.195 | 16.195 s | 686624 KiB |

各次完整双轨波形均通过。CPU 双轨最大误差 `4.4703e-8`，RMSE `7.9626e-9`／`8.0381e-9`；GPU 最大误差 `5.2154e-8`，RMSE `7.6984e-9`／`7.7730e-9`。CPU／GPU 编译分别为 8.417／33.340 秒。另将同一构图规则应用于 801 帧、352800 采样的完整合成窗口：GPU 编译 32.500 秒，一次完整 STFT→网络→ISTFT 为 53.237 秒，最大误差 `2.3097e-7`、整轨 RMSE `2.0802e-9`，峰值 RSS 677696 KiB；这是单次正确性结果，没有完整窗五次热运行结论。两组 GPU 区间均未记录到所筛选的内核故障。

CPU 内存明显高于原转换图。按用户在 2026-09-12 明确选择的速度优先策略，8.74 GiB 本身不构成失败；继续测实际余量、完整任务速度和响应，允许内存投入换取加速。两份图的 CPU 热运行范围有重叠，这些数据尚不能证明手工构图改善了热推理性能；它首先补齐不依赖 Python 转换的原权重路径。

## 共享 core 与 CLI 的原权重路径

构图已移动到 core 的可选 `openvino` 特性，共用原 checkpoint 加载器与音频分窗／重采样／重叠归并；图和权重在内存中传给原生运行时，输入 Tensor 与 FFT 计划跨窗复用。当前产品候选只开放 CPU。异步请求以非负超时轮询，发出取消后等待原生请求终止，再允许复用。首版误把绑定的 `wait_for(-1)` 当作无期限等待，取消检查失败；改为非负超时轮询后，以下检查通过，失败日志保留。

同一真实 3 秒输入中，原生请求提交后取消到 core 返回为 0.001160 秒；复用同一请求完成完整波形，双轨最大误差 `5.2154e-8`，RMSE `7.9921e-9`／`8.0640e-9`。原权重构图 4.198 秒、编译准备 9.338 秒、总加载 13.588 秒；一次完整窗口调用为 26.917 秒，进程峰值 RSS 8932132 KiB，约 8.52 GiB。这是接口验证，不是五次热运行的最终性能结论。

四组边界音频也通过：4097 采样双声道信号、单采样单声道、7 采样 22.05 kHz 重采样和 48 kHz 静音。输出长度、双轨门限、完成进度均符合共享调度；静音严格为零并跳过网络。信号样本另一次原生取消为 0.000840 秒，同一请求随后成功复用。

CLI 使用 `--backend openvino-cpu`，直接读取原始 `.ckpt`，运行时仍为 `PATH=/nonexistent`。真实片段两轨浮点 WAV 与参考吻合；中文路径、拒绝覆盖、Ctrl-C 退出 130 和无残缺输出均通过。此次完整文件任务 41.489 秒，包括加载 13.162 秒与窗口网络 27.435 秒；Ctrl-C 到进程退出为 0.895 秒，包含模型销毁等进程收尾，不能与 core 返回延迟混为一个指标。尚未完成该入口的整曲、跨窗及 GUI 验收，默认后端继续由完整性能证据决定。

```sh
cargo build --release --locked -p uvr-cli --features openvino
env LD_LIBRARY_PATH=/home/fuurin/code/UVR/.local/openvino-2026.3.1/lib RAYON_NUM_THREADS=8 target/release/uvr separate-1296 models/model_bs_roformer_ep_368_sdr_12.9628.ckpt <输入音频> <输出目录> --backend openvino-cpu
```

## CPU 执行图与 RoPE 成对拆分

2026-09-12 用 `profile_roformer_openvino.py` 比较原转换图、原 Rust 图和只改变 RoPE 提取方式的候选。设备、FP32、8 线程、单 stream 和真实 3 秒输入固定；每图两次完整波形均通过，另保存逐算子 profiling 和原生执行图。以下编译及带采样执行数据用于定位，正式热运行对照另记。

| 图 | 编译／请求准备 | 准备后的 RSS | 两次带 profiling 的网络耗时 |
| --- | --- | --- | --- |
| 原转换图 | 3.054 s | 1912428 KiB | 26.230／25.904 s |
| 原 Rust 图 | 7.852 s | 9094632 KiB | 27.149／31.188 s |
| Rust RoPE 成对拆分 | 3.023 s | 1919916 KiB | 26.622／26.211 s |

原 Rust 执行图含 96 个 `StridedSlice`；候选先把 head 最后一维拆为 `[32, 2]`，再用 `Split` 取奇偶分量，保留 `[-odd, even]`、cos／sin、完整注意力和原权重。这 96 个执行节点被 48 个 `Split` 替代，原生图仍有 306 个 FullyConnected 与 341 个 Subgraph。Rust 构图节点由 5989 降为 5509。

固定版本 OpenVINO 的 [StridedSlice 源码](https://github.com/openvinotoolkit/openvino/blob/2026.3.1/src/plugins/intel_cpu/src/nodes/strided_slice.cpp#L793) 在准备阶段预计算 `srcIndices`／`dstIndices`。本例的末轴步长 2 使每个输出元素需要两个 `size_t` 索引；96 个节点、每个 `62 × 301 × 8 × 32` 元素，在本机共约 6.83 GiB，吻合 RSS 差异。它是索引表开销，不是额外音频 batch 或有收益的权重缓存；成对拆分同时减少准备工作，为完整窗和并行保留空间，未引入 RSS 验收上限。

三个图的线性层均主要使用 `gemm_mlas_f32` 和 `brgemm_avx2_f32`；候选第二次查询时，二者的聚合算子计数分别约为 12.403／7.383 秒。CPU profiling 计数是已执行调用的累计平均，表中网络墙钟则为每次独立调用，二者不能混同。相应 [MLAS 执行器源码](https://github.com/openvinotoolkit/openvino/blob/2026.3.1/src/plugins/intel_cpu/src/nodes/executors/mlas/mlas_gemm.cpp) 已在创建时预打包并缓存权重，不能把热执行开销直接归因于反复权重打包。继续检验线程配置、窗口并行和实际矩阵内核。

## 产物与复现

原始产物统一放 `benchmarks/artifacts/2026-09-11-roformer-performance/`：

- `openvino-pcm-t8.json`、`.run.log` 与 `openvino-pcm-cpu-source.py`：Python CPU 全部样本和实测脚本快照。脚本 SHA-256 为 `84173a4cf56f26c28709de8587e19a67822a67e807bc6f0ea72e5ab6d1240f06`。
- `openvino-1296-301.xml/.bin`：固定 301 帧、未压缩为 FP16 的完整频谱图。
- `openvino-rust-v1-sha256.json` 与 `openvino-rust-v1.rs`：初次 Rust 探针源码、锁文件和二进制身份。
- `openvino-rust-pcm-gpu*`、`openvino-rust-gpu-crash.txt` 和 `openvino-rust-gpu-intel-crash.txt`：初次与 ICD 隔离后的崩溃、资源记录及后续编译并发实验。
- `openvino-gpu-run3-root-stacks.txt`、`openvino-gpu-kernel-events.txt`、`openvino-gpu-run3-engines.json`：第三次推理的等待栈、内核事件和设备计数器。`intel_gpu_top -J -o` 把外层方括号输出到 stdout，文件内保存逗号分隔的对象；解析时额外包裹方括号，原文件保留。
- `openvino-rust-v3-sha256.json`、`openvino-rust-v3.rs`：加入逐次 JSONL 与已知驱动环境记录后的探针身份。
- `openvino-rust-pcm-cpu-t8.json/.jsonl/.log/.time`：Rust CPU 的七次完整门限、全部时间、实际配置及资源。二进制 SHA-256 为 `f98e0138851aadb5c5a3669e2bf07e48a9650ccc9178d75f164fddd646468052`，报告同时保存 IR、权重数据与输入清单摘要。
- `openvino-python-pcm-gpu-streams.*`、`openvino-python-gpu-kernel-events.txt`、`openvino-pcm-python-gpu-streams-source.py`：Python 同图重现、类型修正和实测源码快照。
- `clintercept-install.json`、`openvino-rust-pcm-gpu-inorder.*`：诊断层来源、文件摘要、首个顺序队列成功进程；`.time` 完整保存实际环境设置，包括探针 JSON 未收集的 `LD_PRELOAD` 和 `CLI_*`。
- `run-openvino-queue-controls.py`、`openvino-rust-pcm-gpu-queue-*`：关闭强制顺序的对照及独立重复。各次 `.invocation.json` 在运行前保存完整命令、环境覆盖和库／二进制摘要，结束时补实际退出码；`.kernel.log` 保留该次执行区间的内核故障。
- `openvino-rust-1296-{301,801}.{xml,bin,provenance.json}`、`openvino-rust-export-{301,801}.time`：原权重 Rust 构图、来源和全部数据摘要。
- `openvino-rust-graph-v1.rs`、`openvino-rust-export-v1.rs`、`openvino-rust-v4.rs`、`openvino-rust-graph-v1-sha256.json`：此轮实测代码与二进制身份。构图器后来移到 core，使用这些快照复核当时实现。
- `run-openvino-rust-graph-validation.py`、`openvino-rust-checkpoint-*`：新图的单次预检、CPU／GPU 热运行与 801 帧检查，每次均保留完整命令、退出码、时间、资源和逐次门限。
- `openvino-core-real-audio*`、`openvino-core-boundaries*`：直接从 checkpoint 到共享 PCM 调度、原生取消和复用；`v2` 为修正取消收尾后的真实片段报告，最初失败日志仍在。
- `openvino-cli-real-audio/`：真实 WAV、双轨摘要、完整命令、运行环境、覆盖拒绝和取消检查；外层 `.time` 的资源统计包含验证进程及其 CLI 子进程，不用作单个 CLI 进程的独立 RSS 结论。
- `openvino-core-v2-sha256.json`、`openvino-core-v2.rs`、`openvino-core-ir-v2.rs`、`verify-vr-cli-openvino-source.py`：通过实际接口检查的源码与二进制身份。CLI SHA-256 为 `7db6495f89c238e5af84d7c54fb7c2c30c7a6385f8977d39d38cbb913c95e8c8`。
- `openvino-{original,rust,rust-split}-profile.json` 及 `.runtime.xml`：CPU 三图的诊断执行、逐算子耗时、实际内核、内存阶段与波形门限。
- `openvino-rust-split-{301,801}.{xml,bin,provenance.json}`、`run-openvino-rope-split.py`：成对拆分候选与无采样 A→B→A／完整窗实验；每轮的 `.invocation.json`、`.resources.jsonl`、`.time` 保存命令、摘要、逐秒进程资源和实际系统余量。

执行入口、参数、原生库路径与计时边界见 [Python 参考工具](../tools/reference/README.md#openvino-开发期后端实验) 和 [Rust 后端探针](backend-probe/README.md#openvino-rust-完整单窗实验)。GPU 计时使用同步 `InferRequest::infer`，输入 Tensor 在准备时创建，每次计入特征填入；输出复制、Rust DSP 和伴奏残差都计时。

采用仍需更多 shape／整曲／跨窗、同协议性能对照、缓存策略、系统内存余量与 GUI 验证；GPU 还需受支持的稳定运行时配置。模块或单窗证据不能替代四套模型和最终处理链验收。

新增可选 OpenVINO 特性的 release 探针构建与全目标 Clippy 通过；两个 workspace 的格式检查、Python 脚本语法与 diff 空白检查通过。完整 PCM 使用以上真实参考检查，不以检查工具通过替代数值或性能证据。
