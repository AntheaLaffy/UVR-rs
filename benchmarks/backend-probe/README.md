# CPU 后端探针

以实际 5-HP、DeEcho 和 1296 权重及中间激活比较 Candle CPU 与 Burn Flex。此 crate 有独立 Cargo workspace／锁文件；模块结果用于缩小兼容性与性能风险，不代表整模型速度排名。完整 HP Karaoke 原型位于 core 的可选 `burn-cpu` 特性下，探针直接调用该实现。

从仓库根目录准备输入并构建：

```sh
uv run --frozen --project tools/reference python -B tools/reference/audit_vr.py --model 5_HP-Karaoke-UVR.pth --threads 2 --forward --output benchmarks/artifacts/vr-reference/5hp.json
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_probe.py --audit benchmarks/artifacts/vr-reference/5hp.json --output benchmarks/artifacts/backend-probe/5hp
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --bins
```

两个测量必须顺序运行，避免争用 CPU：

```sh
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/candle-time.txt env RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-candle benchmarks/artifacts/backend-probe/5hp benchmarks/artifacts/backend-probe/candle.json
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/burn-time.txt env RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-burn benchmarks/artifacts/backend-probe/5hp benchmarks/artifacts/backend-probe/burn.json
```

使用 Candle 0.10.2 CPU（关闭默认特性）、Burn Flex 0.21.0（std、SIMD、Rayon），FP32、batch=1、相同线程预算。输入包括第一层卷积块、stride=2 编码块、ASPP 的空洞 depthwise／pointwise 卷积块，以及 ASPP 输出上的双线性上采样。卷积块包含未折叠 BatchNorm 和真实激活函数；上采样覆盖 `align_corners=true`，另用 false 作为边界控制。所有输入、权重与参考结果都有 SHA-256，测量前校验。

计时包含 CPU 运算、临时张量与输出物化，排除文件 I/O 和初始张量准备。分别记录首次执行、额外预热 1 次后的 5 次样本；首次执行不是完整进程冷启动。每次输出都检查形状与逐元素误差，当前阈值为 `1e-3 + 1e-4 * abs(reference)`；不达标则非零退出，不生成成功报告。

`time -v` 的峰值 RSS 涵盖整个探针进程，包括输入和期望输出驻留，不能视为单模块的额外内存。原始报告放 Git 忽略目录；分享摘要时记录版本、锁文件／样本摘要、构建设置和本机约束。模块实验未覆盖整曲调度、Windows 和听评，不能据这些局部结果直接确定最终后端。

## LSTM 与 RoFormer

先依照 [验证工具说明](../../tools/reference/README.md) 准备 DeEcho、1296 的固定权重、配置、源码与严格加载报告，然后执行：

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_sequence_probe.py --output benchmarks/artifacts/backend-probe/sequence
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/sequence-candle-time.txt env RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-candle benchmarks/artifacts/backend-probe/sequence benchmarks/artifacts/backend-probe/sequence-candle.json
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/sequence-burn-time.txt env RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-burn benchmarks/artifacts/backend-probe/sequence benchmarks/artifacts/backend-probe/sequence-burn.json
```

需要 DeEcho 报告位于 `benchmarks/artifacts/vr-reference/deecho.json`，1296 报告位于同目录 `1296.json`。生成器校验权重／源码／配置摘要并再次严格加载，再捕获真实前向输入。

8 个样本覆盖 DeEcho 三种 LSTM 尺寸、短序列双批次、1296 时间／频率注意力、零输入和前馈。LSTM 按基础算子实现，不含线性投影后的 BatchNorm；注意力包含 RMSNorm、RoPE、Softmax、门控和投影，前馈使用精确 GELU。RoFormer 保留实际 801／62 的序列长度，仅取部分独立批次条目；具体形状、计时边界与结果见 [实验记录](../2026-09-10-sequence-cpu-blocks.md)。

## 完整 VR 掩码网络

`probe-vr` 从原始 `.pth` 在 Rust 内加载模型，不需要导出权重。下面的 Python 命令只产生测试输入与参考掩码：

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_model.py --audit benchmarks/artifacts/vr-reference/5hp.json --output benchmarks/artifacts/backend-probe/5hp-model
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --bin probe-vr
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/5hp-model-time.txt /usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-vr models/5_HP-Karaoke-UVR.pth benchmarks/artifacts/backend-probe/5hp-model benchmarks/artifacts/backend-probe/5hp-model.json
```

6-HP 使用 `6hp.json`、`6hp-model` 和 `models/6_HP-Karaoke-UVR.pth` 替换对应参数，两个测量顺序运行。三组输入为 320 帧普通幅度谱、272 帧静音、304 帧脉冲；每次测量都包含完整网络、Nyquist 复制、上下文裁剪、输入张量构建和输出检查前的物化。文件 I/O、模型加载在计时循环前，加载耗时另外打印。错误窗口长度、输入尺寸、NaN 和负幅度也必须被接口拒绝。

此接口输入是已经准备好的多频带幅度谱。模型用 128 帧左右上下文，输出时间宽度是输入减 256；这与整段音频的采样长度不同。当前没有解码、重采样、掩码后处理、分轨重建或整曲 RTF，不能把掩码测试当作音频分离验收。

DeEcho 使用同一个探针，显式指定 `--deecho`；其上下文为左右各 64 帧，三组输入长度为 320／144／176 帧：

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_model.py --audit benchmarks/artifacts/vr-reference/deecho.json --output benchmarks/artifacts/backend-probe/deecho-model
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/deecho-model-time.txt /usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-vr --deecho models/UVR-DeEcho-DeReverb.pth benchmarks/artifacts/backend-probe/deecho-model benchmarks/artifacts/backend-probe/deecho-model.json
```

只检查数值回归时增加 `--check-only`，每个样本执行一次，报告标为 `verification_only`，没有热运行计时字段。共享代码修改后可用它验证各网络，不必重复完整性能测量：

```sh
/usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-vr --check-only models/5_HP-Karaoke-UVR.pth benchmarks/artifacts/backend-probe/5hp-model benchmarks/artifacts/backend-probe/5hp-model-regression.json
```

## 完整 PCM 音频

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_audio.py --variant 5hp --output benchmarks/artifacts/backend-probe/5hp-audio
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --bin probe-vr-audio
/usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-vr-audio models/5_HP-Karaoke-UVR.pth benchmarks/artifacts/backend-probe/5hp-audio benchmarks/artifacts/backend-probe/5hp-audio.json
```

其他模型使用对应 `6hp`／`deecho` 生成参数和原始权重路径。模型类型、采样率、窗口与参考窗口数由样本清单提供；探针调用 core 的完整 `VrSeparator`，检查两轨波形、精确长度、有限值、逐元素及 RMS 误差、阶段进度和取消。阈值来自工程基线，和单独掩码探针不同。

默认完整 PCM 检查每组只执行一次，报告 `verification_only`，包含单次阶段时间以定位开销。文件到文件验证另外使用 [CLI 验证工具](../../tools/reference/README.md)。合成结果见 [VR 音频记录](../2026-09-10-vr-cpu-audio.md)。

性能实验添加 `--measure`，每组输入依次首次执行、额外预热一次、热运行五次，始终复用同一个模型。每次输出均检查两轨逐样本和 RMS 门限；计时只包围完整 PCM 分离，排除误差计算、输入文件读取、模型加载和编码。报告保留全部阶段／窗口耗时、RTF 和首次进度时间，不将首次执行误称为进程冷启动。`--case` 可选单个样本；不指定时测清单中的所有样本。

```sh
/usr/bin/time -v -o benchmarks/artifacts/backend-probe/5hp-pcm-perf.time env RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-vr-audio --measure --case local_audio models/5_HP-Karaoke-UVR.pth benchmarks/artifacts/backend-probe/5hp-real-audio benchmarks/artifacts/backend-probe/5hp-pcm-perf.json
uv run --offline --frozen --project tools/reference python -B tools/reference/benchmark_vr_audio_reference.py --fixtures benchmarks/artifacts/backend-probe/5hp-real-audio --output benchmarks/artifacts/backend-probe/5hp-pcm-reference-perf.json --threads 2
```

两条命令顺序执行。Python 参考使用相同清单、权重、配置、预设、计时边界和两轨门限，也记录首次执行、额外预热和五次热运行。Rust／参考的阶段划分有少量准备工作的边界差异，因此以完整 PCM 总时间比较，解码／编码性能另测。

## 完整 RoFormer 网络与音频

```sh
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --bin probe-roformer --bin probe-roformer-audio
/usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-roformer models/model_bs_roformer_ep_368_sdr_12.9628.ckpt benchmarks/artifacts/backend-probe/1296-f64-model benchmarks/artifacts/backend-probe/1296-f64-model.json
/usr/bin/env PATH=/nonexistent RAYON_NUM_THREADS=2 benchmarks/backend-probe/target/release/probe-roformer-audio models/model_bs_roformer_ep_368_sdr_12.9628.ckpt benchmarks/artifacts/backend-probe/1296-f64-audio benchmarks/artifacts/backend-probe/1296-f64-audio.json
```

两个探针均从原始 checkpoint 严格加载全部 699 个张量。前者检查原始完整网络波形、自然 ISTFT 长度、进度与取消；后者调用 core 的完整音频接口，检查精确长度、单／双声道、重采样、尾窗、交叠、两轨波形与窗内取消。数据由 [独立生成器](../../tools/reference/README.md) 生成。它们按工程基线的完整波形门限检查，报告单次 `verification_only` 结果，不将计时作为五次热运行性能结论。

当前采用 v1.1 的 float64 正向频谱、FP32 网络与重建，需生成器的 `--stft64` 样本。原始 FP32 频谱的成功／失败产物保留作历史记录，不能覆盖后宣称与原结果完全一致。`probe-roformer --dump-output` 可在检查前将实际波形写为 `<报告名>.<样本序号>.f32`，包括未通过门限时的输出；为保留证据，不覆盖既有该文件。逐样本门限作用于全部声道，RMS 门限按双声道整轨计算。

`probe-roformer --measure --case <样本>` 可测原始完整网络的首次执行、额外预热和五次热运行，每次都验证波形。报告按 `execution=first/warmup/warm` 区分，记录 `single_execution_seconds` 与进度步骤间的 `step_seconds`；计时包含精确 STFT、网络及 ISTFT，排除加载、参考读取与验证，不包含文件调度、解码和编码。`--dump-output` 在重复测量时只保存首次波形。报告路径必须不存在。

`probe-burn` 的 RoFormer 模块可用 `UVR_PROBE_CASE_PREFIX=1296` 筛选。`UVR_LINEAR_LAYOUT=batched/flattened` 比较线性层布局；`UVR_PARALLEL_GELU=1`、`UVR_FUSED_RMS_NORM=1`、`UVR_FUSED_ROPE=1` 分别启用候选，默认均关闭。探针直接编译 core 的同一 CPU 内核源码，数值对照仍来自独立 Python 清单。实验中只改变指定开关，固定总线程预算并顺序执行，详见 [1296 优化记录](../2026-09-11-roformer-cpu-optimization.md)。

## OpenVINO Rust 完整单窗实验

`probe-openvino-audio` 使用安全的 OpenVINO Rust 绑定和 core 的原始 float64 STFT／float32 ISTFT，执行完整频谱网络并逐次检查两轨波形。这个入口读取固定 shape IR，用于测 Rust 调用和设备执行；图既可由参考工具转换，也可由下述 Rust 构图器从原始 checkpoint 生成。共享调度与取消通过单独的 core 探针验证。

```sh
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --features openvino --bin probe-openvino-audio
env LD_LIBRARY_PATH=/home/fuurin/code/UVR/.local/openvino-2026.3.1/lib RAYON_NUM_THREADS=8 PATH=/nonexistent benchmarks/backend-probe/target/release/probe-openvino-audio CPU benchmarks/artifacts/2026-09-11-roformer-performance/openvino-1296-301.xml benchmarks/artifacts/backend-probe/1296-f64-real-audio local_audio benchmarks/artifacts/backend-probe/1296-openvino-rust-pcm.json
```

本机从固定 OpenVINO 2026.3.1 wheel 的 `openvino/libs/` 复制原生库到上述本地目录，增加 `libopenvino_c.so` 到 `libopenvino_c.so.2631` 的符号链接；来源与逐文件摘要保存在实验目录的 `openvino-native-libraries.json`。其他机器将 `LD_LIBRARY_PATH` 换为自己的原生运行时位置。执行不调用 Python，输出必须不存在。输入必须是完整 hop 的单窗，IR 尺寸不符直接拒绝。

将 `CPU` 改为 `GPU` 可测核显。OpenCL 设备运行时必须安装并可访问；`OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd` 可限定官方 Intel 驱动，`UVR_OV_COMPILATION_THREADS=1` 用于单独诊断图编译并发，均不改变网络精度。报告包含 IR、权重数据与二进制摘要，以及首次、额外预热、五次热运行的分阶段时间和波形检查。同步推理和输出复制计入网络时间，输入 Tensor 在加载阶段分配、每次计入特征写入；用 `/usr/bin/time -v` 另记进程资源。结果与限制见 [完整后端实验](../2026-09-11-roformer-openvino.md)。

CPU 探针另支持 `UVR_OV_CPU_PINNING=YES|NO`、`UVR_OV_HYPER_THREADING=YES|NO` 和 `UVR_OV_CORE_TYPE=ANY_CORE|PCORE_ONLY|ECORE_ONLY`，报告同时记录请求值和后端实际采用值。标准 Python 包装器 `tools/reference/run_roformer_openvino_probe.py` 暴露相应参数并采样系统可用内存；执行图与逐算子分析工具见 [参考工具说明](../../tools/reference/README.md#openvino-开发期后端实验)。默认探针计时不启用算子 profiling。

`--check-only` 只做一次完整波形检查，报告明确标记为验证用途。若 IR 旁有 `.provenance.json`，探针验证 XML／权重摘要及原 checkpoint 与参考清单的一致性。当前核显还需实验记录中的顺序队列诊断配置，不能仅凭 `GPU` 枚举成功视为稳定可用。

Rust 构图器与 core 后端共用实现，严格消费原始 699 个张量，接受 4～801 帧：

```sh
cargo build --release --locked --manifest-path benchmarks/backend-probe/Cargo.toml --features openvino --bin export-roformer-openvino --bin probe-openvino-core-audio
env PATH=/nonexistent benchmarks/backend-probe/target/release/export-roformer-openvino models/model_bs_roformer_ep_368_sdr_12.9628.ckpt 301 benchmarks/artifacts/backend-probe/1296-rust-301.xml
env LD_LIBRARY_PATH=/home/fuurin/code/UVR/.local/openvino-2026.3.1/lib RAYON_NUM_THREADS=8 PATH=/nonexistent benchmarks/backend-probe/target/release/probe-openvino-core-audio models/model_bs_roformer_ep_368_sdr_12.9628.ckpt benchmarks/artifacts/backend-probe/1296-f64-real-audio benchmarks/artifacts/backend-probe/1296-openvino-core.json
```

core 探针直接加载 `.ckpt`，在内存中构图并编译，调用共享的重采样、分窗、重叠归并与伴奏残差。首个非静音样本检查原生请求提交后的取消，再复用同一个请求验证完整输出；各样本记录实际线程设置、加载／编译与波形误差。它用于接口正确性和响应验证，不提供五次热运行的性能结论。
