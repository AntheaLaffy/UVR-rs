# 推理框架与专用实现：先验比较

日期：2026-09-09。状态：讨论草案。本轮核对了官方文档和当前主分支资料，没有导出或运行目标模型，也没有得到框架性能排名。依赖能力必须在实验时固定版本后重新验证。

本文件修订 [H05](inference-priors.md#h05rust-技术栈候选) 的候选顺序：Burn 与 Candle 并列比较；ONNX 路线进入讨论，是否采用原生运行时取决于纯 Rust 边界。

## B01：区分模型格式、框架与执行内核

ONNX 是计算图、算子与张量交换格式，不等于 ONNX Runtime，也不直接决定实现语言。以下路线可以分别实验：

| 路线 | Rust 侧职责 | 要验证的风险 |
| --- | --- | --- |
| 手写 Burn 模型 + 合适 CPU 后端 | 在 Rust 中描述网络，使用后端算子 | 网络语义、权重映射、后端覆盖与实际形状性能 |
| ONNX → burn-onnx → Burn | 生成可修改的 Rust 网络代码及权重产物 | 导出与导入覆盖、生成代码、后端算子三个环节 |
| 手写 Candle 模型 + CPU 后端 | 在 Rust 中描述网络与兼容处理 | 算子细节、调度开销、布局与内存 |
| ONNX → tract | Rust 集成 ONNX 导入、图优化与执行 | 固定版本的算子／opset／形状支持，目标平台运行 |
| ONNX → ONNX Runtime，经 ort 调用 | Rust 编写 DSP／应用调度，原生运行时执行网络 | 导出精度、算子内核、优化匹配、原生库分发 |

ONNX Runtime 的原生核心不满足“推理运行时也必须 Rust”的严格定义；通过 Rust API 调用它可以不依赖 Python，但不能因此称为纯 Rust 推理。ONNX 格式本身不与纯 Rust 路线冲突。

资料：[ONNX 算子规范](https://onnx.ai/onnx/operators/)、[ort](https://docs.rs/ort/latest/ort/)、[Burn ONNX 导入](https://github.com/tracel-ai/burn-onnx)、[tract](https://github.com/sonos/tract)。

## B02：ONNX Runtime 值得作为候选与性能对照

先验判断：它值得进入 CPU 对比，即使最终产品选择 Rust 原生运行时，也可在允许开发工具依赖的前提下作为独立执行结果与性能对照。PyTorch 的已固定参考管线仍负责定义目标行为，不能将导出后 ORT 输出自动当作正确真值。

其官方文档提供 [图优化](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)、[执行提供程序](https://onnxruntime.ai/docs/execution-providers/) 和按后端区分的 [算子内核表](https://github.com/microsoft/onnxruntime/blob/main/docs/OperatorKernels.md)。默认 CPU 后端与 oneDNN／OpenVINO 等选项不能视为同一配置，首次比较建议先用默认 CPU 后端。

对 VR，可尝试只导出幅度谱到掩码的网络，将多频带 DSP 留在 Rust。对 RoFormer，可尝试用实部／虚部浮点张量表示输入与掩码，将 STFT／ISTFT 留在 Rust。这样选择的假设是减少导出边界复杂度，不是断言 ONNX 不支持 STFT。

验证重点：Resize 坐标规则、空洞／分组卷积、LSTM 方向与权重排列、动态时间维、RMSNorm／RoPE 的表达及注意力优化是否匹配该网络。ONNX 规范存在某算子，不代表导出器和具体执行后端的全部属性组合都可用。

修订条件：如果导出图需要大量定制、峰值内存／性能不合适或原生分发不可接受，就不作为产品后端；有成熟 CPU 内核也不能事先承诺它比 Rust 后端快。

## B03：Burn 与 Candle 并列，而非后备

先验判断：Burn 的后端分离和模型导入值得与 Candle 同时评估。此前“先 Candle，不行再 Burn”的排序依据不足，撤销该排序；保留 Candle 为候选，不预设赢家。

当前官方 [Burn README](https://github.com/tracel-ai/burn) 列出了 Flex 与 CubeCL CPU 路线；[burn-flex 配置](https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/Cargo.toml) 使用 gemm，提供 SIMD 和 Rayon 特性。不能继续用旧版本的 NdArray 后端印象代表当前全部 CPU 路线。发布版本与主分支能力可能不同，实验必须注明实际可用版本。

纯 Rust CPU 的首次比较建议考察 Burn Flex；CubeCL CPU 的工具链、内核编译依赖和平台分发另行核对，不能仅因上层代码是 Rust 就推断所有依赖都符合约束。后端装饰器／融合能力也不能自动套用到每个后端。

查阅的 [burn-onnx 支持表](https://github.com/tracel-ai/burn-onnx/blob/main/SUPPORTED-ONNX-OPS.md) 列出 Conv2d、LSTM、Resize、Attention 支持，而 RMSNormalization、RotaryEmbedding 条目未支持。这个表描述 ONNX 算子映射，不能据此断言 Burn 无法实现 RMSNorm／RoPE。可以核对基础算子分解、手写 Rust 模块或导入扩展；最后需验证与指定权重的精确语义。

[burn-onnx](https://github.com/tracel-ai/burn-onnx) 提供生成 Rust 代码、权重产物和自定义算子／覆盖入口。导入支持、张量后端支持、数值等价和高性能是不同结论，需要分开实验。

## B04：不完善时先定位缺口，再决定手写层级

| 问题 | 优先尝试 | 不足以得出的结论 |
| --- | --- | --- |
| 缺一个上层网络模块 | 用已有张量算子组合实现 | 整个框架不可用 |
| ONNX 导入缺某节点 | 调整导出表达、基础算子分解或手写该节点 | 后端没有执行这些数学运算的能力 |
| 插值／归一化等语义不一致 | 写局部兼容实现并逐层对照 | 换一个相似算子就能复现权重 |
| 形状变换与临时张量占用过高 | 改布局、复用缓冲、减少拷贝和物化 | 只需换更快的 GEMM |
| 某个算子确为热点 | 专用内核、融合或高性能计算库 | 需要从零开发完整通用框架 |
| 多处关键路径都有不合适的抽象／调度 | 比较替代框架与专用执行器的总成本 | 手写路线天然更快或跨平台无成本 |

我们大概率需要自己实现模型结构与 UVR DSP 兼容逻辑；是否要自己写 SIMD 卷积、矩阵乘微内核，是另一层问题。

## B05：专用执行器是可验证的第三条工程路线

本项目只执行三种固定网络，不需要训练和通用图编辑，因此可以研究“固定网络执行代码 + 高性能原语”的实现方式。它需要承担权重映射、张量布局、临时内存规划、并行策略、平台兼容和数值验证；范围比通用框架小，但不能忽略维护成本。

可研究的专用计算方法：

- VR 卷积：根据真实形状比较分块 im2col + GEMM 与直接／depthwise 卷积；避免一次物化巨大 im2col。不能假设 FFT 卷积对小卷积核更快。
- LSTM：合并门计算，预先批量计算窗口内输入投影，保留递归状态计算；前后方向分别处理并恢复时间顺序。
- RoFormer：批量矩阵乘、减少频带布局拷贝；内存有压力时比较保持全局 Softmax 的分块注意力。RoPE／归一化／门控融合必须先证明是热点并对齐误差。
- DSP：复用 FFT plan 与 scratch buffer、重采样状态和窗口，按参考归一化语义限制整曲数据驻留。

严格 Rust 路线可先复用 RustFFT 与 Rust GEMM 等原语，再补专用模块；已有框架也可能使用同一 GEMM，因此手写的收益可能来自内存／融合／调度，而不是矩阵乘本身。

若允许原生计算库，[oneDNN](https://github.com/uxlfoundation/oneDNN) 提供深度学习计算原语，可作为卷积／矩阵乘／循环网络等计算的候选。Rust 调用这类库属于 Rust 应用加原生计算依赖，不能标为严格的全 Rust 计算路径。FFI 绑定、数据重排、线程池和 Windows 分发成本都需纳入比较。

修订条件：只有当框架方案确有已测量瓶颈或兼容缺口，且局部专用实现收益足以覆盖维护成本时，才扩大手写范围。

## B06：建议的最小比较实验

核心比较对象建议为 Candle CPU、Burn Flex；ONNX Runtime 默认 CPU 后端作为条件允许时的对照与产品候选；tract 做 ONNX Rust 路线的适配筛查。不要先把三个整模型在四个框架中全部移植。

1. 固定参考版本和权重，提取代表性输入／输出及真实张量形状。
2. 比较 VR 卷积／ASPP 小块、双向 LSTM 模块、RoFormer 的投影＋注意力＋前馈小块；仅矩阵乘基准不足以决定框架。
3. 测 FP32、batch=1、相同输入和线程预算下的冷／热时间、峰值内存及数值误差。检查转换与拷贝成本，避免比较不同工作量。
4. 筛选候选后只做一个整模型，验证片段结果和端到端开销；再决定主后端与局部专用算子。

完整评价标准还包括所需兼容代码量、Windows 构建／运行、依赖分发及后续维护成本。未做这些实验前，所有优先级和性能预期均保留为先验假设。
