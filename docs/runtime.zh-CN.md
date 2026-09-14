# 推理运行时与参数调优

[English](runtime.md) | 简体中文 | [日本語](runtime.ja.md)

桌面应用与 CLI 使用同一套 Rust 推理参数和校验规则，无需转换原始 UVR 权重即可调整 CPU 占用与内存使用。建议从推荐配置开始；增加批次或并行窗口会提高内存需求，不一定让模型更快。

## 推荐配置

| 模型 | 运行时 | 当前默认参数 |
| --- | --- | --- |
| 5-HP / 6-HP | Burn CPU，使用共享 VR 优化内核 | 512 帧、推理批次 1、窗口并发 4 |
| DeEcho | Burn CPU | 512 帧、推理批次 1、窗口并发 1 |
| BS-RoFormer 1296 | 构建支持且可用时选择 OpenVINO CPU，否则使用 Burn CPU | OpenVINO：FP32、延迟模式、单 stream；Burn：时间批次 62、频率批次 301、flattened 线性层布局、窗口并发 1 |

CPU 线程数默认为可用 CPU 数与 8 的较小值。配置依据本项目在 Intel i5-13420H 上已有的测量，适合作为调优起点，不代表所有机器的最快点。切换运行时保持原始权重、输出增益、FP32 网络精度和既定音频处理约定。

OpenVINO 产品入口目前只支持 1296 的 CPU 推理；实验中的 GPU 与其他后端不作为可用选项。应用确认 OpenVINO CPU 原生运行时可用后才推荐它，否则推荐 Burn；显式指定不可用的后端会返回错误。

### 尚未完成的调优

DeEcho 的批次与窗口并发调优尚未完成。当前实现强制 `inference_batch=1`、`window_parallelism=1`；窗口长度仍可调整，默认 512 帧，合法范围为 144–2048 的 16 倍数。双向 LSTM 要求每窗状态独立，未来上下文也使跨窗状态复用不能直接视为等价，但这不排除独立窗口并发。固定为 1 是当前实现的保守限制：DeEcho 并发 1／2／4 的完整音频对照，包括状态隔离和输出检查，尚未完成。HP 的实测不能证明 DeEcho 提高并发一定更慢。

BS-RoFormer 1296 的 Burn 与 OpenVINO 路径也尚未完成调优。Burn 的布局和注意力批次调整已有计时与波形检查；OpenVINO CPU 已有重复短音频对照，并接入 GUI／CLI。这些都是阶段性结果，完整窗口重复测量、近期布局改动的交错前后对照，以及整曲和多模型处理链性能验收仍待完成。当前默认值是调优起点，不代表已找到最快配置。待办与通过条件见[性能协议](performance.md#尚未完成的调优与验收)。

## 桌面设置

任务表单的「推理运行时」提供后端与计算线程选项，「高级推理参数」按当前模型展开。配置摘要和任务记录显示本次任务实际使用的选项。

- 切换模型时，分别保留各 VR 模型的窗口、批次设置，以及 1296 的后端和 Burn 参数。
- DeEcho 固定批次 1、窗口并发 1；HP 的批次大于 1 时按批处理，独立窗口并发控件禁用。
- OpenVINO 使用自身调度，选中后隐藏 Burn 专属批次和布局控件。
- 设置保存在本地；「恢复默认」恢复当前模型推荐参数及默认线程数。
- 任务期间锁定控件；任务结束或取消后可重新设置，下一任务使用新的独立线程池，无需重启应用。

语言菜单支持中文、日文和英文；切换时保留路径与推理参数，并即时更新正在运行的任务状态。「外观」提供跟随系统、浅色、深色，以及紫罗兰、海蓝、青绿主题。语言与外观偏好独立保存。

## CLI 参数

选项放在必需的位置参数之后，顺序不限。枚举使用下表的 ASCII 名称，数字使用十进制正整数。未知或重复选项、缺少参数值、参数不适用于当前模型或后端时，在文件任务开始前返回用法错误，退出码 2。

语言选项放在命令之前：`uvr --lang zh-CN --help`、`uvr --lang ja --help` 或 `uvr --lang en --help`。`UVR_LANG` 设置 CLI 默认语言，`--lang` 优先。供脚本读取的输出键、文件名和参数值不随语言改变。

| CLI 选项 | 适用范围 | 默认值与限制 |
| --- | --- | --- |
| `--threads N` | 所有模型与运行时 | 可用 CPU 数与 8 的较小值；正整数 |
| `--backend burn\|openvino-cpu` | `separate-1296` | 可用时 OpenVINO CPU，否则 Burn |
| `--window-frames N` | `separate-vr` | 512；16 的倍数；HP 为 272–2048，DeEcho 为 144–2048 |
| `--inference-batch N` | `separate-vr` | 1；范围 1–4；DeEcho 实际固定为 1 |
| `--parallel-windows N` | VR，或 Burn 1296 | 范围 1–8；HP 默认 4，DeEcho 和 1296 默认 1；HP 批次 >1 时实际并发为 1 |
| `--time-batch N` | Burn 1296 | 62；正整数 |
| `--frequency-batch N` | Burn 1296 | 301；正整数 |
| `--linear-layout flattened\|batched` | Burn 1296 | `flattened` |

RoFormer 批次合并独立注意力序列，不缩短序列本身；超过实际频带或帧数的批次自然受输入形状限制。调整 VR 窗口大小可能改变分离结果，与已有参考样本比较时应使用默认窗口。

```sh
# pnpm build:native 构建下面的本机优化程序。
target/native/release/uvr separate-vr 5hp \
  models/5_HP-Karaoke-UVR.pth input.wav outputs/5hp \
  --threads 8 --window-frames 512 --inference-batch 1 --parallel-windows 4

target/native/release/uvr separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296 \
  --backend openvino-cpu --threads 8

target/native/release/uvr separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296-burn \
  --backend burn --threads 8 --time-batch 62 --frequency-batch 301 \
  --parallel-windows 1 --linear-layout flattened
```

CLI 会报告实际生效配置，包括 DeEcho 和 HP 批处理的并发调整。已有同名音轨不覆盖。Ctrl-C 请求取消，退出码为 130；VR 等待当前窗口或并发组结束，串行 Burn 1296 支持窗内检查，OpenVINO 支持原生请求取消。

### 兼容环境变量

CLI 的优先级为：显式选项 > 对应环境变量 > 默认值。已被选项覆盖的环境变量不再解析；OpenVINO 忽略 Burn 专属环境变量。

| 环境变量 | 对应 CLI 选项 |
| --- | --- |
| `RAYON_NUM_THREADS` | `--threads` |
| `UVR_ROFORMER_TIME_BATCH` | `--time-batch` |
| `UVR_ROFORMER_FREQUENCY_BATCH` | `--frequency-batch` |
| `UVR_ROFORMER_WINDOW_PARALLELISM` | Burn 1296 的 `--parallel-windows` |
| `UVR_LINEAR_LAYOUT` | `--linear-layout` |

GUI 使用有效的 `RAYON_NUM_THREADS` 作为初始线程设置，已保存或在界面中编辑的数值优先。其他推理参数来自共享默认值和可见控件，不会被隐藏的 `UVR_*` 环境变量覆盖。

## 构建本机优化程序

本机配置使用 release 优化和 `-C target-cpu=native`，接入已保留的共享优化内核。CLI 与 GUI 一起构建到独立目录，便于区别于通用 CPU 构建。

```sh
pnpm install --frozen-lockfile
make
target/native/release/uvr-gui
```

该构建面向 Linux x86_64，默认从 `.local/openvino-2026.3.1/lib/` 读取 OpenVINO 原生库，也可通过 `UVR_OPENVINO_LIB_DIR` 指定目录。目录须包含 `libopenvino_c.so`、CPU 插件及其依赖。脚本把所需 CPU 库复制到 `target/native/release/lib/`，两个程序均直接加载；移动程序时保留旁边的 `lib/`。运行产物不需要 Python 进程、转换脚本或激活开发环境。

```sh
UVR_OPENVINO_LIB_DIR=/path/to/openvino/lib pnpm build:native

# 尚未安装 OpenVINO 时，构建本机优化的 Burn 版本。
pnpm build:native --burn-only
target/native-burn/release/uvr-gui
```

Burn-only 产物位于 `target/native-burn/release/`，与 `target/native/release/` 分开，避免之前 OpenVINO 构建留下的可选库混入分发目录。每种构建均在各自 `release/build-info.json` 中记录可执行文件体积、SHA-256 与构建设置。

本次启用 OpenVINO 的 native 构建中，CLI 为 6.94 MB、GUI 为 18.91 MB，可选 OpenVINO 库目录另加 86.80 MB；GUI 与这些库合计 105.71 MB，不含模型权重和系统库。准确字节数、摘要及历史构建见[体积证据](performance-summary.zh-CN.md)。

`make` 是本机构建的简短入口，底层调用 `pnpm build:native`；`make native-burn` 生成独立的 Burn-only 产物。本机产物使用构建时 CPU 的指令集；面向其他 CPU 时，使用普通 `pnpm gui:build` / `cargo build --release --locked -p uvr-cli` 生成通用构建。两种方式都不捆绑模型权重。

交叉编译 Windows 本身不会导致推理变慢，性能取决于 release 优化、目标 CPU 指令集、工具链和 Windows 调度。当前 native 脚本仅支持 Linux；由 Windows 用户本机构建并完成实际推理验证，是发布前核验该平台的可行分工。面向其他电脑的发布包不要直接套用开发者的 `target-cpu=native` 配置。

## 性能证据与维护

HP 默认配置有完整 5-HP、6-HP 音频实验支撑；Burn 1296 的布局与批次配置有波形检查和耗时记录；OpenVINO 有完整短音频波形对照、重复 CPU 测量和直接从原始 checkpoint 构图的 Rust 路径。这些结果不能推导出对 UVR 的普遍速度优势、模型质量提升，或整曲和多模型处理链已经完成验收。

详细记录见 [VR CPU 实验](../benchmarks/2026-09-11-vr-cpu-optimization.md)、[Burn 1296 实验](../benchmarks/2026-09-11-roformer-cpu-optimization.md)、[OpenVINO 实验](../benchmarks/2026-09-11-roformer-openvino.md) 与 [性能协议](performance.md)。共享参数契约和维护检查见[贡献指南](../CONTRIBUTING.zh-CN.md)。
