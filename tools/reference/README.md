# 独立验证工具

运行时不依赖 Python；此目录仅用于开发期权重核验和参考样本生成，不参与 Cargo／pnpm workspace、应用构建或分离任务。Python 3.12、CPU PyTorch 和 NumPy 由 `pyproject.toml`／`uv.lock` 固定。

从仓库根目录重新生成合成 DSP 样本：

```sh
uv sync --frozen --project tools/reference --python 3.12
uv run --frozen --project tools/reference python -B tools/reference/generate_dsp.py
cargo test --locked -p uvr-core --test dsp_reference
```

生成器直接调用 PyTorch STFT／ISTFT。`core/tests/fixtures/dsp/manifest.json` 记录版本、尺寸、填充和 SHA-256；每个 `.f32` 文件按小端 FP32 保存输入、原频谱实虚对、修改后频谱实虚对、重建波形、修改频谱的重建波形。频谱按 `[frequency, time, real/imag]` 排列。普通 `cargo test` 直接读取这些固定样本，不启动 Python。重新生成后必须核对差异及重新运行 Rust 对照测试。

VR 权重核验需要先按 [目标来源清单](../../references/targets.json) 的 URL 下载完整权重和元数据。权重放 `models/`，元数据放清单的 `local` 路径；不要将未完成的 `.part` 文件重命名为最终文件。发布记录未提供整文件摘要，因此首次下载计算的 SHA-256 是本地身份记录，不是发布者签名。

```sh
uv run --frozen --project tools/reference python -B tools/reference/audit_vr.py --model 5_HP-Karaoke-UVR.pth --forward --output benchmarks/artifacts/vr-reference/5hp.json
```

脚本检查固定 upstream 提交、源码变更、元数据摘要、文件大小、UVR 标识、完整张量键／形状／有限值，使用 `weights_only=True` 加载并输出核验报告。`--forward` 用固定合成幅度谱运行掩码网络并记录算子形状；它不验证音频质量。网络仅依赖 `spec_utils.crop_center`，脚本从固定源码提取该函数，避免为纯张量验证导入音频编解码和 GUI。不会执行完整 `separate.py`，也不会修改参考源码。

1296 的独立参考源码按 [源码清单](../../references/roformer-source.json) 下载到各自 `local` 路径；该清单固定提交和每个文件的 SHA-256，脚本加载前逐项核对。还需要目标清单中固定的 YAML 配置。

```sh
uv run --frozen --project tools/reference python -B tools/reference/audit_roformer.py --threads 2 --forward --output benchmarks/artifacts/vr-reference/1296.json
```

默认用 4096 个合成采样验证前向，记录网络实际输出长度。它使用原始权重和发布配置，设备固定为 CPU；模型实现来自单独固定的参考版本，现已选作本项目的 [工程基线](../../docs/baseline.md)。构造函数为查询频点数运行一次矩形窗 STFT，可能产生 PyTorch 窗警告；真正音频前向使用参考代码的 Hann 窗。

实际网络的后端比较入口见 [CPU 后端探针](../../benchmarks/backend-probe/README.md)。`generate_vr_probe.py` 捕获卷积／上采样，`generate_sequence_probe.py` 捕获 LSTM／RoFormer；Rust 模块测量进程只读取数据文件。

`generate_vr_model.py --audit <核验报告> --output <目录>` 生成完整 HP Karaoke 或 DeEcho 掩码对照，包含普通幅度谱、静音、稀疏脉冲和三种窗口长度。它仅保存输入与期望输出；完整 Rust 网络直接读取原始 `.pth`，不要求 Python 转换权重。

## OpenVINO 开发期后端实验

`benchmark_vr_openvino.py` 将核验过的原始模型转换成固定尺寸的完整掩码图。它只用于筛选后端，产品仍从 Rust 加载原权重；Python 转换、编译和推理耗时分别报告。OpenVINO 按独立实验依赖固定版本，不改参考环境的锁文件：

```sh
uv run --frozen --project tools/reference --with openvino==2026.3.1 python -B tools/reference/benchmark_vr_openvino.py --audit benchmarks/artifacts/vr-reference/5hp.json --fixtures benchmarks/artifacts/backend-probe/5hp-model --output benchmarks/artifacts/backend-probe/5hp-openvino-t8.json --threads 8
```

输出文件必须不存在。默认选择 `magnitude`；可用 `--case` 指定清单中的其他输入。`--save-ir <路径.xml>` 可保存未压缩成 FP16 的 IR。CPU 使用 FP32、单 stream，记录实际线程／绑核／超线程设置及运行图的内核和精度。每组首次执行、额外预热、五次热运行均做完整掩码门限检查，失败保留输出并非零退出。该计时没有 DSP、默认 512 帧文件分离或 GPU 完整任务结果，不能直接作为产品后端验收。

1296 的模块探针取原始模型的第一层时间／频率注意力和前馈，读取同一份 Rust 模块清单，保留 801／62 序列长度：

```sh
uv run --frozen --project tools/reference --with openvino==2026.3.1 python -B tools/reference/benchmark_roformer_openvino.py --fixtures benchmarks/artifacts/backend-probe/sequence --output benchmarks/artifacts/backend-probe/1296-openvino-t8.json --threads 8
```

该工具先核对原模型输出，再转换固定 shape 并执行七次门限检查；报告每个模块的转换／编译耗时、全部热运行、实际设备设置与内核。`--device GPU` 选择核显，保持 FP32；`--threads` 在 GPU 模式下只控制原始 PyTorch 参考的 CPU 线程。`--compilation-threads` 单独控制 OpenVINO 图编译并发。模块结果不包含完整网络的 STFT／ISTFT、窗口调度或产品运行时集成。

完整单窗 PCM 实验覆盖所有 24 个时间／频率 Transformer、分频投影与掩码头。正向 STFT 使用基线的 float64，图内网络和 ISTFT 保持 FP32；原 PyTorch 包装先通过整轨门限，然后测转换图的首次执行、额外预热和五次热运行：

```sh
uv run --frozen --project tools/reference --with openvino==2026.3.1 python -B tools/reference/benchmark_roformer_openvino_audio.py --fixtures benchmarks/artifacts/backend-probe/1296-f64-real-audio --output benchmarks/artifacts/backend-probe/1296-openvino-pcm-t8.json --threads 8 --save-ir benchmarks/artifacts/backend-probe/1296-openvino-301.xml
```

`--device GPU` 和 `--compilation-threads` 与模块工具一致。`--ir <已有.xml>` 可复用转换图，仍重新核对原模型和全部波形；输入 shape 必须完全匹配，不能把固定短窗图直接用作完整 8 秒图。只接受 44.1 kHz、完整 hop、1323～352800 采样的单窗；计时包含 STFT、同步网络执行、复制、ISTFT 和伴奏残差，排除加载／转换／编译、文件编解码和验证。各阶段、完整波形误差、实际内核及进程资源写入新报告；失败保留 PCM 并非零退出。开发期 IR 的 Rust 执行入口见 [后端探针](../../benchmarks/backend-probe/README.md#openvino-rust-完整单窗实验)。

`verify_vr_cli.py --backend openvino-cpu --openvino-lib <原生库目录>` 可用相同音频门限检查 1296 的可选 CLI 后端，需先构建 `cargo build --release --locked -p uvr-cli --features openvino`。验证仍由 Rust CLI 直接加载原始 `.ckpt`；Python 只准备和比较 WAV。脚本同时检查拒绝覆盖，`--cancel-check` 检查 Ctrl-C 退出与输出清理，报告保存完整命令、实际子进程环境和源码／二进制摘要。

`profile_roformer_openvino.py --ir <图.xml> --fixtures <样本目录> --output <新报告.json>` 记录 CPU 编译前后内存、两次逐算子耗时、执行内核及完整波形门限，并导出 `.runtime.xml` 执行图。使用上述固定 OpenVINO 环境运行；这两次带 profiling 的执行用于定位热点，不作为五次热运行性能结果。

`run_roformer_openvino_probe.py --ir <图.xml> --fixtures <样本目录> --output <新报告.json>` 用标准 Python 调度 Rust 探针，保存完整命令、版本摘要、原生进程 CPU 时间／RSS 和系统可用内存的一秒采样。默认首次、额外预热加五次热运行；`--check-only` 只做一次正确性检查。`--threads`、`--cpu-pinning YES|NO`、`--hyper-threading YES|NO`、`--core-type ANY_CORE|PCORE_ONLY|ECORE_ONLY` 用于单因素 CPU 调度实验。内存保护检查实际系统余量，默认剩余不足 1.5 GiB 时停止该实验；它不是产品 RSS 上限，可按机器和前台应用余量用 `--min-available-gib` 调整。`--device GPU --inorder-diagnostic` 仅启用已安装的 OpenCL 顺序队列诊断层，仍需另行验证受支持的产品 GPU 配置。

## VR 多频带 DSP

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_dsp.py
cargo test --locked -p uvr-core --test vr_dsp_reference
```

参考固定 librosa 0.9.2、SciPy 1.16.3 和现有 NumPy 2.2.6，从固定 `spec_utils.py` 提取原函数；不导入完整 UVR GUI 或修改子模块。librosa 的旧 dtype 表需要已移除的 NumPy Python 类型别名，生成器仅在自身进程恢复这些别名；setuptools 80.9.0 提供其旧资源接口。依赖由 uv.lock 固定，普通 Rust 测试读取 2.1 MiB 合成样本，无需 Python。

`core/tests/fixtures/vr-dsp/manifest.json` 记录 14 组重采样、三套模型各 6 组多频带样本的参数、输入／频谱／掩码／双轨波形摘要。明确采用工程基线的 polyphase 重建、完整 hop 补零及尾部裁剪；未覆盖频点填零。另将这些频点分别填 0／1，证明上游未初始化分配会改变结果；静音归一化除零也保存复现结论。生成器不以 Rust 输出反写黄金数据。

## 文件编解码与完整音频

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_audio_io.py
cargo test --locked -p uvr-core --features audio-io --test audio_io_reference
uv run --frozen --project tools/reference python -B tools/reference/generate_vr_audio.py --variant 5hp --output benchmarks/artifacts/backend-probe/5hp-audio
```

编解码样本由 SoundFile 0.14.0／libsndfile 1.2.2 独立生成并解码，覆盖 WAV 的 8／16／24／32 位整数及浮点、FLAC、MP3 单／双声道和 gapless 长度。样本约 232 KiB，都是自制合成音，黄金 PCM 与文件分别保存摘要；另检查非有限数、空音频和多声道拒绝。Rust 测试覆盖取消时临时输出清理、已有文件保护、发布竞态、浮点增益与截断音频检测。

`generate_vr_audio.py` 默认生成六组完整 PCM 参考，使用已核验网络和上述 DSP 基线，窗口为 512。使用 `--audio-file <本地音频>` 可改为单个真实片段，默认取 60 秒处的 3 秒，范围由 `--start-seconds`／`--duration-seconds` 指定。真实音频、原始文件摘要和片段参数仅写入被忽略的实验目录，不提交音频内容。

编译 CLI 后，可以将固定 PCM 样本写成 WAV，再验证整个命令行链和 Ctrl-C；此时被验证的 Rust 子进程仅收到 `PATH=/nonexistent` 与两线程环境，不调用 Python：

```sh
cargo build --release --locked -p uvr-cli
uv run --frozen --project tools/reference python -B tools/reference/verify_vr_cli.py --fixtures benchmarks/artifacts/backend-probe/5hp-audio --case signal --output benchmarks/artifacts/cli-5hp --cancel-check
```

验证输出目录必须不存在；脚本检查两轨 WAV 的形状和基线误差，再检查重复执行拒绝覆盖。取消测试等待进入推理后发出 SIGINT，要求退出码 130 且没有正式输出。它不构成五次热运行性能测量或听评。

完整 PCM 性能对照使用 `benchmark_vr_audio_reference.py --fixtures <音频清单目录> --output <新报告.json> --case <样本名> --threads 2`。它复用严格加载的原始模型，逐次验证两轨误差，保留首次执行、额外预热和五次热运行的完整分离／分阶段／每窗耗时及 RTF。与 Rust 的 `probe-vr-audio --measure` 顺序运行，命令和计时边界见 [探针说明](../../benchmarks/backend-probe/README.md#完整-pcm-音频)。该比较排除解码、编码和模型加载，不能替代文件任务或整曲处理链的性能验收。

## 完整 1296 网络与音频调度

```sh
uv run --frozen --project tools/reference python -B tools/reference/generate_roformer_model.py --stft64 --output benchmarks/artifacts/backend-probe/1296-f64-model
uv run --frozen --project tools/reference python -B tools/reference/generate_roformer_audio.py --stft64 --output benchmarks/artifacts/backend-probe/1296-f64-audio
```

网络生成器直接使用固定原始模型，覆盖普通短窗口、最短反射输入、静音；`--full-window` 生成完整 352800 采样发布窗口，`--audio-file <文件>` 生成 60 秒处的 8 秒真实窗口。原始网络输出仍为输入长度向下取整到 hop，不能代替完整文件接口的长度保证。

工程基线 v1.1 要求上述两个生成器使用 `--stft64`：固定原始网络外包裹正向 STFT 的 float64 计算，结果转 complex64，ISTFT 的窗系数来自同一个 float64 Hann 窗。模型层、权重和验收门限保持原定义。省略该参数只用于重现旧 FP32 STFT 诊断，不应拿旧黄金输出验证新精度预设。`generate_roformer_dsp.py` 直接调用 PyTorch float64 STFT，产生普通 Rust 测试读取的 644 KiB 自制样本；不依赖完整模型加载。

音频生成器先按工程基线复制单声道、polyphase 重采样并补齐 hop，再使用固定 `mdxc_separator.py` 的原始尾窗调度与交叠累加函数、SciPy 的 Hamming 窗和完整原始网络，最后裁回长度、计算伴奏残差。它默认覆盖四组短音频／边界；`--boundary` 检查两个 8 秒窗口的尾部交叠，`--audio-file` 使用真实片段（默认 3 秒，可用 `--duration-seconds` 设置）。所有完整权重／真实音频输出留在被忽略的实验目录。Rust 对照入口见 [探针说明](../../benchmarks/backend-probe/README.md)。
