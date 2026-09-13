# 后验知识与验证记录

更新时间：2026-09-11。源码检查只能证明参考实现的行为；没有运行模型就不能宣称音质、速度或导出可行性。固定参考版本见 [references.md](references.md)。

## K001：初始化前目录状态

根 Cargo.toml 包含空的 `author=`，成员写为不存在的 `cli/`；`core/` 当时声明名为 `cli` 的 Hello World 二进制，根 `src/` 也只有 Hello World。GUI 只有方向说明。父仓库尚无提交，两个参考目录已有独立 Git 历史。

处理：core 改为共享库，CLI 放入 `cli/`，GUI 宿主放入 `gui/src-tauri/`。原上游用途说明移到 [upstream-notes.md](upstream-notes.md)，避免为了项目注释修改第三方源码。

## K002：VR 至少存在两个网络变体

本地 [separate.py](../upstream/separate.py) 的 `SeperateVR.seperate` 按权重大小和元数据选择 `nets.CascadedASPPNet` 或 `nets_new.CascadedNet`。`vr_denoiser(is_deverber=True)` 使用新版网络，`nout=64`、`nout_lstm=128`，配置为 `4band_v3`。新版 [layers_new.py](../upstream/lib_v5/vr_network/layers_new.py) 的 `LSTMModule` 包含双向 LSTM。

旧版网络 `offset=128`，新版 `offset=64`。VR 的参考前后处理涉及多频带重采样、频谱拼接、掩码调整、原相位重建及可选 TTA／高频 mirroring。依据：[网络目录](../upstream/lib_v5/vr_network/)、[spec_utils.py](../upstream/lib_v5/spec_utils.py)。

限制：尚未成功读取本项目所需的全部权重并核验哈希，5-HP 与 6-HP 的各自预处理参数不能仅凭文件名最终确定；不能把所有 VR 模型视为同一个网络换权重。

## K003：1296 的发布配置与本地参考缺口

已查阅的 [1296 配置](https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_368_sdr_12.9628.yaml) 指定 44.1 kHz、双声道、STFT 2048、hop 441、dim 512、depth 12、heads 8、单目标 Vocals；配置分块为 352800 个采样，重叠数为 4。伴奏重建需与实际参考程序的残差计算和增益处理对齐。

已查阅的 [BS-RoFormer 实现](https://raw.githubusercontent.com/nomadkaraoke/python-audio-separator/main/audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py) 包含分频带投影、RMSNorm、RoPE、时间／频带注意力及复数掩码估计。本地固定版本 `upstream/` 未提供该网络。

限制：这两条远程链接是可变分支内容，尚未固定为实验参考；没有运行 1296、转换权重或验证纯 Rust 算子。开始实现前必须另行锁定代码与配置，不可将当前网络代码默认视为客户使用的版本。

## K004：初始化验证

环境：Linux，rustc 1.98.1、cargo 1.98.1、Node.js 24.20.0、pnpm 12.1.0；pkg-config 检测到 GTK 3.24.52 与 WebKitGTK 2.52.6。

已通过 `cargo fmt --all -- --check`、`cargo check --workspace --all-targets --locked`、`cargo clippy --workspace --all-targets --locked -- -D warnings`、`pnpm build`。CLI 无参数、帮助、版本返回成功；未知命令和多余参数返回退出码 2。Cargo metadata 验证三个 Rust 成员，文档本地链接检查通过。

首次 Tauri 编译因缺少默认图标失败；加入可再生成的 SVG／PNG 占位图标及显式配置后通过。`pnpm --filter @uvr/gui tauri build --debug --no-bundle` 成功调用前端构建并生成 `target/debug/uvr-gui`。本次未启动桌面窗口做交互验证，也未生成安装包。

`git submodule status` 与暂存 gitlink 对应上述固定提交，两个参考工作树均干净；暂存检查确认无 node_modules、target、dist 或模型权重混入。未测量音频推理性能，未验证 Windows／macOS，未测试模型音质。

## K005：权重身份核验基线

日期：2026-09-09。项目基线 `b5418ed` 加本轮未提交改动；参考仍为 `upstream` 的 `5517e0cf0d1acd16a1618eeedec596957523f9e1`。

该版本 `UVR.py:get_model_hash` 对末尾 `10000 * 1024` 字节计算 MD5，文件较短时改用整文件。它是查询 `models/VR_Models/model_data/model_data.json` 的键，不能替代实验要求的整文件 SHA-256。

`uvr-core::weights::fingerprint` 与 `uvr inspect-weights <文件>` 用 64 KiB 缓冲在一次读取中计算两种摘要和字节数，不加载 checkpoint。测试覆盖空文件、`abc` 标准摘要、截取边界前／恰好／后一字节、修改非摘要前缀与摘要后缀、长度变化、读取错误，以及 CLI 的路径空格、文件缺失、目录输入和退出码。边界黄金值由 Node.js `node:crypto` 独立生成。

本地已有的 **UVR-DeNoise-Lite.pth（非四套目标模型，仅作核验样本）** 实测：

```text
size_bytes: 17922277
sha256: 0023492fe98c406817b5253965de19ede65d1c147db015a3a428f07602e99571
uvr_md5: 51ea8c43a6928ed3c10ef5cb2707d57b
```

复现命令：`cargo run --locked -p uvr-cli -- inspect-weights upstream/models/VR_Models/UVR-DeNoise-Lite.pth`。SHA-256 与系统 `sha256sum` 一致；MD5 与 Node.js 对相同后缀的摘要一致，并匹配固定元数据中的 `1band_sr44100_hl1024`、`Noise`、`nout=16`、`nout_lstm=128` 条目。

检查通过：`cargo test --locked -p uvr-core -p uvr-cli`（3 项核心测试、1 项 CLI 集成测试）、`cargo fmt --all -- --check`、`cargo check --workspace --all-targets --locked`、`cargo clippy --workspace --all-targets --locked -- -D warnings`、`pnpm build`、`git diff --check`。锁文件保留原有依赖版本，仅增加 MD5 包及 core 的摘要依赖关系。

限制：只证明文件身份算法及该样本的元数据命中；没有核验四套目标权重来源、分发许可、张量形状或网络输出。检查期间要求文件保持不变；长度与修改时间检查不构成原子快照。没有选择推理后端，也没有推理性能或 Windows 运行结论。

## K006：Rust STFT／ISTFT 数值基线

日期：2026-09-10。项目基线 `b5418ed` 加本轮未提交改动；RustFFT 6.4.1。`core::dsp::Stft` 实现单声道 FP32、偶数 FFT 尺寸、周期 Hann 窗、中心 constant／reflect padding、包含 Nyquist 的单边复数频谱、窗能量归一化和指定／默认长度重建。FFT plan、窗和工作缓冲可复用。声道调度、重采样及完整模型管线尚未实现。

独立参考环境固定 Python 3.12、PyTorch `2.13.0+cpu`、NumPy `2.2.6`，锁文件位于 `tools/reference/uv.lock`。生成器直接调用 `torch.stft`／`torch.istft`；[8 组样本清单](../core/tests/fixtures/dsp/manifest.json) 保存具体尺寸与文件 SHA-256。覆盖静音、首尾脉冲、短音频、非整数 hop 长度、非 2 次幂 FFT、两种 padding，以及复数掩码改变频谱后的重建。不会只靠前向／逆向相互抵消的往返测试判断正确性。

`cargo test --locked -p uvr-core -p uvr-cli` 共 6 项测试通过。DSP 对照逐元素检查 `abs(error) <= absolute + relative * abs(reference)`：频谱阈值为 `8e-5 + 2e-5 * abs(reference)`，重建阈值为 `2e-6 + 2e-5 * abs(reference)`；它们是当前合成样本的回归阈值，不是已确认的整模型音质验收标准。畸形尺寸、非有限数、反射填充过短输入的失败场景也通过。完整 workspace 的 Clippy 检查已通过。

运行边界：普通 Cargo 测试只读取固定 `.f32` 样本，不启动 Python。CLI 的 `ldd` 结果仅包含系统 C／GCC 运行库；应用源码与构建配置未引用 Python、PyO3 或验证工具目录。Python 仅用于用户允许的独立验证，未成为应用构建或运行依赖。生成方法见 [验证工具说明](../tools/reference/README.md)。

限制：当前结论只针对该 PyTorch 版本与样本；未证明全部 UVR／librosa 预处理等价、真实音频分离质量、推理性能或 Windows 行为。

## K007：目标来源固定与 VR 验证工具

日期：2026-09-10。[目标来源清单](../references/targets.json) 记录四套指定权重的 GitHub release asset ID、字节数、下载 URL，以及 `TRvlvr/application_data` 提交 `3826b05b570dbd4fbedbc807758803b35348ba1b` 下的模型清单、VR 元数据和 1296 配置 SHA-256。该发布记录的权重 digest 字段为空；来源固定不等同于已获得发布者整文件校验和。

`tools/reference/audit_vr.py` 将整文件身份、UVR 后缀 MD5、元数据、配置和严格网络加载结果关联，并可执行合成幅度谱前向推理。固定参考仍为 `upstream` 的 `5517e0cf0d1acd16a1618eeedec596957523f9e1`。这属于开发期验证工具；不会被产品调用。

目标权重的下载曾遇到 HTTP/2 重置及 TLS 断连，断点续传后四套文件均已下载完成，大小匹配发布记录。核验结果见 K008／K009；权重分发条件仍待确认，不据下载清单将其标为获准分发。

## K008：三套目标 VR 权重严格匹配参考网络

日期：2026-09-10。使用 K007 的固定来源、PyTorch `2.13.0+cpu`、CPU FP32、2 线程。逐个执行 `tools/reference/audit_vr.py --model <文件> --threads 2 --forward --output <报告>`，所有张量键与形状严格匹配，张量与输出均为有限值；完整身份记录见 [已核验权重](../references/verified-weights.json)。

| 权重 | 配置 | 网络 | 张量数 | 合成输入 → 掩码 |
| --- | --- | --- | --- | --- |
| 5-HP | `4band_v2_sn` | CascadedASPPNet | 459 | `[1,2,673,320]` → `[1,2,673,64]` |
| 6-HP | `3band_44100_msb2` | CascadedASPPNet | 459 | `[1,2,641,320]` → `[1,2,641,64]` |
| DeEcho-DeReverb | `4band_v3` | CascadedNet | 689 | `[1,2,673,320]` → `[1,2,673,192]` |

5-HP、6-HP 的元数据主输出均为 `Instrumental`、`is_karaoke=true`；DeEcho 为 `No Reverb`。旧版 offset=128、新版 offset=64 与执行结果一致。5-HP 的 SHA-256／UVR 标识还与 Rust CLI 输出交叉核对一致。

原始报告：`benchmarks/artifacts/vr-reference/{5hp,6hp,deecho}.json`，含每个张量与算子形状。适用范围：证明这三套实际权重可加载对应参考网络并运行合成幅度谱；未完成 Rust 网络移植、多频带音频管线、主唱／和声听评或分发条件核验。

## K009：1296 权重与固定 BS-RoFormer 参考匹配

日期：2026-09-10。参考为 `nomadkaraoke/python-audio-separator` 提交 `bf1164aa0f1ee1d1d0ef0f09b315f7659fc06bab`；源码与配置摘要见 [源码清单](../references/roformer-source.json) 和 [目标清单](../references/targets.json)。PyTorch `2.13.0+cpu`，FP32，CPU 2 线程。

实际 `model_bs_roformer_ep_368_sdr_12.9628.ckpt` 大小 639317465 字节，SHA-256 为 `f6c94864adfb73bbb0ca58ec14d58dd0b364549e9fb61433ae51916f3e2f8d0b`。699 个张量全部严格加载成功、值有限。`audit_roformer.py --threads 2 --forward` 用 `[1,2,4096]` 合成音频运行原网络，得到有限输出 `[1,2,3969]`。

该参考的 ISTFT 未传入原始长度，4096 个输入采样按 hop=441 默认重建为 `floor(4096/441)*441=3969`。Rust 调度器需要显式处理分块／填充和裁剪，不能假定网络总是保持任意输入长度。发布 chunk_size=352800 是 441 的整数倍。

报告位于 `benchmarks/artifacts/vr-reference/1296.json`。该结果证明权重／配置／选定网络匹配及短合成前向可执行；客户参考版本尚未确定，没有验证真实音频质量、Rust RoFormer、整曲残差与归一化或实时性能。

## K010：首轮 Candle／Burn Flex CPU 模块对照

日期：2026-09-10。两个独立 Rust 探针均通过实际 5-HP 中间激活的卷积、空洞 depthwise／pointwise、BatchNorm、激活函数及双线性上采样检查。固定版本、输入摘要、全部样本位置、误差与计时范围见 [实验记录](../benchmarks/2026-09-10-vr-cpu-blocks.md)。

本轮在所测卷积块上 Burn Flex 更快，在所测双线性上采样上 Candle 更快。两者继续保留为候选；LSTM、RoFormer 与整模型尚待对照，不能以本轮选择最终后端或承诺实时性能。实验框架留在独立 workspace，未加入产品依赖。`cargo clippy --manifest-path benchmarks/backend-probe/Cargo.toml --all-targets --locked -- -D warnings` 与 release 构建通过。

## K011：DeEcho LSTM 与 1296 关键模块已有 Rust 对照

日期：2026-09-10。两套 Rust 实现通过全部 8 个新样本，包括三种实际 DeEcho 双向 LSTM 形状、短序列双批次、1296 时间／频率注意力、零输入和前馈。权重、参考、误差、计时边界与全部原始样本位置见 [序列模块实验](../benchmarks/2026-09-10-sequence-cpu-blocks.md)。

结合 K010 的卷积结果，选择 Burn Flex 做首个完整 HP Karaoke 网络原型。其 LSTM、时间注意力和前馈在本轮更快；Candle 的频率注意力更快，不能把这个选择扩展成所有模型的性能排名。主后端与分发方案还需整模型、音频链和目标平台验证。

## K012：5-HP／6-HP 完整 Rust 掩码网络通过参考对照

日期：2026-09-10。`core/src/vr.rs` 的可选 `burn-cpu` 实现直接加载原始权重，校验每套 459 个张量，执行完整 CascadedASPPNet。两套权重各有 320 帧普通幅度谱、272 帧静音和 304 帧脉冲对照，所有重复执行均通过逐元素阈值；最大绝对误差分别为 `2.4617e-5`、`2.8610e-5`。验证进程在 `PATH=/nonexistent` 下运行，模型载入与前向没有调用 Python。

完整测量、误差、构建／数据摘要及复现命令见 [HP CPU 网络记录](../benchmarks/2026-09-10-hp-cpu-network.md)。这推进了 P2 的旧版 VR 网络基线；输入仍是准备好的多频带幅度谱，尚未接通音频处理链。DeEcho／RoFormer 全网络、端到端音频质量、目标平台和性能验收均未完成。

验证：启用 `uvr-core/burn-cpu` 的 core／CLI 测试、workspace 全目标检查、core 与探针 Clippy、两套权重的完整网络 release 对照、`pnpm build`、格式检查通过。CLI 帮助文案修订后重跑命令行集成测试通过；参考子模块保持无改动。

## K013：DeEcho 完整 Rust 网络通过参考对照

日期：2026-09-10。`core/src/vr/deecho.rs` 严格加载原始权重的 689 个张量，完整网络的普通幅度谱、静音、脉冲全部重复执行通过原有掩码门限。最大绝对误差分别为 `2.8014e-6`、`2.6453e-4`、`8.0246e-4`，后两项明显更大，尚未定位传播来源。记录与原始产物位置见 [DeEcho CPU 网络实验](../benchmarks/2026-09-10-deecho-cpu-network.md)。

执行时 `PATH=/nonexistent`，加载与推理不调用 Python。core／探针 Clippy 和 release 构建通过；共享卷积与窗口接口调整后，两套 HP 各三组 `--check-only` 回归通过。此结论仅覆盖完整掩码，仍需音频 DSP、波形误差与听评，RoFormer 完整 Rust 网络也未完成。

## K014：项目自行固定工程基线

日期：2026-09-10。用户说明客户没有提供基线并授权本项目决定，因此固定 [工程基线 v1](baseline.md) 的参考提交、CPU FP32 配置、音频预设、输出约定与数值／性能预算。它是后续实验的验收约定，不能当作已通过的实验结果；不再把客户版本或参数缺失列为实现阻塞。

## K015：三套 VR 多频带 DSP 与重采样通过独立对照

日期：2026-09-10。`core/src/resample.rs` 实现零相位、零边界、Kaiser β=5 的 polyphase FIR；`core/src/vr_dsp.rs` 实现三套配置的采样率级联、声道变换、STFT、频带拼接／预滤波、掩码补集及两轨重建。输入补齐最高频带的完整 hop，输出裁回重采样后的精确长度。HP-6 使用全局 mid_side_b2；HP-5 的旧网络分支不启用配置中的逐频带 stereo_n，与固定参考分支一致。

独立生成器直接调用 SciPy 1.16.3、librosa 0.9.2 及固定 UVR 的 DSP 函数。14 组重采样和三套模型各 6 组音频样本全部通过原定 DSP 门限；覆盖单采样、短单声道、48 kHz、静音、首尾脉冲与普通双声道。`cargo test --locked -p uvr-core --test vr_dsp_reference` 的参考与错误输入测试通过。每个文件、配置及参考摘要见 [样本清单](../core/tests/fixtures/vr-dsp/manifest.json)。

参考差异实验已执行：将上游 `np.ndarray` 未覆盖频点分别填为 0 与 1，零输入的重建波形最大差异为 HP-5／DeEcho 的 0.00208333、HP-6 的 0.01507911；说明未初始化分配不能提供稳定基线。本项目将这些频点填零，并统一 polyphase 重采样。另复现全零幅度直接除以最大值会产生 NaN；音频接口在此处跳过网络并返回精确全零输出。

当前结果证明 DSP 合成对照，尚不证明完整网络波形、音频解码／编码、模型效果或整曲性能。完整 PCM 音频接口已接通，真实权重的波形对照继续进行。

## K016：三套 VR 完整 PCM 音频链通过独立对照

日期：2026-09-10。三套原始权重各六组完整 PCM 波形对照全部通过预先固定的逐样本与每轨 RMS 门限，覆盖静音、脉冲、短单声道、重采样和窗口交界。两轨形状精确、值有限，静音严格为零，进度／取消检查通过。最大绝对误差分别为 5-HP 的 `2.3842e-7`、6-HP 的 `2.6822e-7`、DeEcho 的 `8.8290e-7`。参考、源码、数据摘要及复现方法见 [完整 PCM 记录](../benchmarks/2026-09-10-vr-cpu-audio.md)。这是单次正确性验证，未得出性能或听感结论。

## K017：VR CLI 文件处理与真实音频对照通过

日期：2026-09-11。纯 Rust 解码、原始模型加载、完整分离及 WAV 写入已在三套 VR 的相同本地 3 秒真实片段上通过独立对照。编解码另覆盖 8 组自制 WAV／FLAC／MP3 黄金样本。运行环境不提供 Python 可执行路径；中文路径、重复执行拒绝覆盖、取消退出及输出清理均经过检查。具体误差、文件摘要与复现入口见 [VR CLI 记录](../benchmarks/2026-09-11-vr-cli-audio.md)。

限制：5-HP 在网络中收到取消后约 15.161 秒停止，仍受当前窗口计算约束；没有完整处理链、GUI、音质或五次热运行结论。三套音频接口的正确性不能用于推断 RoFormer 的完整行为。

## K018：1296 短音频通过，完整窗口差异定位到频谱

日期：2026-09-11。完整 Rust 1296 网络严格加载全部 699 个张量，三组短网络与四组短音频的波形、长度、静音及取消对照通过。完整 8 秒合成窗口未通过 RMS 门限；将 Rust 频谱送入原始网络后复现几乎相同的波形误差，低能量频带的 RMSNorm 放大了前处理舍入差异。门限与原始黄金数据均保留，尚未将该模型标为完整验收通过。具体成功／失败记录、隔离方法和源码摘要见 [1296 数值记录](../benchmarks/2026-09-11-roformer-numerics.md)。

## K019：1296 采用明确的频谱精度基线并通过完整窗口

日期：2026-09-11。进一步对齐 Hann 系数并只提高 Rust FFT 精度仍未通过原 FP32 参考，因此固定工程基线 v1.1：两侧都用 float64 Hann、窗乘法与正向 FFT，再转 complex64；网络及 ISTFT 保持 FP32。原失败、旧参考及参数差异完整保留。三个独立小型频谱样本通过；同一个 352800 采样输入的完整 Rust 波形整轨 RMSE 为 `1.7671e-9`、最大误差 `2.2352e-7`，通过原门限。证据与新旧摘要见 [数值记录](../benchmarks/2026-09-11-roformer-numerics.md)。此结果不构成原库 FP32 FFT 逐样本复刻、跨窗、真实音频、性能或音质验收。

## K020：性能目标修订与首轮完整 PCM 优化对照

日期：2026-09-11。用户明确要求充分压榨 Rust 推理性能，撤销允许慢于 Python 参考的验收标准；[工程基线 v1.2](baseline.md) 将固定两线程作为对照，并要求继续验证硬件、后端和主要瓶颈。超过参考不构成优化完成，数值门限沿用 v1.1。

三套 VR 的 BatchNorm／激活融合已经完成真实片段的五次热运行、9 组掩码及 18 组 PCM 边界对照。DeEcho 中位数改善 7.7%，仍需排除环境漂移；两套 HP 的差异处于波动范围。Winograd 缓冲复用未能通过 A→B→A 证明收益，已撤回。5-HP 的 8 线程完整 PCM 已完成同线程 Python 对照，OpenVINO FP32 完整掩码候选也通过数值门限，但尚无默认窗口完整音频的后端结论。原始样本、构建、误差和采用限制见 [优化记录](../benchmarks/2026-09-11-vr-cpu-optimization.md)。这不代表四套模型、整曲或处理链性能完成。

## K021：1296 的布局遍历与串行激活开销

日期：2026-09-11。完整窗 CPU 采样确认 GEMM 以外还有明显的通用 strided 遍历与串行 GELU 开销。RoPE／RMSNorm 融合和 GELU 并行保留原计算顺序，三个内核逐位对照通过；实际第一层注意力在模块重复测量中明显改善，前馈收益仍受波动影响。完整 8 秒输出与既有 Rust 基线逐字节相同，短输入、重采样和真实片段 PCM 也通过原门限。

原始记录、资源使用和采用范围见 [1296 CPU 优化实验](../benchmarks/2026-09-11-roformer-cpu-optimization.md)。212.574 秒是一次完整窗正确性运行，没有五次热运行的完整窗结论；也不能用 8 线程与其他日期的 2 线程单值计算公平加速比。OpenVINO 的 FP32 模块候选支持继续研究成熟后端，尚未接入 Rust 原权重产品路径。1296 性能未完成验收。

## K022：关键实验缺少依赖时先安装，推进完整后端验证

日期：2026-09-11。用户明确要求把安装缺失实验依赖作为重要原则，不得因缺少模块／工具／运行时而降低目标或跳过关键实验，已写入 [性能协议](performance.md#执行方法)。本机安装 Intel compute runtime、graphics compiler 和 clinfo 后，OpenVINO 能识别 Intel UHD 核显。

1296 的完整频谱网络在 OpenVINO CPU FP32／8 线程下完成首次、额外预热和五次热运行，全部通过双轨 PCM 门限，Python 中位数 34.600 秒。新增安全 Rust 绑定探针，复用 core 的 float64 STFT／float32 ISTFT，七次完整门限也全部通过，中位数 35.001 秒、峰值 RSS 约 1.76 GiB；两组热运行范围重叠。

核显先遇到编译器崩溃，限制编译并发后进入推理，但连续运行触发 i915 fence 超时。补装 GPU 监控工具保存等待栈和设备计数器；关闭直接提交没有解决问题，Python 同图也重现。继续安装 Intel OpenCL Intercept Layer 3.0.6，强制顺序队列的三个独立进程共 21 次波形全部通过，保留诊断层但恢复乱序队列则再次超时。此证据支持继续检验队列／依赖路径；运行时间仍有漂移，尚未成为受支持的产品后端。记录、版本、来源与采用边界见 [完整后端实验](../benchmarks/2026-09-11-roformer-openvino.md)。产品直接从原权重构图及四模型全链路调优仍在推进。

## K023：实际内存可承受时，以速度优先

日期：2026-09-12。用户明确允许在机器内存足够、不会耗尽系统余量的范围内增加内存换取速度。因此 [工程基线 v1.3](baseline.md) 撤销此前 VR 4 GiB／1296 8 GiB 的固定 RSS 上限。1296 Rust 构图 CPU 实验约 8.74 GiB 的峰值本身不再构成验收失败；继续检验完整任务速度、可用内存与响应，不以压低 RSS 取代性能目标。此次观测本机约 23 GiB 物理内存、18 GiB 可用、无 swap；这是启动前快照，不是任务期间余量或其他机器的保证。质量门限和原始权重不变。

## K024：Rust 原始权重构图与可选 OpenVINO CPU 文件入口

日期：2026-09-12。Rust 严格读取原始 699 个张量并生成完整 1296 图，不依赖 Python 转换；301 帧 CPU／GPU 七次完整波形均通过，801 帧 GPU 一次完整窗口通过。构图已移入 core，在内存中传递给原生后端，并与原来的音频调度共用重采样、窗边界和残差。真实片段、单采样、极短重采样、静音等检查通过；原生请求取消后可复用。最初误用负超时的取消失败与修正后的结果都保留。

CLI 可选 `openvino` 构建和 `--backend openvino-cpu` 已直接从原 `.ckpt` 完成真实双轨 WAV 对照、拒绝覆盖与 Ctrl-C 清理。单次完整任务为 41.489 秒，原生取消到 core 返回约 1.16 毫秒，CLI 取消到进程退出约 0.895 秒；这些响应和单次结果不替代完整热运行、整曲、跨窗、GUI 或四模型性能验收。实测源码、二进制摘要、资源与限制见 [完整后端实验](../benchmarks/2026-09-11-roformer-openvino.md)。

## K025：ATRI 风格 Logo 接入与原生窗口检查

日期：2026-09-12。代码为 `b5418ed` 基础上的未提交工作树。按用户提供的 ATRI 图片设计银发红瞳、紫色耳机与双声波图案；桌面图标、界面 Logo 和 favicon 共用 [PNG 源图](../gui/src-tauri/icons/atri-v1/source.png)。图案由内置 image_gen 生成，来源、两次完整提示词及源图 SHA-256 记录在 [生成清单](../gui/src-tauri/icons/atri-v1/generation.json)。Tauri CLI 导出桌面 PNG、ICO 与 ICNS；源图使用不透明深紫背景，界面以圆角显示。

`pnpm build`、`cargo check --locked -p uvr-gui`、`pnpm gui:build` 与 `git diff --check` 通过，生成 `target/release/uvr-gui`。现有 `tools/reference/verify_gui.py --smoke-only` 在虚拟显示器中通过原生启动、四个模型选项与缺失输入恢复检查；[检查报告](../benchmarks/artifacts/gui-atri-logo-2026-09-12/native/report.json) 保存二进制摘要，[原生窗口截图](../benchmarks/artifacts/gui-atri-logo-2026-09-12/native/ready.png) 确认 48 像素 Logo 正常显示。本条验证范围为图标资源、桌面构建与窗口启动。

## K026：前后端共享运行时配置与本机优化 GUI

日期：2026-09-13。为使已测得的 CPU 优化能被普通用户直接使用，CLI 与 GUI 统一通过 core 的 `RuntimeOptions` 创建每任务线程池，显式传递 VR 窗口／批次／并发及 Burn 1296 注意力批量／布局。可用时 1296 默认选择 OpenVINO CPU，VR 采用 512 帧、批次 1、HP 并发 4；线程默认最多 8，DeEcho 和批处理 HP 显示实际生效的调度限制。运行时参数与环境变量优先级见三语[运行时指南](runtime.zh-CN.md)。

`pnpm build:native` 已构建当前 CPU 的优化版 CLI 和 GUI，分别为 6,943,112 与 18,905,752 字节；可选 OpenVINO CPU 原生库单独计量，模型仍在外部。精确摘要与历史性能口径见[记录](../benchmarks/2026-09-13-runtime-summary.json)。程序恢复初始化提交 `06eaaea` 的 UVR 图标；中日英、系统／亮／暗色及三种主题均可在任务运行中切换并保存。

完整 workspace 全特性测试、Clippy 严格检查、格式和前端构建通过。并行运行测试时发现原下载取消测试在连接前计时，繁忙机器可能提前取消并让本地服务器一直等待连接；改为收到响应、进入正文等待后开始取消计时，继续检验停滞下载的响应与原文件保护。浏览器验证覆盖运行时 9 组、外观 7 组、国际化 6 组，另以可信鼠标点击检验折叠高级参数的 6 组原生表单校验。

最终 GUI 在 `PATH=/nonexistent` 下直接加载原权重，1296 的 3 秒真实片段经 OpenVINO CPU／8 线程通过双轨波形、拒绝覆盖和取消清理；取消约 0.456 秒。[报告](../benchmarks/artifacts/gui-runtime-2026-09-13-1296/report.json) 保留二进制和波形摘要。5-HP 的 10 秒真实片段经 Burn／8 线程也通过同样检查；取消等待当前窗口组，约 30.306 秒，见[报告](../benchmarks/artifacts/gui-runtime-2026-09-13-5hp/report.json)。这些是功能与正确性运行，不替代独立性能复测、全部模型／整曲验收或 Windows 实机音频验证。

## 后续条目格式

记录编号、日期、问题、参考版本／权重校验和、执行命令、输入与参数、结果、产物位置、适用范围及仍未解决的问题。性能实验同步链接 [性能文档](performance.md)，避免复制两套数值。
