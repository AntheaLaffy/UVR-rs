# 后验知识与验证记录

更新时间：2026-09-09。源码检查只能证明参考实现的行为；没有运行模型就不能宣称音质、速度或导出可行性。固定参考版本见 [references.md](references.md)。

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

## 后续条目格式

记录编号、日期、问题、参考版本／权重校验和、执行命令、输入与参数、结果、产物位置、适用范围及仍未解决的问题。性能实验同步链接 [性能文档](performance.md)，避免复制两套数值。
