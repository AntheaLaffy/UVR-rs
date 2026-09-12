# 开发约定

先阅读 [任务](docs/tasks.md)、[工程基线](docs/baseline.md) 与 [先验知识](docs/prior-knowledge.md)，再按证据更新 [后验知识](docs/posterior-knowledge.md)。core 的可选 `burn-cpu` 提供 VR 网络与 PCM 任务，`audio-io` 提供文件编解码，CLI 默认启用两者。未实现能力不能作为可用 API 或界面功能展示。

`core/` 持有音频处理和推理逻辑；CLI 与 Tauri 宿主调用库，前端只负责交互。需要共享的参数与任务协议在出现真实实现需求后定义。

运行时不得依赖 Python。`tools/reference/` 的 Python／PyTorch 仅用于独立验证；应用构建、启动和音频处理不能调用该环境。参考工具依赖由自己的 `uv.lock` 固定，普通 Cargo 测试使用已保存的样本，不要求 Python。重新生成参考样本的方法见 [验证工具说明](tools/reference/README.md)。

## 本地检查

```sh
cargo fmt --all -- --check
cargo test --locked -p uvr-core -p uvr-cli
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/burn-cpu
cargo clippy --locked -p uvr-core --all-targets --features burn-cpu -- -D warnings
pnpm build
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --locked -p uvr-cli -- --help
cargo run --locked -p uvr-cli -- --version
git diff --check
```

新 clone 先运行 `pnpm install --frozen-lockfile`。核心开发可以只检查 `cargo check --locked`；完整桌面检查需要安装系统依赖。普通 Cargo 测试覆盖文件摘要、UVR 标识、DSP／编解码对照、取消／输出保护与 CLI 失败场景。依赖大型权重的完整网络及音频验证另见 [探针说明](benchmarks/backend-probe/README.md)，不把缺少本地权重时的普通测试通过当作模型已验收。

Cargo.lock 与 pnpm-lock.yaml 随应用提交。权重、客户音频、生成分轨和大型基准产物放在被忽略的目录；可共享的基准清单与结果摘要需记录输入校验和。

桌面图标、界面 Logo 与 favicon 共用 `gui/src-tauri/icons/atri-v1/source.png`，避免更新后出现不同标志。图案按用户提供的 ATRI 图片设计，由内置 image_gen 生成；参考说明与完整提示词保存在同目录的 `generation.json`。源图是带不透明深紫背景的 PNG 位图。

更新源图后，用 Tauri CLI 重新导出并同步前端资源。临时目录用于接收全平台产物，仓库只保留当前桌面应用需要的文件：

```sh
pnpm --filter @uvr/gui tauri icon src-tauri/icons/atri-v1/source.png --output /tmp/uvr-atri-icons
cp /tmp/uvr-atri-icons/{32x32.png,64x64.png,128x128.png,128x128@2x.png,icon.png,icon.ico,icon.icns} gui/src-tauri/icons/atri-v1/
cp /tmp/uvr-atri-icons/128x128.png gui/public/branding/uvr-atri.png
cp /tmp/uvr-atri-icons/32x32.png gui/public/branding/uvr-atri-32.png
```

## 知识记录

- 客户要求与偏好直接标注为要求，不伪装成实验结论。
- 未验证判断保留在先验文档，并给出验证方法。
- 后验条目写明日期、代码版本、证据位置、结果与适用范围；修正结论时保留失效原因。
- 性能结论必须对应 [性能协议](docs/performance.md)，不能把空壳构建耗时作为推理性能。
- 提交说明解释动机、范围和验证限制；参考子模块更新需明确新的提交号与原因。
