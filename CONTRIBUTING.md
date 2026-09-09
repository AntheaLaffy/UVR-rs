# 开发约定

先阅读 [任务](docs/tasks.md) 与 [先验知识](docs/prior-knowledge.md)，再按证据更新 [后验知识](docs/posterior-knowledge.md)。推理后端未决定，初始化阶段不引入推理框架或假实现 API。

`core/` 持有音频处理和推理逻辑；CLI 与 Tauri 宿主调用库，前端只负责交互。需要共享的参数与任务协议在出现真实实现需求后定义。

## 本地检查

```sh
cargo fmt --all -- --check
pnpm build
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --locked -p uvr-cli -- --help
cargo run --locked -p uvr-cli -- --version
git diff --check
```

新 clone 先运行 `pnpm install --frozen-lockfile`。核心开发可以只检查 `cargo check --locked`；完整桌面检查需要安装系统依赖。当前没有推理测试，不为占位模块编写空测试；进入实现阶段后用参考中间张量、音频回归和失败场景验证行为。

Cargo.lock 与 pnpm-lock.yaml 随应用提交。权重、客户音频、生成分轨和大型基准产物放在被忽略的目录；可共享的基准清单与结果摘要需记录输入校验和。

GUI 占位图标的源文件是 `gui/src-tauri/icons/source.svg`，修改后运行 `pnpm --filter @uvr/gui tauri icon src-tauri/icons/source.svg --output src-tauri/icons --png 32` 再提交生成的 PNG。

## 知识记录

- 客户要求与偏好直接标注为要求，不伪装成实验结论。
- 未验证判断保留在先验文档，并给出验证方法。
- 后验条目写明日期、代码版本、证据位置、结果与适用范围；修正结论时保留失效原因。
- 性能结论必须对应 [性能协议](docs/performance.md)，不能把空壳构建耗时作为推理性能。
- 提交说明解释动机、范围和验证限制；参考子模块更新需明确新的提交号与原因。
