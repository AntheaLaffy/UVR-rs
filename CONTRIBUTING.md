# Contributing to UVR Rust

English · [简体中文](CONTRIBUTING.zh-CN.md) · [日本語](CONTRIBUTING.ja.md)

Help us make local audio separation easier to use and its results easier to verify. Start with the [user guide](README.md) and [runtime reference](docs/runtime.md). Inference changes must preserve the documented audio semantics and quality thresholds; a faster kernel is useful when it improves a real task without breaking those guarantees.

Release and workspace-maintenance commands are documented in [Workspace maintenance](docs/maintenance.md).

## Find the right layer

| Location | Responsibility |
| --- | --- |
| `core/` | Audio decoding/DSP, original checkpoint loading, inference and file tasks |
| `crates/` | Focused crates.io facades; keep them thin and preserve `uvr-core` as the implementation source |
| `cli/` | Command parsing, environment compatibility, terminal progress and exit codes |
| `gui/src/` | Desktop interaction, validation, preferences and progress display |
| `gui/src-tauri/` | Native desktop commands, model management and task coordination |
| `tools/reference/` | Independent reference generation and verification; never the product runtime |
| `benchmarks/` | Reproducible correctness and performance experiments |
| `upstream/`, `agent/deepseek-harness/` | Pinned references outside the product workspace |

The Cargo workspace defaults to core and CLI so command-line development does not require desktop libraries. The pnpm workspace owns the GUI. Reference submodules are optional for ordinary builds; initialize them only for work requiring upstream sources, following [References](docs/references.md).

Keep inference and audio behavior in core. CLI and desktop use `RuntimeOptions` and `separate_file_with_options`, including validation, effective scheduling parameters and a thread pool for each task. Expose an option only when the selected model/backend implements it. Avoid process-wide environment mutation: the CLI resolves its legacy environment overrides into explicit task options, while the desktop sends its selected settings.

Core's `burn-cpu` feature enables inference and PCM tasks; `audio-io` adds codecs and file tasks. CLI enables both. `openvino` is optional and needs the native CPU runtime to be available. Python/PyTorch belongs only to independent verification: normal builds, application startup and separation must not call it. Fixed fixtures keep normal Cargo tests independent of that environment.

## Check a change

Use a Rust 2024 toolchain. For desktop work, also install Node.js 24, pnpm 12.1.0 and the [Tauri system prerequisites](https://v2.tauri.app/start/prerequisites/), then run `pnpm install --frozen-lockfile`.

Run relevant checks while iterating and complete applicable workspace checks before handoff:

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

For backend or runtime changes, also check the optional build:

```sh
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/openvino,uvr-cli/openvino
cargo check --workspace --all-targets --locked --all-features
```

Core-only development can start with `cargo check --locked`. Tests cover fingerprints, DSP/codecs, cancellation, output protection, runtime validation and CLI failures. Passing these checks without large local weights does **not** establish full-model acceptance. Use the [backend probes](benchmarks/backend-probe/README.md) and [independent verification tools](tools/reference/README.md) for model and waveform comparisons; state exactly what you ran and what remains unverified.

For GUI changes, verify the native app as well as the browser preview. Exercise model/backend switching, validation, saved settings, cancellation and output handling when affected. A successful web build cannot establish native command or inference behavior.

For Windows work, pair Linux development with build and runtime checks on a Windows collaborator's machine; Windows acceptance is still open. Compare release optimizations, CPU instruction targets, toolchains, backend settings and thread scheduling before attributing a speed difference to cross-compilation. `tools/build-native.mjs` is Linux-only; do not carry `target-cpu=native` from the build machine into a general Windows release.

## Report a bug or propose a change

A useful report identifies the user-visible problem and gives the shortest reproduction: application revision and build command, OS/CPU, model, actual runtime settings, input format/length, expected behavior and the relevant error or task log. For performance reports include input and weight hashes, timing boundaries and repeated measurements. Avoid attaching private audio or large weights; use a shareable synthetic reproduction when it demonstrates the same problem.

Keep each change reviewable around one problem. Explain the reason, resulting behavior, important tradeoffs and validation limits in the PR description. Update CLI help, GUI controls and runtime documentation together when changing a shared option. Keep the English, Chinese and Japanese interface text and user documentation aligned; leave command names, paths and option values identical across translations. Check language switching and light/dark appearance when changing visible UI text or styles.

Commit `Cargo.lock` and `pnpm-lock.yaml` with dependency changes. Keep original weights, user audio, generated stems and large benchmark artifacts in ignored directories. Share manifests and result summaries with input checksums. Reference-submodule updates must identify the new commit and why it is needed.

## Preserve the evidence

The research notes are currently mostly Chinese. Read [Scope](docs/tasks.md), [Baseline](docs/baseline.md) and [Prior knowledge](docs/prior-knowledge.md) before changing inference assumptions. Update [Validated findings](docs/posterior-knowledge.md) with the date, code revision, evidence location, result and applicable limits.

- Record user requirements as requirements, not experimental findings.
- Keep unverified judgments in prior-knowledge notes, with a verification method.
- Follow the [performance protocol](docs/performance.md). Keep model, input, quality thresholds and timing boundaries comparable; local operator speedups or build times do not establish end-to-end inference gains.
- Preserve failed or superseded findings and explain why they no longer apply. Do not widen quality tolerances to hide a regression.

Use the [benchmark index](benchmarks/README.md) for detailed experiment history. The README should help a new user choose and run the product, rather than carry an experimental archive.

## Shared branding assets

Desktop icons, the app logo and favicon derive from the original `gui/src-tauri/icons/source.svg`, a light green waveform on a dark blue-green background. Keep this source as the shared identity so desktop and web assets remain consistent. Generated desktop icons live in `gui/src-tauri/icons/uvr/`; web assets are `gui/public/branding/uvr.png` and `uvr-32.png`.

After editing the source, export through Tauri and copy only the platform assets used by this project:

```sh
pnpm --filter @uvr/gui tauri icon src-tauri/icons/source.svg --output /tmp/uvr-icons
cp /tmp/uvr-icons/{32x32.png,64x64.png,128x128.png,128x128@2x.png,icon.png,icon.ico,icon.icns} gui/src-tauri/icons/uvr/
cp /tmp/uvr-icons/128x128.png gui/public/branding/uvr.png
cp /tmp/uvr-icons/32x32.png gui/public/branding/uvr-32.png
```
