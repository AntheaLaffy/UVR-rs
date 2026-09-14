# Less waiting. Less runtime to carry.

English | [简体中文](performance-summary.zh-CN.md) | [日本語](performance-summary.ja.md)

UVR Rust focuses on CPU separation speed and a small native application. Current local VR observations are **RTF ≈2 versus ≈6 for the Python reference**, an approximately **3× processing-rate difference**. The current Linux native CLI is **6.94 MB** and desktop executable **18.91 MB**, without bundling Python or PyTorch. Optional OpenVINO acceleration adds **86.80 MB** of libraries; the GUI with those libraries is **105.71 MB**, excluding model weights and OS libraries.

## What the speed numbers mean

RTF is processing time divided by audio duration; lower is faster. RTF 2 means about 20 seconds to separate 10 seconds of audio. It does not mean live real-time processing, which requires RTF ≤1.

| Archived observation | Rust native | Python reference |
| --- | --- | --- |
| Model | 5-HP | 5-HP |
| Machine | Intel i5-13420H, Linux | Same machine |
| Audio duration | 10 seconds | 3 seconds |
| CPU threads | 8 | 2 |
| Window / batch / parallel windows | 512 / 1 / 4 | 512 / 1 / 1 |
| Five warm runs, median RTF | 1.9917 | 6.0809 |

These are the project's observed local configurations, including the benefit of its newer CPU scheduling and kernels. Input duration and thread budgets differ. The ≈3× headline describes those observations; it is not a controlled same-resource comparison or a claim about every UVR model. The measured boundary includes PCM analysis, inference and reconstruction, excluding model loading and file decoding/encoding. Rust's five warm outputs passed the existing waveform tolerances.

The [tracked data summary](../benchmarks/2026-09-13-runtime-summary.json) preserves each sample, configuration and source checksum. The earlier [VR experiment log](../benchmarks/2026-09-11-vr-cpu-optimization.md) contains separate matched-resource tests; keep the two comparison types distinct when publishing results.

DeEcho's batch/window-concurrency tuning and BS-RoFormer 1296's performance tuning are unfinished. DeEcho's enforced batch 1 and single window have no completed concurrency 1/2/4 audio comparison; RoFormer's Burn/OpenVINO measurements remain partial, with full-track and multi-model-chain validation outstanding. The 5-HP speed figures above do not describe either model. See the [runtime guide](runtime.md#tuning-still-in-progress) for current limits and remaining work.

## What is included in the size

The September 13 build completed at 10:06:28 UTC on the Intel i5-13420H. It uses Linux x86_64 release optimization, `-C target-cpu=native`, stripped symbols and the OpenVINO feature. Its `target/native/release/build-info.json` records executable bytes, SHA-256 hashes and build settings. Those hashes were checked against the final files and copied into the [tracked summary](../benchmarks/2026-09-13-runtime-summary.json), along with every staged library's size and hash.

| Current payload | Bytes | MB |
| --- | ---: | ---: |
| CLI executable `uvr` | 6,943,112 | 6.94 |
| Desktop executable `uvr-gui` | 18,905,752 | 18.91 |
| Optional OpenVINO CPU libraries, `lib/` | 86,800,978 | 86.80 |
| CLI + OpenVINO libraries | 93,744,090 | 93.74 |
| GUI + OpenVINO libraries | 105,706,730 | 105.71 |

MB means 1,000,000 bytes. The optional-library total sums the ten staged files, including copied aliases, without compression or filesystem block rounding. All totals exclude model weights, OS libraries and metadata files. The GUI includes its inference code; running it does not require adding the CLI executable. A Burn-only build goes into `target/native-burn/release/` and needs none of these OpenVINO libraries; its executable size is not inferred from this OpenVINO-enabled build.

The historical September 12 release remains recorded separately: `uvr` was 9,122,024 bytes (9.12 MB), and `uvr-gui` was 25,811,240 bytes (25.81 MB). These older artifacts preceded the current UI/runtime update. The size reduction belongs to the current build, including symbol stripping; it is separate from the archived inference timings above.

The Burn runtime does not carry Python or PyTorch. Tauri uses the platform's webview rather than shipping a Chromium browser. OpenVINO is optional for accelerating 1296 and adds the native libraries counted above. All models still need their original weights, and Linux desktop use requires the usual GTK/WebKit system libraries.

Upstream UVR's Windows/macOS installers already bundle their Python dependencies, so the difference is the runtime being distributed, not a claim that every UVR user has to install Python manually. See the [fixed upstream installation notes](../upstream/README.md).

## Reproduce and improve

Build with `pnpm build:native`, select the recommended runtime and use the [runtime guide](runtime.md) for exact options. For a publishable comparison, use the same audio, weight checksum, preset, timing boundary and hardware; record thread budgets, native libraries and the full set of warm runs. Treat waveform checks as a requirement for adopting a speed improvement.

The detailed [performance protocol](performance.md) and experiment logs are currently in Chinese. New comparative results should add evidence rather than replace the source or conditions of earlier numbers.
