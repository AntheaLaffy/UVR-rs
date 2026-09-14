# Inference runtimes and tuning

English | [简体中文](runtime.zh-CN.md) | [日本語](runtime.ja.md)

The desktop app and CLI use the same Rust inference settings and validation. You can tune CPU use and memory without converting your original UVR weights. Start with the recommended settings; larger batches and more parallel windows can increase memory use without making a model faster.

## Recommended settings

| Model | Runtime | Current defaults |
| --- | --- | --- |
| 5-HP / 6-HP | Burn CPU with the shared optimized VR kernels | 512 frames, inference batch 1, 4 parallel windows |
| DeEcho | Burn CPU | 512 frames, inference batch 1, 1 parallel window |
| BS-RoFormer 1296 | OpenVINO CPU when built in and available; otherwise Burn CPU | OpenVINO: FP32, latency mode, one stream. Burn: time batch 62, frequency batch 301, flattened linear layout, one parallel window |

The CPU thread budget defaults to the smaller of 8 and the available CPU count. These settings come from the project's current measurements on an Intel i5-13420H; they are a useful starting point, not a guarantee of the fastest settings on every machine. Original weights, output gain, FP32 network precision and the documented audio processing remain the same across runtime selection.

OpenVINO support is currently limited to 1296 on CPU. Experimental GPU and other backend probes are not available product options. The app checks for a usable OpenVINO CPU runtime before recommending it. If it is unavailable, the recommendation is Burn; explicitly requesting an unavailable backend returns an error.

### Tuning still in progress

DeEcho's batch and window-concurrency tuning is unfinished. The implementation currently enforces `inference_batch=1` and `window_parallelism=1`; the window length remains configurable, with a default of 512 frames and a range of 144–2048 in multiples of 16. Its bidirectional LSTM requires independent state for each window, and future context prevents assuming equivalent state reuse across windows. These constraints do not rule out independent-window concurrency. The fixed values are conservative implementation limits: complete DeEcho audio comparisons at concurrency 1/2/4, including state isolation and output checks, have not been completed. The HP measurements do not prove that higher DeEcho concurrency would be slower.

BS-RoFormer 1296 tuning is also unfinished on both Burn and OpenVINO. Burn's layout and attention-batch changes have timings and waveform checks; OpenVINO CPU has repeated short-audio comparisons and is available in the app and CLI. These are partial results. Repeated full-window measurements, controlled before/after comparisons for recent layout changes, and full-track and multi-model-chain performance validation remain outstanding. Current defaults are starting points, not a completed search for the fastest configuration. See the [remaining work and acceptance criteria](performance.md#尚未完成的调优与验收).

## Desktop controls

The **Inference runtime** section of the task form contains the backend and CPU thread budget. Expand **Advanced inference settings** for the selected model's options. The controls, configuration summary and task log show which settings apply to the next task.

- Switching models preserves each VR model's window and batch preferences, and the 1296 backend and Burn settings.
- DeEcho uses batch 1 and one parallel window. HP batching above 1 uses batch scheduling, so its independent window concurrency control is disabled.
- OpenVINO uses its own scheduling; Burn-specific batch and layout controls are hidden when it is selected.
- Settings are saved locally. **Reset defaults** restores the selected model's recommended settings and the default thread budget.
- Controls are locked during a task. After completion or cancellation, the next task creates its own thread pool using the new settings; no restart is needed.

The language selector supports English, Simplified Chinese and Japanese. Changing language preserves paths and inference settings, and updates task status while processing. The **Appearance** menu offers system, light and dark modes, with violet, blue and teal accents. Language and appearance preferences are saved independently.

## CLI options

Options follow the required positional arguments and can appear in any order. Values use ASCII names and positive decimal integers. Unknown or repeated flags, missing values, and parameters for the wrong model or backend return usage error code 2 before the file task starts.

Use the global language option before the command: `uvr --lang en --help`, `uvr --lang ja --help` or `uvr --lang zh-CN --help`. `UVR_LANG` sets the default CLI language; `--lang` overrides it. Machine-readable output keys, filenames and option values are unchanged.

| CLI flag | Applies to | Default / constraint |
| --- | --- | --- |
| `--threads N` | All models and runtimes | Up to 8 available CPU threads; positive integer |
| `--backend burn\|openvino-cpu` | `separate-1296` | OpenVINO CPU if available, otherwise Burn |
| `--window-frames N` | `separate-vr` | 512; multiple of 16; 272–2048 for HP, 144–2048 for DeEcho |
| `--inference-batch N` | `separate-vr` | 1; range 1–4, effective value fixed to 1 for DeEcho |
| `--parallel-windows N` | VR, or 1296 with Burn | Range 1–8; HP default 4, DeEcho and 1296 default 1; HP with batch >1 uses effective value 1 |
| `--time-batch N` | 1296 with Burn | 62; positive integer |
| `--frequency-batch N` | 1296 with Burn | 301; positive integer |
| `--linear-layout flattened\|batched` | 1296 with Burn | `flattened` |

RoFormer batches group independent attention sequences, without shortening the sequence being processed. Values larger than the available bands or frames are naturally limited by the input shape. Changing VR window size may change the separation result; use the default when comparing against existing reference fixtures.

```sh
# The native binaries are produced by pnpm build:native.
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

The CLI prints the effective configuration, including DeEcho and batched HP scheduling adjustments. Output directories must not contain the same target filenames: existing tracks are never overwritten. Ctrl-C requests cancellation; the exit code is 130. VR cancellation waits for the current window or parallel group; sequential Burn 1296 supports checks within a window, and OpenVINO supports native request cancellation.

### Existing environment variables

Explicit CLI flags override the corresponding environment variable, which overrides the default. An explicitly overridden variable is not parsed; OpenVINO ignores the Burn-only variables.

| Environment variable | Corresponding CLI flag |
| --- | --- |
| `RAYON_NUM_THREADS` | `--threads` |
| `UVR_ROFORMER_TIME_BATCH` | `--time-batch` |
| `UVR_ROFORMER_FREQUENCY_BATCH` | `--frequency-batch` |
| `UVR_ROFORMER_WINDOW_PARALLELISM` | `--parallel-windows` for Burn 1296 |
| `UVR_LINEAR_LAYOUT` | `--linear-layout` |

The GUI uses a valid `RAYON_NUM_THREADS` value as its initial thread preference; a saved or edited GUI value takes precedence. Its other settings come from the shared defaults and visible controls, without hidden `UVR_*` overrides.

## Build the optimized local applications

The measured native build uses release optimization and `-C target-cpu=native`, with the existing optimized shared kernels. It builds both applications into a separate target directory so they cannot be confused with portable builds.

```sh
pnpm install --frozen-lockfile
make
target/native/release/uvr-gui
```

This Linux x86_64 build expects the OpenVINO native libraries in `.local/openvino-2026.3.1/lib/`, or in the directory specified by `UVR_OPENVINO_LIB_DIR`. That directory must include `libopenvino_c.so` and its CPU plugin and dependencies. The script copies the required CPU libraries into `target/native/release/lib/`; both executables load them directly. Keep `lib/` beside the executables. No Python process, conversion script or environment activation is needed to run the built applications.

```sh
UVR_OPENVINO_LIB_DIR=/path/to/openvino/lib pnpm build:native

# Build the native Burn runtime when OpenVINO is not installed.
pnpm build:native --burn-only
target/native-burn/release/uvr-gui
```

The Burn-only build writes to `target/native-burn/release/`, separate from `target/native/release/`, so an earlier OpenVINO build cannot leave optional libraries in its distribution directory. Each build writes executable sizes, SHA-256 hashes and build settings to its own `release/build-info.json`.

The current OpenVINO-enabled native build has a 6.94 MB CLI and 18.91 MB GUI; the optional OpenVINO library directory adds 86.80 MB. GUI plus those libraries totals 105.71 MB, excluding model weights and OS libraries. Exact byte counts, hashes and historical measurements are in the [size evidence](performance-summary.md).

`make` is the short local entry point for `pnpm build:native`; `make native-burn` selects the independent Burn-only output. A native executable targets the build machine's CPU features: use the ordinary `pnpm gui:build` / `cargo build --release --locked -p uvr-cli` commands for a generic build intended for other CPUs. Neither build mode bundles model weights.

Cross-compiling Windows binaries does not inherently reduce inference speed. Release optimization, target CPU instructions, toolchain and Windows scheduling determine the result. The native helper here is Linux-only; building and testing Windows on a Windows machine is a practical way to validate the platform before release. Do not apply one developer's `target-cpu=native` settings to an untested distribution for other CPUs.

## Evidence and maintenance

The current HP defaults are supported by complete 5-HP and 6-HP audio experiments. Burn 1296's layout and batch defaults have waveform checks and recorded timings. OpenVINO has complete short-audio waveform comparisons, repeated CPU measurements, and a Rust path that builds the graph directly from the original checkpoint. These results do not establish universal speedups over UVR, better model quality, or complete long-track and multi-model-chain validation.

The detailed research notes are currently in Chinese: [VR CPU experiments](../benchmarks/2026-09-11-vr-cpu-optimization.md), [Burn 1296 experiments](../benchmarks/2026-09-11-roformer-cpu-optimization.md), [OpenVINO experiments](../benchmarks/2026-09-11-roformer-openvino.md), and the [performance protocol](performance.md). For contributor checks and the shared configuration contract, see [CONTRIBUTING](../CONTRIBUTING.md).
