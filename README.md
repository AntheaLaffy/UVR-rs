# UVR Rust

English · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

**VR separation at about 3× the CPU speed of the Python reference, with a 6.94 MB native CLI.**

A focused alternative for UVR users who want faster CPU processing and a smaller application. UVR Rust loads original UVR weights directly, with a desktop app and a scriptable CLI. It does not need to carry Python or PyTorch to process audio.

| What matters | UVR Rust |
| --- | --- |
| VR CPU inference | RTF ≈2 versus ≈6 for the Python comparison: about 3× faster |
| Current native executable sizes | CLI 6.94 MB; desktop app 18.91 MB, excluding weights and optional libraries |
| Optional 1296 acceleration libraries | OpenVINO adds 86.80 MB; CLI + libraries 93.74 MB, GUI + libraries 105.71 MB |
| Original weights | Direct `.pth` / `.ckpt` loading; no conversion |
| Daily workflow | Local processing, GUI and CLI, progress, cancellation and output protection |

Speed compares recorded local 5-HP configurations with different clips and thread counts; it is not a comparison at equal resources. RTF 2 means roughly 20 seconds of processing per 10 seconds of audio. Sizes describe the September 13 Linux native release with symbols stripped; all totals exclude model weights and system libraries. The GUI already includes inference and does not require the CLI file. See the [performance and size evidence](docs/performance-summary.md) for exact bytes, hashes, historical builds and measurement conditions.

## Choose a model

| You want to… | Model / CLI key | Original weight file | Output tracks |
| --- | --- | --- | --- |
| Separate vocals and accompaniment | BS-RoFormer 1296 / `1296` | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | `vocals`, `instrumental` |
| Try Karaoke separation | 5-HP / `5hp` | `5_HP-Karaoke-UVR.pth` | `primary`, `residual` |
| Compare a second Karaoke model | 6-HP / `6hp` | `6_HP-Karaoke-UVR.pth` | `primary`, `residual` |
| Reduce echo and reverb | DeEcho / `deecho` | `UVR-DeEcho-DeReverb.pth` | `primary`, `residual` |

For Karaoke and DeEcho, `primary` is the model output and `residual` comes from the complementary mask. Audition both tracks on your material; these labels do not promise a perfect lead/backing-vocal or dry/reverb split. Weights are downloaded separately, not bundled with the app. Sources and exact filenames are recorded in the [model manifest](references/targets.json).

Single-model CLI and desktop processing are implemented. Automatic multi-model chains, broader platform coverage and full-song quality/performance acceptance remain in progress. The interface and user documentation support English, Simplified Chinese and Japanese. This project supports the four listed checkpoints, rather than the entire UVR model catalog.

## Use it

### Desktop

Launch `uvr-gui`, select an input file and model, then choose model and output directories. The model manager checks existing weights and can download missing models from the offered sources. Once weights are ready, separation works offline.

Inference runtime settings expose the available backend and thread count; advanced settings expose the applicable window, batch and layout controls. Settings are remembered locally, with a reset to the recommendations for the selected model. The task log records effective settings so you can reproduce a run in the CLI. Both entry points use the same Rust inference implementation and parameter rules.

Choose English, Simplified Chinese or Japanese, follow the system's light/dark mode or set your own, and select an accent color. Language and appearance preferences are saved locally.

### Command line

After the standard build below, run from the repository root:

```sh
./target/release/uvr --lang en separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296

./target/release/uvr --lang en separate-vr deecho \
  models/UVR-DeEcho-DeReverb.pth input.wav outputs/deecho

./target/release/uvr --lang en --help
```

Use `target/native/release/uvr` instead after the native build. `--threads` works with both model families; VR exposes `--window-frames`, `--inference-batch` and `--parallel-windows`. 1296 also exposes `--backend`, and its Burn path supports attention batch sizes, window parallelism and linear layout. See the [complete option reference and examples](docs/runtime.md).

Put `--lang en`, `--lang ja` or `--lang zh-CN` before the command, or set `UVR_LANG`; explicit selection wins. The default remains Chinese. Machine-readable stdout field names stay the same across languages, and underlying technical errors retain their original text.

Inputs: mono/stereo WAV, FLAC and MP3. Outputs: two 44.1 kHz stereo, 32-bit float WAVs named `<input>_<model>_<track>.wav`, without normalization or clipping. Mono becomes stereo and other sample rates are resampled. Existing outputs are preserved. Cancel with Ctrl-C or the desktop cancel button; latency depends on the active computation. CLI exit codes: `0` success, `1` processing failure, `2` invalid arguments, `130` cancellation.

`inspect-audio <file>` reports decoded audio properties. `inspect-weights <file>` reports size, SHA-256 and the UVR metadata MD5; computing a fingerprint alone does not establish model support.

## Build and install

For the fastest local build on Linux x86_64, install `make` and the prerequisites described in the [runtime guide](docs/runtime.md), then run:

```sh
pnpm install --frozen-lockfile
make
```

The optimized programs are written to `target/native/release/`; keep its `lib/` directory beside them when OpenVINO is enabled. Use `make native-burn` for a Burn-only build. Generic builds, Windows guidance, CI artifacts, and dependency details are in the [runtime guide](docs/runtime.md). Development builds are available from [GitHub Actions](https://github.com/AntheaLaffy/UVR-rs/actions/workflows/build.yml).

If `make` is unavailable on your system, use the exact underlying commands in the [runtime guide](docs/runtime.md) instead.

## Contribute and explore

Start with [Contributing](CONTRIBUTING.md) for architecture, checks, bug reports and evidence requirements. The [runtime guide](docs/runtime.md) explains settings and deployment; [benchmark records](benchmarks/README.md) preserve measured results and remaining limitations.

Most research notes are currently in Chinese: [scope and acceptance](docs/tasks.md), [engineering baseline](docs/baseline.md), [performance protocol](docs/performance.md), [validated findings](docs/posterior-knowledge.md), and [upstream references](docs/references.md). UVR Rust builds on the referenced model and algorithm work.
