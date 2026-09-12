"""Test the complete 1296 spectral network on OpenVINO with baseline PCM gates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np
import openvino as ov
from openvino.properties import streams
import torch

from audit_vr import sha256
from benchmark_vr_audio_reference import verify
from generate_roformer_model import load_model
from roformer_precision import precise_stft
from verify_vr_cli import load_tensor


class SpectralNetwork(torch.nn.Module):
    """Use original modules; keep the prescribed FP64 STFT outside the graph."""

    def __init__(self, model):
        super().__init__()
        self.band_split = model.band_split
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.mask = model.mask_estimators[0]

    def forward(self, features):
        x = self.band_split(features)
        batch, frames, bands, width = x.shape
        for time_transformer, frequency_transformer in self.layers:
            x = x.permute(0, 2, 1, 3).reshape(batch * bands, frames, width)
            x = time_transformer(x)
            x = x.reshape(batch, bands, frames, width).permute(0, 2, 1, 3)
            x = frequency_transformer(x.reshape(batch * frames, bands, width))
            x = x.reshape(batch, frames, bands, width)
        return self.mask(self.final_norm(x))


def analyze(wave, kwargs):
    with precise_stft(), torch.inference_mode():
        spectrum = torch.stft(torch.from_numpy(wave), **kwargs, return_complex=True)
    frames = spectrum.shape[-1]
    # Feature order is frequency, stereo channel, real/imaginary component.
    features = torch.view_as_real(spectrum).permute(2, 1, 0, 3).reshape(1, frames, 4100)
    return spectrum, features.contiguous()


def reconstruct(mask, spectrum, wave, kwargs, audio):
    frames = spectrum.shape[-1]
    mask = torch.from_numpy(mask).reshape(frames, 1025, 2, 2).permute(2, 1, 0, 3).contiguous()
    with precise_stft(), torch.inference_mode():
        vocals = torch.istft(spectrum * torch.view_as_complex(mask), **kwargs).numpy()
    return np.stack([vocals, wave - vocals]) if audio else vocals[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", default="local_audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--compilation-threads", type=int)
    parser.add_argument("--save-ir", type=Path)
    parser.add_argument("--ir", type=Path, help="reuse a previously converted graph; waveform gates still apply")
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("threads must be positive and output must not exist")
    if args.compilation_threads is not None and args.compilation_threads < 1:
        parser.error("compilation threads must be positive")
    if args.save_ir and (args.save_ir.exists() or args.save_ir.with_suffix(".bin").exists()):
        parser.error("IR output must not exist")
    trace = args.output.with_suffix(".jsonl").open("x")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if not any("float64" in s.get("stft_precision", "") for s in manifest["sources"]):
        raise ValueError("fixture must use baseline FP64 forward STFT")
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    wave = load_tensor(args.fixtures, manifest["tensors"][case["input"]]).copy()
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    audio = "audio" in manifest
    if audio:
        params = manifest["audio"]["cases"][args.case]
        if params["sample_rate"] != 44100 or params["windows"] != 1:
            raise ValueError("probe requires one window at 44100 Hz")
    else:
        expected = expected[None]
    if wave.shape[0] == 1:
        wave = wave.repeat(2, axis=0)
    if wave.shape[0] != 2 or not 1323 <= wave.shape[1] <= 352800 or wave.shape[1] % 441:
        raise ValueError("probe requires a complete-hop single window, 1323..352800 samples")
    start = time.perf_counter()
    model, _, sources = load_model()
    if not any(s.get("sha256") == sources[0]["sha256"] for s in manifest["sources"]):
        raise ValueError("fixture/checkpoint mismatch")
    kwargs = model.stft_kwargs
    wrapped = SpectralNetwork(model).eval().requires_grad_(False)
    load_seconds = time.perf_counter() - start
    spectrum, features = analyze(wave, kwargs)
    with torch.inference_mode():
        original_mask = wrapped(features).numpy()
    original_checks = verify(reconstruct(original_mask, spectrum, wave, kwargs, audio), expected)
    print("Original spectral wrapper passes complete waveform gates", flush=True)
    trace.write(json.dumps({"event": "original_wrapper_passed", "checks": original_checks,
                           "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                           "script_sha256": sha256(Path(__file__))}) + "\n")
    trace.flush()
    core = ov.Core()
    if args.device not in core.available_devices:
        raise ValueError(f"{args.device} unavailable: {core.available_devices}")
    start = time.perf_counter()
    if args.ir:
        converted = core.read_model(str(args.ir))
        if list(converted.input(0).shape) != list(features.shape):
            raise ValueError("saved IR shape differs; convert and validate this window separately")
    else:
        with torch.inference_mode():
            converted = ov.convert_model(wrapped, example_input=(features,))
        converted.reshape(list(features.shape))
    conversion_seconds = time.perf_counter() - start
    print(f"{'Read' if args.ir else 'Converted'} complete spectral network in {conversion_seconds:.3f}s", flush=True)
    if args.save_ir:
        ov.save_model(converted, str(args.save_ir), compress_to_fp16=False)
    settings = {"INFERENCE_PRECISION_HINT": ov.Type.f32, "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": streams.Num(1)}
    if args.device == "CPU":
        settings["INFERENCE_NUM_THREADS"] = args.threads
    if args.compilation_threads is not None:
        settings["COMPILATION_NUM_THREADS"] = args.compilation_threads
    start = time.perf_counter()
    compiled = core.compile_model(converted, args.device, settings)
    request = compiled.create_infer_request()
    compile_seconds = time.perf_counter() - start
    runs = []
    passed = True
    for iteration in range(7):
        print(f"Run {iteration + 1} starting", flush=True)
        start = time.perf_counter()
        spectrum, features = analyze(wave, kwargs)
        analysis_seconds = time.perf_counter() - start
        network_start = time.perf_counter()
        request.infer({0: features.numpy().copy()}, share_inputs=True)
        mask = request.get_output_tensor(0).data.copy()
        network_seconds = time.perf_counter() - network_start
        output = reconstruct(mask, spectrum, wave, kwargs, audio)
        total_seconds = time.perf_counter() - start
        try:
            checks = verify(output, expected)
        except ValueError as error:
            with args.output.with_suffix(".failed.f32").open("xb") as file:
                file.write(output.astype("<f4").tobytes())
            checks = {"failure": str(error)}
            passed = False
        runs.append({"total_seconds": total_seconds, "rtf": total_seconds / (wave.shape[1] / 44100),
                     "analysis_seconds": analysis_seconds, "network_seconds": network_seconds,
                     "reconstruction_seconds": total_seconds - analysis_seconds - network_seconds,
                     "checks": checks})
        trace.write(json.dumps({"iteration": iteration, **runs[-1]}) + "\n")
        trace.flush()
        print(f"Run {iteration + 1}: {total_seconds:.3f}s, passed={passed}", flush=True)
        if not passed:
            break
    kernels = Counter()
    for node in compiled.get_runtime_model().get_ordered_ops():
        info = node.get_rt_info()
        fields = []
        for name in ("layerType", "runtimePrecision", "primitiveType"):
            value = info[name] if name in info else None
            fields.append(str(getattr(value, "value", value)))
        kernels[tuple(fields)] += 1
    supported = {str(name) for name in compiled.get_property("SUPPORTED_PROPERTIES")}
    process = resource.getrusage(resource.RUSAGE_SELF)
    ir_path = args.ir or args.save_ir
    report = {"backend": f"OpenVINO {ov.__version__}", "device": args.device,
              "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
              "host_threads": args.threads,
              "requested_settings": {name: str(value) for name, value in settings.items()},
              "threads": args.threads if args.device == "CPU" else None,
              "reused_ir": bool(args.ir),
              "ir_sha256": sha256(ir_path) if ir_path else None,
              "ir_weights_sha256": sha256(ir_path.with_suffix(".bin")) if ir_path else None,
              "process": {"peak_rss_kib_linux": process.ru_maxrss,
                          "user_seconds": process.ru_utime, "system_seconds": process.ru_stime},
              "passed": passed, "mode": "single_window_pcm" if audio else "raw_window_pcm",
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": sha256(Path(__file__)),
              "checkpoint_sha256": sources[0]["sha256"], "shape": list(features.shape),
              "model_load_seconds": load_seconds, "conversion_seconds": conversion_seconds,
              "compile_seconds": compile_seconds, "original_wrapper_checks": original_checks,
              "first_execution": runs[0], "warmup": runs[1] if len(runs) > 1 else None,
              "warm": runs[2:] if passed else [], "runs": runs,
              "actual_settings": {name: str(compiled.get_property(name)) for name in (
                  "NUM_STREAMS", "INFERENCE_PRECISION_HINT", "INFERENCE_NUM_THREADS",
                  "ENABLE_CPU_PINNING", "SCHEDULING_CORE_TYPE", "ENABLE_HYPER_THREADING",
                  "EXECUTION_DEVICES") if name in supported},
              "kernels": [{"layer_type": key[0], "runtime_precision": key[1],
                           "execution_type": key[2], "count": count} for key, count in kernels.items()],
              "timing": "single-window PCM including baseline FP64 STFT, synchronous full FP32 network, FP32 ISTFT and copies; excludes model load, conversion, compilation, fixture I/O, encoding and verification"}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    trace.close()
    if not passed:
        raise SystemExit("complete OpenVINO PCM failed gate; failed output retained")
    print(f"Complete PCM median: {statistics.median(r['total_seconds'] for r in runs[2:]):.3f}s", flush=True)


if __name__ == "__main__":
    main()
