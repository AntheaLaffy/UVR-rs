"""Measure original 1296 attention/feed-forward modules on OpenVINO FP32."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import openvino as ov
from openvino.properties import streams
import torch

from audit_vr import sha256
from generate_roformer_model import load_model
from verify_vr_cli import load_tensor


def check(output, expected):
    if output.shape != expected.shape or not np.isfinite(output).all():
        raise ValueError("module output shape or finiteness failure")
    delta = output.astype(np.float64) - expected.astype(np.float64)
    return {"max_absolute_error": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "failed_elements": int(np.count_nonzero(
                np.abs(delta) > 1e-3 + 1e-4 * np.abs(expected)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--compilation-threads", type=int)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("threads must be positive and output must not exist")
    if args.compilation_threads is not None and args.compilation_threads < 1:
        parser.error("compilation threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    start = time.perf_counter()
    model, _, sources = load_model()
    model.requires_grad_(False)
    if not any(s.get("sha256") == sources[0]["sha256"] for s in manifest["sources"]):
        raise ValueError("fixture/checkpoint mismatch")
    modules = {
        "1296-attention-time": model.layers[0][0].layers[0][0],
        "1296-attention-freq": model.layers[0][1].layers[0][0],
        "1296-attention-zero": model.layers[0][1].layers[0][0],
        "1296-feed-forward-time": model.layers[0][0].layers[0][1],
    }
    del model
    load_seconds = time.perf_counter() - start
    core = ov.Core()
    settings = {"INFERENCE_PRECISION_HINT": ov.Type.f32,
                "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": streams.Num(1)}
    if args.device == "CPU":
        settings["INFERENCE_NUM_THREADS"] = args.threads
    if args.compilation_threads is not None:
        settings["COMPILATION_NUM_THREADS"] = args.compilation_threads
    if args.device not in core.available_devices:
        raise ValueError(f"{args.device} unavailable: {core.available_devices}")
    measurements = []
    passed = True
    for case in manifest["cases"]:
        if case["name"] not in modules:
            continue
        data = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
        expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
        module = modules[case["name"]]
        # Populate the original rotary module's nonpersistent caches before tracing.
        with torch.inference_mode():
            original = module(torch.from_numpy(data.copy())).numpy()
        original_check = check(original, expected)
        if original_check["failed_elements"]:
            raise ValueError("original module differs from fixture")
        start = time.perf_counter()
        with torch.inference_mode():
            converted = ov.convert_model(module, example_input=(torch.from_numpy(data.copy()),))
        converted.reshape(list(data.shape))
        conversion_seconds = time.perf_counter() - start
        start = time.perf_counter()
        compiled = core.compile_model(converted, args.device, settings)
        request = compiled.create_infer_request()
        compile_seconds = time.perf_counter() - start
        times, checks = [], []
        for iteration in range(7):
            start = time.perf_counter()
            request.infer({0: data.copy()}, share_inputs=True)
            output = request.get_output_tensor(0).data.copy()
            times.append(time.perf_counter() - start)
            checks.append(check(output, expected))
            if checks[-1]["failed_elements"]:
                with args.output.with_suffix(f".{case['name']}.failed.f32").open("xb") as file:
                    file.write(output.astype("<f4").tobytes())
                passed = False
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
        measurements.append({
            "name": case["name"], "shape": list(data.shape),
            "original_module_check": original_check,
            "conversion_seconds": conversion_seconds, "compile_seconds": compile_seconds,
            "first_execution_seconds": times[0], "warmup_executions": 1,
            "warm_seconds": times[2:] if passed else [], "checks": checks,
            "actual_settings": {name: str(compiled.get_property(name)) for name in (
                "NUM_STREAMS", "INFERENCE_PRECISION_HINT", "INFERENCE_NUM_THREADS",
                "ENABLE_CPU_PINNING", "SCHEDULING_CORE_TYPE", "ENABLE_HYPER_THREADING",
                "EXECUTION_DEVICES") if name in supported},
            "kernels": [{"layer_type": key[0], "runtime_precision": key[1],
                         "execution_type": key[2], "count": count}
                        for key, count in kernels.items()],
        })
        if not passed:
            break
        print(f"{case['name']}: {statistics.median(times[2:]):.6f}s, seven checks passed", flush=True)
    if not measurements:
        raise ValueError("no matching module cases")
    report = {"backend": f"OpenVINO {ov.__version__}", "device": args.device,
              "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
              "host_threads": args.threads,
              "threads": args.threads if args.device == "CPU" else None,
              "settings": {name: str(value) for name, value in settings.items()},
              "available_devices": core.available_devices, "passed": passed,
              "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "checkpoint_sha256": sources[0]["sha256"], "script_sha256": sha256(Path(__file__)),
              "model_load_seconds": load_seconds, "absolute_tolerance": 1e-3,
              "relative_tolerance": 1e-4, "measurements": measurements,
              "timing": "synchronous module inference including input/output copies; excludes conversion, compilation, fixture I/O and verification"}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("OpenVINO module failed gate; failure retained")


if __name__ == "__main__":
    main()
