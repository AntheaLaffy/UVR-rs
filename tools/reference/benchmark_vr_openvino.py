"""Explore OpenVINO kernels on audited VR masks; development only, not a product runtime."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import openvino as ov
import torch

from audit_vr import sha256
from generate_vr_model import load_model
from verify_vr_cli import load_tensor


class PredictMask(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict_mask(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="magnitude")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--save-ir", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("threads must be positive and output must not exist")
    if args.save_ir and (args.save_ir.exists() or args.save_ir.with_suffix(".bin").exists()):
        parser.error("IR output must not exist")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    data = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    start = time.perf_counter()
    model, audited, _ = load_model(args.audit)
    model_load_seconds = time.perf_counter() - start
    if not any(source.get("sha256") == audited["sha256"] for source in manifest["sources"]):
        raise ValueError("fixture and checkpoint identities differ")
    wrapped = PredictMask(model).eval().requires_grad_(False)
    start = time.perf_counter()
    with torch.inference_mode():
        converted = ov.convert_model(wrapped, example_input=(torch.from_numpy(data.copy()),))
    converted.reshape(list(data.shape))
    conversion_seconds = time.perf_counter() - start
    print(f"Converted {list(data.shape)} in {conversion_seconds:.3f}s", flush=True)
    if args.save_ir:
        ov.save_model(converted, str(args.save_ir), compress_to_fp16=False)
    core = ov.Core()
    settings = {"INFERENCE_PRECISION_HINT": ov.Type.f32,
                "PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1}
    if args.device.startswith("CPU"):
        settings["INFERENCE_NUM_THREADS"] = args.threads
    start = time.perf_counter()
    compiled = core.compile_model(converted, args.device, settings)
    request = compiled.create_infer_request()
    compile_seconds = time.perf_counter() - start
    times = []
    checks = []
    for iteration in range(7):
        start = time.perf_counter()
        # Include caller-owned input creation and output materialization, as in Rust.
        request.infer({0: data.copy()}, share_inputs=True)
        output = request.get_output_tensor(0).data.copy()
        times.append(time.perf_counter() - start)
        if output.shape != expected.shape or not np.isfinite(output).all():
            raise ValueError("OpenVINO output shape or finiteness failure")
        delta = output.astype(np.float64) - expected.astype(np.float64)
        failures = int(np.count_nonzero(np.abs(delta) > 1e-3 + 1e-4 * np.abs(expected)))
        checks.append({"max_absolute_error": float(np.max(np.abs(delta))),
                       "rmse": float(np.sqrt(np.mean(delta ** 2))),
                       "failed_elements": failures})
        print(f"Run {iteration + 1}: {times[-1]:.3f}s, {failures} elements outside gate", flush=True)
        if failures:
            with args.output.with_suffix(".failed.f32").open("xb") as file:
                file.write(output.astype("<f4").tobytes())
            break
    kernels = Counter()
    for node in compiled.get_runtime_model().get_ordered_ops():
        info = node.get_rt_info()
        fields = []
        for name in ("layerType", "runtimePrecision", "primitiveType"):
            value = info[name] if name in info else None
            fields.append(str(getattr(value, "value", value)))
        kernels[tuple(fields)] += 1
    passed = not any(c["failed_elements"] for c in checks)
    report = {"backend": f"OpenVINO {ov.__version__}", "device": args.device,
              "available_devices": core.available_devices, "threads": args.threads,
              "settings": {key: str(value) for key, value in settings.items()},
              "actual_settings": {name: str(compiled.get_property(name)) for name in
                                  ("NUM_STREAMS", "INFERENCE_PRECISION_HINT", "PERFORMANCE_HINT")
                                  + (("INFERENCE_NUM_THREADS", "ENABLE_CPU_PINNING",
                                      "SCHEDULING_CORE_TYPE", "ENABLE_HYPER_THREADING")
                                     if args.device.startswith("CPU") else ())},
              "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "checkpoint_sha256": audited["sha256"], "script_sha256": sha256(Path(__file__)),
              "model_load_seconds": model_load_seconds, "conversion_seconds": conversion_seconds,
              "compile_seconds": compile_seconds, "passed": passed,
              "shape": list(data.shape), "first_execution_seconds": times[0],
              "warmup_executions": 1, "warm_seconds": times[2:] if passed else [],
              "checks": checks, "absolute_tolerance": 1e-3, "relative_tolerance": 1e-4,
              "timing": "synchronous inference including input creation and output copy; excludes conversion, compilation, fixture I/O and verification",
              "kernels": [{"layer_type": k[0], "runtime_precision": k[1],
                           "execution_type": k[2], "count": count} for k, count in kernels.items()]}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("OpenVINO did not pass the mask gate; report and failed output retained")
    print(f"OpenVINO median: {statistics.median(times[2:]):.3f}s", flush=True)


if __name__ == "__main__":
    main()
