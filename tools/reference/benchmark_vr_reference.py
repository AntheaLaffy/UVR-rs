"""Time the fixed PyTorch reference on the same mask fixtures as the Rust probe."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from generate_vr_model import load_model
from verify_vr_cli import load_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="magnitude")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    case = next(case for case in manifest["cases"] if case["name"] == args.case)
    data = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    start = time.perf_counter()
    model, audited, _ = load_model(args.audit)
    load_seconds = time.perf_counter() - start
    if audited["sha256"] != manifest["sources"][0]["sha256"]:
        raise ValueError("fixture and model identities differ")
    samples = []
    max_error = 0.0
    with torch.inference_mode():
        for _ in range(7):
            start = time.perf_counter()
            # Like the Rust probe, include input creation and output materialization.
            output = model.predict_mask(torch.from_numpy(data.copy())).contiguous().numpy().copy()
            samples.append(time.perf_counter() - start)
            if output.shape != expected.shape or not np.isfinite(output).all():
                raise ValueError("reference output shape or finiteness failure")
            error = np.abs(output.astype(np.float64) - expected.astype(np.float64))
            if np.any(error > 1e-3 + 1e-4 * np.abs(expected)):
                raise ValueError("reference output tolerance exceeded")
            max_error = max(max_error, float(error.max()))
    report = {"backend": f"PyTorch {torch.__version__}", "threads": args.threads,
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "shape": list(data.shape),
              "model_load_seconds": load_seconds, "first_execution_seconds": samples[0],
              "warmup_executions": 1, "warm_seconds": samples[2:], "max_absolute_error": max_error}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Reference median: {statistics.median(samples[2:]):.3f}s ({args.threads} threads)", flush=True)


if __name__ == "__main__":
    main()
