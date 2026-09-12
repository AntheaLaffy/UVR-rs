"""Measure complete PCM separation using the pinned VR reference and audio gates."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import warnings

import numpy as np
import torch

from audit_vr import ROOT, sha256
from generate_vr_dsp import analyze, config, reference
from generate_vr_model import load_model
from verify_vr_cli import load_tensor


def separate(model, ref, mp, modern, wave, rate, window):
    start = time.perf_counter()
    spectrum, output_samples = analyze(ref, mp, modern, wave, rate)
    magnitude = np.abs(spectrum)
    peak = magnitude.max()
    analysis_seconds = time.perf_counter() - start
    network_start = time.perf_counter()
    windows = []
    if peak == 0:
        result = np.zeros((2, 2, output_samples), dtype=np.float32)
        return result, {"analysis": analysis_seconds, "network": 0.0,
                        "reconstruction": 0.0}, windows
    frames = spectrum.shape[-1]
    roi = window - 2 * model.offset
    padded = np.pad(magnitude, ((0, 0), (0, 0), (model.offset, roi - frames % roi + model.offset)))
    padded /= padded.max()
    patches = (padded.shape[-1] - 2 * model.offset) // roi
    masks = []
    for patch in range(patches):
        window_start = time.perf_counter()
        x = torch.from_numpy(padded[:, :, patch * roi:patch * roi + window][None].copy())
        masks.append(model.predict_mask(x)[0].numpy())
        windows.append(time.perf_counter() - window_start)
    mask = np.concatenate(masks, axis=2)[:, :, :frames]
    network_seconds = time.perf_counter() - network_start
    reconstruction_start = time.perf_counter()
    phase = np.exp(1.j * np.angle(spectrum))
    primary = ref.cmb_spectrogram_to_wave(
        mask * magnitude * phase, mp, is_v51_model=modern)[:, :output_samples]
    residual = ref.cmb_spectrogram_to_wave(
        (1 - mask) * magnitude * phase, mp, is_v51_model=modern)[:, :output_samples]
    result = np.stack([primary, residual]).astype(np.float32)
    return result, {"analysis": analysis_seconds, "network": network_seconds,
                    "reconstruction": time.perf_counter() - reconstruction_start}, windows


def verify(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("audio output shape or finiteness failure")
    checks = []
    for output, target in zip(actual, expected, strict=True):
        delta = output.astype(np.float64) - target.astype(np.float64)
        if np.any(np.abs(delta) > 2e-4 + 2e-3 * np.abs(target.astype(np.float64))):
            raise ValueError("audio sample tolerance exceeded")
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        reference_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
        if rmse > 1e-3 * max(reference_rms, 1e-4):
            raise ValueError("audio RMS tolerance exceeded")
        checks.append({"max_absolute_error": float(np.abs(delta).max()),
                       "rmse": rmse, "reference_rms": reference_rms})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="local_audio")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.output.exists():
        parser.error("output already exists")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    variant = manifest["audio"]["variant"]
    case = next(case for case in manifest["cases"] if case["name"] == args.case)
    params = manifest["audio"]["cases"][args.case]
    data = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    start = time.perf_counter()
    model, audited, _ = load_model(ROOT / f"benchmarks/artifacts/vr-reference/{variant}.json")
    mp, config_path = config(variant)
    if (sha256(config_path) != audited["config_sha256"]
            or not any(source.get("sha256") == audited["sha256"] for source in manifest["sources"])):
        raise ValueError("fixture and model/config identities differ")
    ref = reference()
    load_seconds = time.perf_counter() - start
    runs = []
    with torch.inference_mode():
        for iteration in range(7):
            start = time.perf_counter()
            output, stages, windows = separate(model, ref, mp, variant == "deecho", data,
                                               params["sample_rate"], params["window_frames"])
            total_seconds = time.perf_counter() - start
            if output.shape != (2, 2, params["output_samples"]) or len(windows) != params["patches"]:
                raise ValueError("audio length or window count differs from fixture")
            stems = verify(output, expected)
            if args.case == "silence" and np.any(output != 0):
                raise ValueError("silence output must be exactly zero")
            runs.append({"total_seconds": total_seconds,
                         "rtf": total_seconds / (params["output_samples"] / 44100),
                         "single_execution_seconds": stages, "window_seconds": windows,
                         "stems": stems})
            print(f"{variant}/{args.case} run {iteration + 1}: {total_seconds:.3f}s, gates passed",
                  flush=True)
    report = {"backend": f"PyTorch {torch.__version__}", "threads": args.threads,
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "parameters": params,
              "script_sha256": sha256(Path(__file__)), "model_load_seconds": load_seconds,
              "timing": "complete PCM separation, excluding model load, fixture I/O, encoding and verification",
              "warmup_executions": 1, "first_execution": runs[0], "warmup": runs[1],
              "warm": runs[2:]}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Reference PCM median: {statistics.median(r['total_seconds'] for r in runs[2:]):.3f}s",
          flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pass .* as keyword args", category=FutureWarning)
        warnings.filterwarnings("ignore", message="n_fft=.* is too small for input signal")
        main()
