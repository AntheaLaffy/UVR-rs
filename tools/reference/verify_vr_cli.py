"""Exercise the compiled Rust CLI; Python only prepares and compares verification data."""

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import threading

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def load_tensor(root, record):
    raw = (root / record["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("fixture digest mismatch")
    return np.frombuffer(raw, dtype="<f4").reshape(record["shape"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", default="signal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancel-check", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--backend", choices=("burn", "openvino-cpu"), default="burn")
    parser.add_argument("--openvino-lib", type=Path, help="native runtime directory for the optional CPU backend")
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.threads <= 0:
        parser.error("timeout and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    case = next(c for c in manifest["cases"] if c["name"] == args.case)
    params = manifest["audio"]["cases"][args.case]
    variant = manifest["audio"]["variant"]
    if args.backend == "openvino-cpu" and (variant != "1296" or args.openvino_lib is None):
        parser.error("OpenVINO CPU requires 1296 fixtures and --openvino-lib")
    source = next(s for s in manifest["sources"] if s.get("file", "").endswith((".pth", ".ckpt")))
    waveform = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    audio = args.output / "音频 input.wav"
    sf.write(audio, waveform.T, params["sample_rate"], subtype="FLOAT")
    binary = ROOT / "target/release/uvr"
    environment = {"PATH": "/nonexistent", "RAYON_NUM_THREADS": str(args.threads)}
    if args.backend == "openvino-cpu":
        environment["LD_LIBRARY_PATH"] = str(args.openvino_lib.resolve())
    destination = args.output / "outputs"
    prefix = [str(binary), "separate-1296"] if variant == "1296" else [str(binary), "separate-vr", variant]
    suffix = ["--backend", args.backend] if variant == "1296" else []
    def command_for(directory):
        return prefix + [str(ROOT / "models" / source["file"]), str(audio), str(directory)] + suffix
    command = command_for(destination)
    result = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=args.timeout_seconds)
    (args.output / "cli.stdout").write_text(result.stdout)
    (args.output / "cli.stderr").write_text(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    checks, paths = [], []
    stems = ["vocals", "instrumental"] if variant == "1296" else ["primary", "residual"]
    for index, stem in enumerate(stems):
        path = destination / f"{audio.stem}_{variant}_{stem}.wav"
        paths.append(path)
        actual, rate = sf.read(path, dtype="float32", always_2d=True)
        actual = actual.T
        if rate != 44100 or actual.shape != expected[index].shape or not np.isfinite(actual).all():
            raise ValueError("CLI output format mismatch")
        delta = actual.astype(np.float64) - expected[index].astype(np.float64)
        rmse = float(np.sqrt(np.mean(delta**2)))
        reference_rms = float(np.sqrt(np.mean(expected[index].astype(np.float64)**2)))
        if np.any(np.abs(delta) > 2e-4 + 2e-3 * np.abs(expected[index])) or rmse > 1e-3 * max(reference_rms, 1e-4):
            raise ValueError(f"{stem}: audio tolerance exceeded")
        checks.append({"stem": stem, "max_absolute_error": float(np.max(np.abs(delta))), "rmse": rmse,
                       "reference_rms": reference_rms, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    before = [path.read_bytes() for path in paths]
    repeated = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
    if repeated.returncode != 1 or [path.read_bytes() for path in paths] != before:
        raise ValueError("CLI overwrote an existing output")
    print(f"{variant}: CLI WAV outputs match reference; overwrite refusal passed", flush=True)
    cancel_seconds = None
    if args.cancel_check:
        cancelled_destination = args.output / "cancelled"
        cancel_command = command_for(cancelled_destination)
        process = subprocess.Popen(cancel_command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        watchdog = threading.Timer(120, lambda: process.kill() if process.poll() is None else None)
        watchdog.start()
        log = []
        try:
            for line in process.stderr:
                log.append(line)
                if args.backend == "openvino-cpu":
                    entered_inference = "窗内 2/5" in line
                else:
                    entered_inference = "窗内 96/" in line if variant == "1296" else "推理：0/" in line
                if entered_inference:
                    if args.backend == "openvino-cpu":
                        # Allow feature packing/submission to finish after the STFT progress message.
                        # The core probe independently cancels from a native wait callback.
                        time.sleep(0.25)
                    start = time.monotonic()
                    process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=90)
                    cancel_seconds = time.monotonic() - start
                    log.append(stderr)
                    if process.returncode != 130 or stdout or list(cancelled_destination.iterdir()):
                        raise ValueError("cancelled task published output or returned wrong status")
                    break
            if cancel_seconds is None:
                raise ValueError("CLI never entered inference before cancellation")
        finally:
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
                process.wait()
            (args.output / "cancel.stderr").write_text("".join(log))
        print(f"{variant}: Ctrl-C cancellation passed in {cancel_seconds:.3f}s", flush=True)
    report = {"variant": variant, "case": args.case, "mode": "verification_only", "checks": checks, "backend": args.backend,
              "fixture_sha256": hashlib.sha256(raw).hexdigest(), "cli_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "cancel_seconds": cancel_seconds, "runtime_environment": environment, "overwrite_refusal": True,
              "command": command, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
