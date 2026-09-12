"""Run one Rust probe with resource evidence and protection against actual memory pressure."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def available_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is missing")


def process_sample(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
                fields[line.split(":")[0]] = int(line.split()[1])
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        fields.update({"pid": pid, "user_ticks": int(stat[11]), "system_ticks": int(stat[12])})
        return fields
    except FileNotFoundError:
        return None


def process_tree(root):
    pending = [root]
    result = []
    while pending:
        pid = pending.pop()
        sample = process_sample(pid)
        if sample is None:
            continue
        result.append(sample)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except FileNotFoundError:
            children = []
        pending.extend(int(child) for child in children)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", default="local_audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-pinning", choices=("YES", "NO"))
    parser.add_argument("--hyper-threading", choices=("YES", "NO"))
    parser.add_argument("--core-type", choices=("ANY_CORE", "PCORE_ONLY", "ECORE_ONLY"))
    parser.add_argument("--performance", choices=("LATENCY", "THROUGHPUT"), default="LATENCY")
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--inorder-diagnostic", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--min-available-gib", type=float, default=1.5,
                        help="live system headroom protection, not a process RSS acceptance cap")
    args = parser.parse_args()
    if min(args.threads, args.timeout, args.min_available_gib, args.streams, args.batch) <= 0:
        parser.error("thread count, deadline and headroom must be positive")
    if not 1 <= args.batch <= 8:
        parser.error("batch must be 1..8")
    if args.inorder_diagnostic and args.device != "GPU":
        parser.error("queue interception is a GPU diagnostic")
    if args.device != "CPU" and any((args.cpu_pinning, args.hyper_threading, args.core_type)):
        parser.error("CPU scheduling options require CPU")
    output = args.output.resolve()
    suffixes = (".json", ".jsonl", ".time", ".log", ".invocation.json", ".resources.jsonl")
    if output.suffix != ".json" or any(output.with_suffix(suffix).exists() for suffix in suffixes):
        parser.error("new .json output and sidecars required")
    binary = ROOT / "benchmarks/backend-probe/target/release/probe-openvino-audio"
    overrides = {"LD_LIBRARY_PATH": str(ROOT / ".local/openvino-2026.3.1/lib"),
                 "RAYON_NUM_THREADS": str(args.threads), "PATH": "/nonexistent"}
    for name, value in (("UVR_OV_CPU_PINNING", args.cpu_pinning),
                        ("UVR_OV_HYPER_THREADING", args.hyper_threading),
                        ("UVR_OV_CORE_TYPE", args.core_type)):
        if value is not None:
            overrides[name] = value
    overrides["UVR_OV_PERFORMANCE"] = args.performance
    overrides["UVR_OV_STREAMS"] = str(args.streams)
    if args.device == "GPU":
        overrides.update({"OCL_ICD_VENDORS": "/etc/OpenCL/vendors/intel.icd",
                          "UVR_OV_COMPILATION_THREADS": "1"})
    if args.inorder_diagnostic:
        intercept = ROOT / ".local/clintercept-3.0.6/lib"
        overrides.update({"LD_LIBRARY_PATH": str(intercept) + ":" + overrides["LD_LIBRARY_PATH"],
                          "LD_PRELOAD": str(intercept / "libOpenCL.so"),
                          "CLI_OpenCLFileName": "/usr/lib/libOpenCL.so.1", "CLI_InOrderQueue": "1",
                          "CLI_QueueInfoLogging": "1", "CLI_UniqueFiles": "1",
                          "CLI_DumpDir": str(output.with_suffix(".clintercept"))})
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("CLI_", "UVR_OV_"))
                   and key not in ("LD_PRELOAD", "NEOReadDebugKeys", "EnableDirectSubmission")}
    environment.update(overrides)
    command = ["/usr/bin/time", "-v", "-o", str(output.with_suffix(".time")), str(binary),
               args.device, str(args.ir.resolve()), str(args.fixtures.resolve()), args.case, str(output)]
    if args.check_only:
        command.append("--check-only")
    if args.batch != 1:
        command.extend(["--batch", str(args.batch)])
    metadata = {"start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "command": command, "cwd": str(ROOT), "environment_overrides": overrides,
                "binary_sha256": digest(binary), "runner_sha256": digest(Path(__file__)),
                "ir_sha256": digest(args.ir), "bin_sha256": digest(args.ir.with_suffix(".bin")),
                "manifest_sha256": digest(args.fixtures / "manifest.json"),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "resource_interval_seconds": 1, "validation_only": args.check_only,
                "min_available_gib_protection": args.min_available_gib, "timeout_seconds": args.timeout,
                "governors": {path.parent.parent.name: path.read_text().strip() for path in
                              Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")}}
    invocation = output.with_suffix(".invocation.json")
    with invocation.open("x") as file:
        json.dump(metadata, file, indent=2)
    start = time.monotonic()
    reason = None
    lowest_available = available_kib()
    if lowest_available < args.min_available_gib * 1024 ** 2:
        raise RuntimeError("insufficient actual system headroom before launch")
    with output.with_suffix(".log").open("x") as log, output.with_suffix(".resources.jsonl").open("x") as samples:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - start
                available = available_kib()
                lowest_available = min(lowest_available, available)
                samples.write(json.dumps({"elapsed_seconds": elapsed, "available_kib": available,
                                          "processes": process_tree(process.pid)}) + "\n")
                samples.flush()
                if available < args.min_available_gib * 1024 ** 2:
                    reason = "actual system available memory below protection threshold"
                elif elapsed > args.timeout:
                    reason = "experiment deadline exceeded"
                if reason:
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            metadata.update({"exit_code": process.returncode, "stop_reason": reason,
                             "elapsed_seconds": time.monotonic() - start,
                             "lowest_available_kib": lowest_available,
                             "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            invocation.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in (
        "exit_code", "stop_reason", "elapsed_seconds", "lowest_available_kib")}), flush=True)
    if process.returncode or reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
