"""Inspect CPU execution plans; profiled timings are not acceptance benchmarks."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import time

import openvino as ov
from openvino.properties import streams
import torch

from audit_vr import sha256
from benchmark_roformer_openvino_audio import analyze, reconstruct
from benchmark_vr_audio_reference import verify
from verify_vr_cli import load_tensor


def memory():
    fields = ("VmRSS", "VmHWM", "VmSize")
    return {line.split(":")[0]: line.split(":")[1].strip()
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.split(":")[0] in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--case", default="local_audio")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1 or args.output.exists():
        parser.error("positive thread count and new output required")
    graph_path = args.output.with_suffix(".runtime.xml")
    if graph_path.exists() or graph_path.with_suffix(".bin").exists():
        parser.error("runtime graph output already exists")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest_path = args.fixtures / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not any("float64" in source.get("stft_precision", "") for source in manifest["sources"]):
        raise ValueError("baseline FP64 forward STFT fixture required")
    case = next(case for case in manifest["cases"] if case["name"] == args.case)
    wave = load_tensor(args.fixtures, manifest["tensors"][case["input"]]).copy()
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    audio = "audio" in manifest
    if audio:
        params = manifest["audio"]["cases"][args.case]
        if params["sample_rate"] != 44100 or params["windows"] != 1:
            raise ValueError("single-window 44100 Hz fixture required")
    else:
        expected = expected[None]
    if wave.shape[0] == 1:
        wave = wave.repeat(2, axis=0)
    if wave.shape[0] != 2 or not 1323 <= wave.shape[1] <= 352800 or wave.shape[1] % 441:
        raise ValueError("complete-hop stereo window required")
    provenance_path = args.ir.with_suffix(".provenance.json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if (provenance["xml_sha256"] != sha256(args.ir)
                or provenance["bin_sha256"] != sha256(args.ir.with_suffix(".bin"))
                or not any(source.get("sha256") == provenance["checkpoint_sha256"]
                           for source in manifest["sources"])):
            raise ValueError("IR provenance mismatch")
    kwargs = dict(n_fft=2048, hop_length=441, win_length=2048, normalized=False)
    spectrum, features = analyze(wave, kwargs)
    core = ov.Core()
    start = time.perf_counter()
    model = core.read_model(str(args.ir))
    load_seconds = time.perf_counter() - start
    if list(model.input(0).shape) != list(features.shape):
        raise ValueError("IR/input shape mismatch")
    settings = {"INFERENCE_PRECISION_HINT": ov.Type.f32, "PERFORMANCE_HINT": "LATENCY",
                "NUM_STREAMS": streams.Num(1), "INFERENCE_NUM_THREADS": args.threads,
                "PERF_COUNT": True}
    snapshots = {"after_read": memory()}
    print(f"Read graph in {load_seconds:.3f}s: {snapshots['after_read']}", flush=True)
    start = time.perf_counter()
    compiled = core.compile_model(model, "CPU", settings)
    snapshots["after_compile"] = memory()
    request = compiled.create_infer_request()
    snapshots["after_request"] = memory()
    compile_seconds = time.perf_counter() - start
    print(f"Compiled in {compile_seconds:.3f}s: {snapshots['after_request']}", flush=True)
    runs = []
    for iteration in range(2):
        start = time.perf_counter()
        request.infer({0: features.numpy()}, share_inputs=True)
        seconds = time.perf_counter() - start
        mask = request.get_output_tensor(0).data.copy()
        checks = verify(reconstruct(mask, spectrum, wave, kwargs, audio), expected)
        nodes = [{"name": item.node_name, "type": item.node_type, "exec_type": item.exec_type,
                  "status": str(item.status), "real_us": item.real_time.total_seconds() * 1e6,
                  "cpu_us": item.cpu_time.total_seconds() * 1e6}
                 for item in request.get_profiling_info()]
        totals = defaultdict(float)
        for node in nodes:
            totals[node["type"] + "/" + node["exec_type"]] += node["real_us"]
        runs.append({"iteration": iteration, "profiled_network_seconds": seconds,
                     "checks": checks, "nodes": nodes,
                     "type_microseconds": dict(sorted(totals.items(), key=lambda item: -item[1]))})
        print(f"Profiled run {iteration + 1}: {seconds:.3f}s, waveform passed", flush=True)
    runtime = compiled.get_runtime_model()
    ov.serialize(runtime, str(graph_path))
    nodes = []
    for node in runtime.get_ordered_ops():
        nodes.append({"name": node.get_friendly_name(),
                      "inputs": [str(port.get_partial_shape()) for port in node.inputs()],
                      "outputs": [str(port.get_partial_shape()) for port in node.outputs()],
                      "info": {key: str(getattr(value, "value", value))
                               for key, value in node.get_rt_info().items()}})
    snapshots["after_infer"] = memory()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {"scope": "Two diagnostic executions with per-node profiling; not hot-run acceptance",
              "counter_semantics": "OpenVINO CPU node counters are cumulative means across executions; network wall times are per call",
              "openvino": ov.__version__, "ir_sha256": sha256(args.ir),
              "bin_sha256": sha256(args.ir.with_suffix(".bin")),
              "script_sha256": sha256(Path(__file__)), "manifest_sha256": sha256(manifest_path),
              "shape": list(features.shape), "load_seconds": load_seconds,
              "compile_seconds": compile_seconds, "memory_snapshots": snapshots,
              "max_rss_kib": usage.ru_maxrss, "user_seconds": usage.ru_utime,
              "system_seconds": usage.ru_stime, "runs": runs, "runtime_nodes": nodes,
              "runtime_types": dict(Counter(node["info"].get("layerType") for node in nodes)),
              "actual_settings": {key: str(compiled.get_property(key)) for key in (
                  "NUM_STREAMS", "INFERENCE_NUM_THREADS", "ENABLE_CPU_PINNING",
                  "ENABLE_HYPER_THREADING", "SCHEDULING_CORE_TYPE", "INFERENCE_PRECISION_HINT")}}
    with args.output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
