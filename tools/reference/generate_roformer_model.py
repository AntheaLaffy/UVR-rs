"""Capture the complete, unchanged 1296 network for Rust waveform verification."""

import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from audit_roformer import ConfigLoader, reference_class
from audit_vr import ROOT, sha256
from generate_sequence_probe import Fixtures
from roformer_precision import precise_stft


def load_model():
    audit_path = ROOT / "benchmarks/artifacts/vr-reference/1296.json"
    audit = json.loads(audit_path.read_text())
    weights = ROOT / "models" / audit["file"]
    if not audit["strict_load"] or sha256(weights) != audit["sha256"]:
        raise ValueError("1296 weights differ from strict audit")
    targets = json.loads((ROOT / "references/targets.json").read_text())
    config_ref = next(ref for ref in targets["references"] if ref["local"].endswith("12.9628.yaml"))
    config_path = ROOT / config_ref["local"]
    if sha256(config_path) != config_ref["sha256"]:
        raise ValueError("1296 config checksum mismatch")
    config = yaml.load(config_path.read_text(), Loader=ConfigLoader)
    model_class, source = reference_class()
    with torch.device("meta"):
        model = model_class(**config["model"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True, assign=True)
    model.eval()
    sources = [{"file": audit["file"], "sha256": audit["sha256"],
                "audit_sha256": sha256(audit_path), "source_commit": source["commit"],
                "config_sha256": config_ref["sha256"], "strict_tensor_count": len(model.state_dict())},
               {"file": "tools/reference/uv.lock", "sha256": sha256(ROOT / "tools/reference/uv.lock")},
               {"file": "tools/reference/generate_roformer_model.py", "sha256": sha256(Path(__file__))}]
    return model, config, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-window", action="store_true")
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--stft64", action="store_true", help="engineering baseline v1.1 forward STFT")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, config, sources = load_model()
    fixtures = Fixtures(args.output)
    fixtures.sources = sources
    if args.stft64:
        fixtures.sources.append({"stft_precision": "float64 forward, complex64 network input, float32 inverse",
                                 "precision_source_sha256": sha256(ROOT / "tools/reference/roformer_precision.py")})
    if args.audio_file:
        import soundfile as sf
        with sf.SoundFile(args.audio_file) as source:
            if source.samplerate != 44100 or source.channels not in (1, 2):
                raise ValueError("raw network fixture requires 44.1 kHz mono/stereo")
            source.seek(60 * source.samplerate)
            values = source.read(config["audio"]["chunk_size"], dtype="float32", always_2d=True).T.copy()
        if values.shape[0] == 1:
            values = values.repeat(2, axis=0)
        cases = [("local_audio", torch.from_numpy(values))]
        fixtures.sources.append({"local_audio_sha256": sha256(args.audio_file), "start_sample": 60 * 44100,
                                 "samples": values.shape[1], "decoder": "soundfile " + sf.__version__})
    elif args.full_window:
        t = torch.arange(config["audio"]["chunk_size"], dtype=torch.float32)
        cases = [("full_window", torch.stack((torch.sin(t * 0.073), torch.cos(t * 0.117))) * 0.2)]
    else:
        t = torch.arange(4096, dtype=torch.float32)
        signal = torch.stack((torch.sin(t * 0.073), torch.cos(t * 0.117))) * 0.2
        impulses = torch.zeros(2, 1025)
        impulses[0, 0], impulses[0, -1], impulses[1, 517] = 0.75, -0.5, 0.25
        cases = [("signal", signal), ("minimum_impulses", impulses), ("silence", torch.zeros(2, 4410))]
    executions = {}
    for name, audio in cases:
        start = time.perf_counter()
        with torch.inference_mode(), precise_stft(args.stft64):
            output = model(audio.unsqueeze(0)).squeeze(0)
        executions[name] = time.perf_counter() - start
        if list(output.shape) != [2, audio.shape[1] // 441 * 441]:
            raise ValueError("unexpected raw network output shape")
        fixtures.cases.append({"name": name, "input": fixtures.save(name + "-input", audio),
                               "expected": fixtures.save(name + "-expected", output), "steps": []})
        print(f"1296/{name}: {list(output.shape)}, {executions[name]:.3f}s (verification only)", flush=True)
    fixtures.write()
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["network"] = {"variant": "1296", "sample_rate": 44100, "threads": 2,
                           "output_length": "floor(input_samples / 441) * 441",
                           "single_execution_seconds": executions}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
