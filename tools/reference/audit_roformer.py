"""Validate the 1296 checkpoint against a pinned BS-RoFormer source snapshot."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True

import torch
import yaml

from audit_vr import ROOT, sha256


class ConfigLoader(yaml.SafeLoader):
    pass


# The published config uses tuple tags for frequency bands and loss window sizes.
# Allow only this data tag, without enabling arbitrary Python YAML constructors.
ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))


def reference_class():
    source = json.loads((ROOT / "references/roformer-source.json").read_text())
    for entry in source["files"]:
        if sha256(ROOT / entry["local"]) != entry["sha256"]:
            raise ValueError(f"reference source changed: {entry['local']}")
    directory = ROOT / ".local/roformer-reference"
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    # Load exact source modules without importing the reference application's UI,
    # codecs, or full separator package and its unrelated runtime dependencies.
    load("audio_separator.separator.uvr_lib_v5.device_utils", directory / "device_utils.py")
    package = types.ModuleType("uvr_reference_roformer")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    model_module = load("uvr_reference_roformer.bs_roformer", directory / "bs_roformer.py")
    return model_module.BSRoformer, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.samples <= 1024:
        parser.error("threads must be positive and samples must exceed 1024 for reflection padding")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT / "references/targets.json").read_text())
    asset = next(m for m in manifest["models"] if m["file"].endswith("12.9628.ckpt"))
    weights = ROOT / "models" / asset["file"]
    if weights.stat().st_size != asset["size_bytes"]:
        raise ValueError("incomplete or unexpected checkpoint size")
    config_ref = next(ref for ref in manifest["references"] if ref["local"].endswith("12.9628.yaml"))
    config_path = ROOT / config_ref["local"]
    if sha256(config_path) != config_ref["sha256"]:
        raise ValueError("published config checksum mismatch")
    config = yaml.load(config_path.read_text(), Loader=ConfigLoader)
    model_class, source = reference_class()
    with torch.device("meta"):
        model = model_class(**config["model"])
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    # Nonpersistent rotary caches created on meta are populated by the reference
    # rotary implementation on the first real device call.
    model.eval()
    tensors = []
    for name, tensor in state.items():
        if tensor.layout != torch.strided or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid tensor: {name}")
        tensors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "elements": tensor.numel()})
    result = {"file": weights.name, "size_bytes": asset["size_bytes"], "sha256": sha256(weights),
              "config_sha256": config_ref["sha256"], "source_commit": source["commit"],
              "source_manifest_sha256": sha256(ROOT / "references/roformer-source.json"),
              "torch": torch.__version__, "threads": args.threads, "strict_load": True,
              "tensor_count": len(tensors), "elements": sum(t["elements"] for t in tensors),
              "dtypes": dict(Counter(t["dtype"] for t in tensors)), "tensors": tensors}
    print(f"{weights.name}: {len(tensors)} tensors, strict loading passed", flush=True)
    if args.forward:
        t = torch.arange(args.samples, dtype=torch.float32)
        audio = torch.stack((torch.sin(t * 0.073), torch.cos(t * 0.117))).unsqueeze(0) * 0.2
        with torch.inference_mode():
            output = model(audio)
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite waveform output")
        expected_length = args.samples // config["model"]["stft_hop_length"] * config["model"]["stft_hop_length"]
        if list(output.shape) != [1, 2, expected_length]:
            raise ValueError(f"unexpected waveform shape: {list(output.shape)}")
        result["forward"] = {"input_shape": list(audio.shape), "output_shape": list(output.shape),
                             "peak": float(output.abs().max()), "flash_attn": config["model"]["flash_attn"]}
        print(f"waveform output {list(output.shape)}, finite", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
