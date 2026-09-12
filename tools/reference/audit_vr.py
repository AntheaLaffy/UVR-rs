"""Validate requested VR weights against the pinned upstream networks; development only."""

import argparse
import ast
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

sys.dont_write_bytecode = True

import torch

ROOT = Path(__file__).resolve().parents[2]
VR_FILES = ["5_HP-Karaoke-UVR.pth", "6_HP-Karaoke-UVR.pth", "UVR-DeEcho-DeReverb.pth"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def upstream_networks(commit):
    upstream = ROOT / "upstream"
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("upstream commit differs from references/targets.json")
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", commit, "--", "lib_v5"],
                   check=True, stdout=subprocess.DEVNULL)
    # Networks only use crop_center from spec_utils. Extract that exact function
    # to avoid unrelated audio codec and UI imports during tensor-only validation.
    source = upstream / "lib_v5/spec_utils.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "crop_center")
    module = types.ModuleType("lib_v5.spec_utils")
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), module.__dict__)
    sys.path.insert(0, str(upstream))
    sys.modules["lib_v5.spec_utils"] = module
    return importlib.import_module("lib_v5.vr_network.nets"), importlib.import_module("lib_v5.vr_network.nets_new")


def tensor_shapes(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(item) for item in value]
    return None


def audit(path, asset, metadata, networks, forward):
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"{path.name}: incomplete file or unexpected release size")
    digest = sha256(path)
    with path.open("rb") as stream:
        stream.seek(max(0, asset["size_bytes"] - 10000 * 1024))
        model_id = hashlib.file_digest(stream, "md5").hexdigest()
    params = metadata[model_id]
    config_path = ROOT / "upstream/lib_v5/vr_network/modelparams" / (params["vr_model_param"] + ".json")
    config = json.loads(config_path.read_text())
    bins = config.get("n_bins", config.get("bins"))
    architecture_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    file_kib = (asset["size_bytes"] + 1023) // 1024
    architecture = min(architecture_sizes, key=lambda value: abs(value - file_kib))
    nets, nets_new = networks
    with torch.device("meta"):
        if architecture in [56817, 218409] or ("nout" in params and "nout_lstm" in params):
            model = nets_new.CascadedNet(bins * 2, architecture,
                                        nout=params.get("nout", 32), nout_lstm=params.get("nout_lstm", 128))
        else:
            model = nets.determine_model_capacity(bins * 2, architecture)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(t, torch.Tensor) for t in state.values()):
        raise ValueError("expected a tensor-only state dictionary")
    # Strict loading, not size-based dispatch, proves keys and shapes fit the network.
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    tensors = []
    for name, tensor in state.items():
        if tensor.layout != torch.strided or not torch.isfinite(tensor).all():
            raise ValueError(f"unexpected tensor layout or nonfinite values: {name}")
        tensors.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "elements": tensor.numel()})
    result = {"file": path.name, "size_bytes": asset["size_bytes"], "sha256": digest, "uvr_md5": model_id,
              "metadata": params, "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path),
              "architecture_size_kib": architecture, "network": type(model).__name__, "offset": model.offset,
              "tensor_count": len(tensors), "elements": sum(t["elements"] for t in tensors),
              "dtypes": dict(Counter(t["dtype"] for t in tensors)), "strict_load": True, "tensors": tensors}
    if forward:
        # A deterministic magnitude input checks network execution, not audio quality.
        width = 320
        x = torch.arange(2 * (bins + 1) * width, dtype=torch.float32).reshape(1, 2, bins + 1, width)
        x = (x.remainder(101) + 1) / 102.0
        shapes, handles = {}, []
        def observe(name):
            def hook(_module, inputs, output):
                shapes[name] = {"input": tensor_shapes(inputs), "output": tensor_shapes(output)}
            return hook
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.LSTM, torch.nn.Linear)):
                handles.append(module.register_forward_hook(observe(name)))
        with torch.inference_mode():
            mask = model.predict_mask(x)
        for handle in handles:
            handle.remove()
        expected = [1, 2, bins + 1, width - 2 * model.offset]
        if list(mask.shape) != expected or not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError("mask shape, range, or finiteness check failed")
        result["forward"] = {"input_shape": list(x.shape), "mask_shape": list(mask.shape),
                             "min": float(mask.min()), "max": float(mask.max()), "operator_shapes": shapes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=VR_FILES, action="append")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT / "references/targets.json").read_text())
    metadata_ref = next(ref for ref in manifest["references"] if ref["local"].endswith("vr_model_data_new.json"))
    metadata_path = ROOT / metadata_ref["local"]
    if sha256(metadata_path) != metadata_ref["sha256"]:
        raise ValueError("model metadata checksum mismatch")
    metadata = json.loads(metadata_path.read_text())
    networks = upstream_networks(manifest["uvr_commit"])
    report = {"torch": torch.__version__, "threads": args.threads,
              "manifest_sha256": sha256(ROOT / "references/targets.json"), "models": []}
    for name in args.model or VR_FILES:
        asset = next(asset for asset in manifest["models"] if asset["file"] == name)
        result = audit(args.weights_dir / name, asset, metadata, networks, args.forward)
        report["models"].append(result)
        print(f"{name}: {result['tensor_count']} tensors, strict loading passed", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
