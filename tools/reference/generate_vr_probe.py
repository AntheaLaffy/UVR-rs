"""Capture real VR activations and weights for isolated Rust CPU backend comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_vr import ROOT, sha256, upstream_networks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    report = json.loads(args.audit.read_text())
    audited = report["models"][0]
    if audited["network"] != "CascadedASPPNet" or not audited["strict_load"]:
        raise ValueError("expected a successful old VR audit")
    weights = ROOT / "models" / audited["file"]
    if sha256(weights) != audited["sha256"]:
        raise ValueError("weights differ from audit")
    sources = json.loads((ROOT / "references/targets.json").read_text())
    nets, _ = upstream_networks(sources["uvr_commit"])
    shape = audited["forward"]["input_shape"]
    with torch.device("meta"):
        model = nets.determine_model_capacity((shape[2] - 1) * 2, audited["architecture_size_kib"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True, assign=True)
    model.eval()
    selected = ["stg1_low_band_net.enc1.conv1.conv", "stg1_low_band_net.enc1.conv2.conv",
                "stg1_low_band_net.aspp.conv3.conv"]
    captured = {}
    def capture(name):
        def hook(_module, inputs, output):
            captured[name] = (inputs[0].detach().contiguous(), output.detach().contiguous())
        return hook
    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture(name)) for name in selected + ["stg1_low_band_net.aspp"]]
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    x = (x.remainder(101) + 1) / 102.0
    with torch.inference_mode():
        model.predict_mask(x)
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    tensors, cases = {}, []
    def save(name, tensor):
        raw = tensor.detach().contiguous().numpy().astype("<f4").tobytes()
        filename = name + ".f32"
        (args.output / filename).write_bytes(raw)
        tensors[name] = {"file": filename, "shape": list(tensor.shape), "sha256": hashlib.sha256(raw).hexdigest()}
        return name
    for index, name in enumerate(selected):
        steps = []
        prefix = f"case{index}"
        for i, layer in enumerate(modules[name]):
            key = f"{prefix}-layer{i}"
            if isinstance(layer, torch.nn.Conv2d):
                assert layer.stride[0] == layer.stride[1] and layer.padding[0] == layer.padding[1]
                assert layer.dilation[0] == layer.dilation[1] and layer.bias is None
                steps.append({"kind": "conv2d", "weight": save(key + "-weight", layer.weight),
                              "stride": layer.stride[0], "padding": layer.padding[0],
                              "dilation": layer.dilation[0], "groups": layer.groups})
            elif isinstance(layer, torch.nn.BatchNorm2d):
                steps.append({"kind": "batch_norm", "weight": save(key + "-weight", layer.weight),
                              "bias": save(key + "-bias", layer.bias), "mean": save(key + "-mean", layer.running_mean),
                              "variance": save(key + "-variance", layer.running_var), "epsilon": layer.eps})
            elif isinstance(layer, torch.nn.LeakyReLU):
                steps.append({"kind": "leaky_relu", "slope": layer.negative_slope})
            elif isinstance(layer, torch.nn.ReLU):
                steps.append({"kind": "leaky_relu", "slope": 0.0})
            else:
                raise ValueError(f"unhandled reference layer {layer}")
        inp, out = captured[name]
        cases.append({"name": name, "input": save(prefix + "-input", inp),
                      "expected": save(prefix + "-expected", out), "steps": steps})
    inp = captured["stg1_low_band_net.aspp"][1]
    for align in [True, False]:
        size = [inp.shape[2] * 2, inp.shape[3] * 2]
        out = F.interpolate(inp, size=size, mode="bilinear", align_corners=align)
        prefix = f"resize-{str(align).lower()}"
        cases.append({"name": prefix, "input": save(prefix + "-input", inp),
                      "expected": save(prefix + "-expected", out),
                      "steps": [{"kind": "bilinear", "size": size, "align_corners": align}]})
    manifest = {"torch": torch.__version__, "weight_sha256": audited["sha256"],
                "upstream_commit": sources["uvr_commit"], "tensors": tensors, "cases": cases}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(cases)} real-shape probes to {args.output}")


if __name__ == "__main__":
    main()
