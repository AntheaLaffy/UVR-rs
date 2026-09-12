"""Generate complete VR reference masks without exporting model weights."""

import argparse
import json
from pathlib import Path

import torch

from audit_vr import ROOT, sha256, upstream_networks
from generate_sequence_probe import Fixtures


def load_model(audit_path):
    audited = json.loads(audit_path.read_text())["models"][0]
    if audited["network"] not in {"CascadedASPPNet", "CascadedNet"} or not audited["strict_load"]:
        raise ValueError("expected successful VR audit")
    weights = ROOT / "models" / audited["file"]
    if sha256(weights) != audited["sha256"]:
        raise ValueError("weights differ from audit")
    sources = json.loads((ROOT / "references/targets.json").read_text())
    nets, nets_new = upstream_networks(sources["uvr_commit"])
    bins = audited["forward"]["input_shape"][2]
    with torch.device("meta"):
        if audited["network"] == "CascadedNet":
            model = nets_new.CascadedNet((bins - 1) * 2, audited["architecture_size_kib"],
                                        nout=audited["metadata"].get("nout", 32),
                                        nout_lstm=audited["metadata"].get("nout_lstm", 128))
        else:
            model = nets.determine_model_capacity((bins - 1) * 2, audited["architecture_size_kib"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True, assign=True)
    model.eval()
    return model, audited, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, audited, sources = load_model(args.audit)
    bins = audited["forward"]["input_shape"][2]
    fixtures = Fixtures(args.output)
    for name, frames in [("magnitude", 320), ("silence", 2 * model.offset + 16), ("impulses", 2 * model.offset + 48)]:
        shape = [1, 2, bins, frames]
        if name == "magnitude":
            x = torch.arange(2 * bins * frames, dtype=torch.float32).reshape(shape)
            x = (x.remainder(101) + 1) / 102.0
        else:
            x = torch.zeros(shape)
            if name == "impulses":
                x[0, 0, 31, 0] = 1.0
                x[0, 1, bins // 2, 149] = 0.7
                x[0, 0, bins - 2, -1] = 0.3
        if args.batch > 1:
            # Distinct lanes catch accidental broadcasting or batch mixing.
            lanes = [x, torch.zeros_like(x), x.flip(-1) * 0.5, x.flip(-2) * 0.25]
            x = torch.cat(lanes[:args.batch], dim=0)
        with torch.inference_mode():
            expected = model.predict_mask(x)
        fixtures.cases.append({"name": name, "input": fixtures.save(name + "-input", x),
                               "expected": fixtures.save(name + "-expected", expected), "steps": []})
    fixtures.sources.append({"file": audited["file"], "sha256": audited["sha256"],
                             "config_sha256": audited["config_sha256"], "upstream_commit": sources["uvr_commit"],
                             "audit_sha256": sha256(args.audit), "input": "normalized magnitudes, no audio DSP"})
    fixtures.write()


if __name__ == "__main__":
    main()
