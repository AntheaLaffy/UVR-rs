"""Capture DeEcho recurrence and 1296 attention blocks for Rust CPU probes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from audit_roformer import ConfigLoader, reference_class
from audit_vr import ROOT, sha256, upstream_networks


class CaptureComplete(Exception):
    pass


class Fixtures:
    def __init__(self, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.tensors = {}
        self.cases = []
        self.sources = []

    def save(self, name, tensor):
        if name in self.tensors:
            raise ValueError(f"duplicate tensor {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite tensor {name}")
        raw = tensor.detach().contiguous().numpy().astype("<f4").tobytes()
        filename = name + ".f32"
        (self.output / filename).write_bytes(raw)
        self.tensors[name] = {"file": filename, "shape": list(tensor.shape),
                              "sha256": hashlib.sha256(raw).hexdigest()}
        return name

    def case(self, name, inp, expected, step):
        self.cases.append({"name": name, "input": self.save(name + "-input", inp),
                           "expected": self.save(name + "-expected", expected), "steps": [step]})

    def write(self):
        manifest = {"torch": torch.__version__, "sources": self.sources,
                    "tensors": self.tensors, "cases": self.cases}
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {len(self.cases)} sequence probes to {self.output}", flush=True)


def deecho(fixtures):
    audit_path = ROOT / "benchmarks/artifacts/vr-reference/deecho.json"
    audit = json.loads(audit_path.read_text())["models"][0]
    weights = ROOT / "models" / audit["file"]
    if not audit["strict_load"] or sha256(weights) != audit["sha256"]:
        raise ValueError("DeEcho weights differ from strict audit")
    sources = json.loads((ROOT / "references/targets.json").read_text())
    _, nets_new = upstream_networks(sources["uvr_commit"])
    shape = audit["forward"]["input_shape"]
    with torch.device("meta"):
        model = nets_new.CascadedNet((shape[2] - 1) * 2, audit["architecture_size_kib"],
                                    nout=audit["metadata"].get("nout", 32),
                                    nout_lstm=audit["metadata"].get("nout_lstm", 128))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True, assign=True)
    model.eval()
    selected = ["stg1_low_band_net.0.lstm_dec2.lstm", "stg1_high_band_net.lstm_dec2.lstm",
                "stg3_full_band_net.lstm_dec2.lstm"]
    modules = dict(model.named_modules())
    captured = {}
    def capture(name):
        def hook(_module, inputs, output):
            captured[name] = (inputs[0].detach(), output[0].detach())
        return hook
    handles = [modules[name].register_forward_hook(capture(name)) for name in selected]
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    with torch.inference_mode():
        model.predict_mask((x.remainder(101) + 1) / 102.0)
    for handle in handles:
        handle.remove()
    for index, name in enumerate(selected):
        module = modules[name]
        assert module.num_layers == 1 and module.bidirectional and not module.batch_first
        step = {"kind": "bi_lstm"}
        for direction, suffix in [("forward", ""), ("reverse", "_reverse")]:
            step[direction] = {key: fixtures.save(f"lstm{index}-{key}{suffix}",
                                                  getattr(module, key + "_l0" + suffix))
                               for key in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]}
        inp, expected = captured[name]
        fixtures.case(f"lstm{index}-{name}", inp, expected, step)
        if index == 0:
            # Distinct batch entries and a short sequence expose axis/order bugs.
            short = torch.cat((inp[:7], inp[11:18] * -0.7), dim=1)
            with torch.inference_mode():
                expected, _ = module(short)
            fixtures.case("lstm-short-batch2", short, expected, step)
    fixtures.sources.append({"file": audit["file"], "sha256": audit["sha256"],
                             "audit_sha256": sha256(audit_path), "upstream_commit": sources["uvr_commit"],
                             "input": "same deterministic magnitude input as strict forward audit"})


def roformer(fixtures):
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
    time_attn, time_ff = model.layers[0][0].layers[0]
    freq_attn = model.layers[0][1].layers[0][0]
    captured = {}
    def capture(name, stop=False):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
            if stop:
                raise CaptureComplete
        return hook
    handles = [time_attn.register_forward_pre_hook(capture("time")),
               time_ff.register_forward_pre_hook(capture("ff")),
               freq_attn.register_forward_pre_hook(capture("freq", stop=True))]
    samples = config["audio"]["chunk_size"]
    t = torch.arange(samples, dtype=torch.float32)
    audio = torch.stack((torch.sin(t * 0.073), torch.cos(t * 0.117))).unsqueeze(0) * 0.2
    try:
        with torch.inference_mode():
            model(audio)
    except CaptureComplete:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"time", "ff", "freq"}:
        raise ValueError("expected first time/frequency transformer inputs")
    # Batch entries are independent. Keep full sequence lengths (801 and 62),
    # but sample batch entries to bound unfused Rust attention's score storage.
    for name, module, batch in [("time", time_attn, 2), ("freq", freq_attn, 16)]:
        inp = captured[name][:batch].contiguous()
        step = {"kind": "roformer_attention", "heads": module.heads,
                "norm": fixtures.save(name + "-norm", module.norm.gamma),
                "qkv": fixtures.save(name + "-qkv", module.to_qkv.weight),
                "gates_weight": fixtures.save(name + "-gates-weight", module.to_gates.weight),
                "gates_bias": fixtures.save(name + "-gates-bias", module.to_gates.bias),
                "out": fixtures.save(name + "-out", module.to_out[0].weight),
                "rotary_frequencies": fixtures.save(name + "-rotary-frequencies", module.rotary_embed.freqs)}
        with torch.inference_mode():
            expected = module(inp)
        fixtures.case("1296-attention-" + name, inp, expected, step)
        if name == "freq":
            with torch.inference_mode():
                zero = torch.zeros_like(inp[:1, :3])
                expected = module(zero)
            fixtures.case("1296-attention-zero", zero, expected, step)
    inp = captured["ff"][:2].contiguous()
    step = {"kind": "roformer_feed_forward",
            "norm": fixtures.save("ff-norm", time_ff.net[0].gamma),
            "input_weight": fixtures.save("ff-input-weight", time_ff.net[1].weight),
            "input_bias": fixtures.save("ff-input-bias", time_ff.net[1].bias),
            "output_weight": fixtures.save("ff-output-weight", time_ff.net[4].weight),
            "output_bias": fixtures.save("ff-output-bias", time_ff.net[4].bias)}
    with torch.inference_mode():
        expected = time_ff(inp)
    fixtures.case("1296-feed-forward-time", inp, expected, step)
    fixtures.sources.append({"file": audit["file"], "sha256": audit["sha256"],
                             "audit_sha256": sha256(audit_path), "source_commit": source["commit"],
                             "config_sha256": config_ref["sha256"], "samples": samples,
                             "full_module_input_shapes": {k: list(v.shape) for k, v in captured.items()},
                             "input": "deterministic stereo sin/cos, first transformer; selected batch entries only"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    fixtures = Fixtures(args.output)
    deecho(fixtures)
    roformer(fixtures)
    fixtures.write()


if __name__ == "__main__":
    main()
