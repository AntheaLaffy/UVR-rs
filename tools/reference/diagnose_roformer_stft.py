"""Isolate FFT/input differences; diagnostic outputs never replace golden data."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from audit_vr import sha256
from generate_roformer_model import load_model


def errors(actual, expected):
    a, b = actual.double(), expected.double()
    delta = a - b
    return {"max_absolute_error": delta.abs().max().item(), "rmse": delta.square().mean().sqrt().item(),
            "reference_rms": b.square().mean().sqrt().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--rust-stft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, config, sources = load_model()
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    case = manifest["cases"][0]
    def tensor(name):
        entry = manifest["tensors"][name]
        path = args.fixtures / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("fixture checksum mismatch")
        return torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(entry["shape"]))
    audio, expected = tensor(case["input"]), tensor(case["expected"])
    frames = audio.shape[-1] // 441 + 1
    rust_stft = torch.from_numpy(np.fromfile(args.rust_stft, dtype="<f4").reshape(2, 1025, frames, 2))
    reference = torch.view_as_real(torch.stft(audio, n_fft=2048, hop_length=441, window=torch.hann_window(2048), return_complex=True))
    def band_input(x):
        return x.permute(2, 1, 0, 3).reshape(1, frames, -1)
    a, b = band_input(rust_stft), band_input(reference)
    bands = []
    offset = 0
    with torch.inference_mode():
        for index, count in enumerate(config["model"]["freqs_per_bands"]):
            size = count * 4
            norm = model.band_split.to_features[index][0]
            bands.append({"band": index, "normalization": errors(norm(a[..., offset:offset + size]), norm(b[..., offset:offset + size]))})
            offset += size
    report = {"mode": "diagnostic_only", "sources": sources, "rust_stft_sha256": sha256(args.rust_stft),
              "spectrum": errors(rust_stft, reference), "bands": bands}
    print("STFT error:", report["spectrum"], flush=True)
    print("Largest normalized band error:", max(bands, key=lambda b: b["normalization"]["rmse"]), flush=True)
    if args.forward:
        # Same original network and ISTFT, supplying Rust's precomputed STFT only.
        with torch.inference_mode(), patch("torch.stft", return_value=torch.view_as_complex(rust_stft)):
            output = model(audio.unsqueeze(0))[0]
        report["reference_network_with_rust_stft"] = errors(output, expected)
        output.numpy().astype("<f4").tofile(args.output / "reference-with-rust-stft.f32")
        print("Original network with Rust STFT:", report["reference_network_with_rust_stft"], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
