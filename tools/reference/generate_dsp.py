"""Generate small synthetic fixtures from PyTorch, never from the Rust implementation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination = Path(__file__).resolve().parents[2] / "core/tests/fixtures/dsp"
    destination.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("silence", 320, 80, "constant", 667),
        ("impulse", 640, 108, "constant", 1379),
        ("signal", 768, 216, "constant", 1611),
        ("signal", 960, 480, "constant", 2041),
        ("signal", 1024, 256, "reflect", 2305),
        ("signal", 2048, 441, "reflect", 4853),
        ("short", 640, 108, "constant", 7),
        ("signal", 2, 1, "reflect", 17),
    ]
    manifest = {"torch": torch.__version__, "numpy": np.__version__, "cases": []}
    for index, (kind, n_fft, hop, padding, size) in enumerate(definitions):
        wave = torch.zeros(size, dtype=torch.float32)
        if kind == "impulse":
            wave[0], wave[size // 2], wave[-1] = 0.75, -0.5, 0.25
        elif kind != "silence":
            t = torch.arange(size, dtype=torch.float32)
            wave = 0.4 * torch.sin(t * 0.031) + 0.2 * torch.cos(t * 0.173)
            wave += (t.remainder(17) - 8) * 0.012
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        spectrum = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=window,
                              center=True, pad_mode=padding, normalized=False,
                              onesided=True, return_complex=True)
        weights = torch.linspace(0.2, 0.9, spectrum.shape[0], dtype=torch.float32)
        mask = torch.complex(weights * 0.5, weights * 0.1)[:, None]
        modified = spectrum * mask
        reconstructed = torch.istft(spectrum, n_fft=n_fft, hop_length=hop, window=window,
                                   center=True, normalized=False, onesided=True, length=size)
        filtered = torch.istft(modified, n_fft=n_fft, hop_length=hop, window=window,
                              center=True, normalized=False, onesided=True, length=size)
        # Fixed little-endian FP32 avoids text rounding and host byte order changes.
        values = torch.cat([wave, torch.view_as_real(spectrum).reshape(-1),
                            torch.view_as_real(modified).reshape(-1), reconstructed, filtered])
        raw = values.numpy().astype("<f4").tobytes()
        name = f"{index:02d}-{kind}-{n_fft}-{padding}.f32"
        (destination / name).write_bytes(raw)
        manifest["cases"].append({"file": name, "sha256": hashlib.sha256(raw).hexdigest(),
                                  "n_fft": n_fft, "hop": hop, "padding": padding,
                                  "samples": size, "frames": spectrum.shape[1]})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(definitions)} cases in {destination}")


if __name__ == "__main__":
    main()
