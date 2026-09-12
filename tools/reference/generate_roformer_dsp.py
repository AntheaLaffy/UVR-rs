"""Small independent FP64-forward STFT fixtures for the 1296 precision baseline."""

from pathlib import Path

import torch

from audit_vr import ROOT, sha256
from generate_sequence_probe import Fixtures


def main():
    torch.set_num_threads(2)
    fixtures = Fixtures(ROOT / "core/tests/fixtures/roformer-dsp")
    for count in (1025, 4097, 8193):
        t = torch.arange(count, dtype=torch.float32)
        audio = torch.stack((torch.sin(t * 0.073), torch.cos(t * 0.117))) * 0.2
        audio[0, 0], audio[1, -1] = 0.75, -0.5
        spectrum = torch.stft(audio.double(), n_fft=2048, hop_length=441,
                              window=torch.hann_window(2048, dtype=torch.float64), return_complex=True).to(torch.complex64)
        name = f"stft-{count}"
        fixtures.cases.append({"name": name, "input": fixtures.save(name + "-input", audio),
                               "expected": fixtures.save(name + "-expected", torch.view_as_real(spectrum)), "steps": []})
    fixtures.sources = [{"file": "tools/reference/generate_roformer_dsp.py", "sha256": sha256(Path(__file__)),
                         "precision": "float64 periodic Hann, multiply, reflect STFT; cast to complex64",
                         "uv_lock_sha256": sha256(ROOT / "tools/reference/uv.lock")}]
    fixtures.write()


if __name__ == "__main__":
    main()
