"""Engineering baseline: FP64 forward STFT, FP32 network and reconstruction."""

from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def precise_stft(enabled=True):
    if not enabled:
        yield
        return
    original_stft, original_istft = torch.stft, torch.istft
    window64 = torch.hann_window(2048, dtype=torch.float64)

    def forward(input, *args, window=None, **kwargs):
        if input.dtype != torch.float32 or kwargs.get("n_fft") != 2048 or kwargs.get("hop_length") != 441:
            raise ValueError("precision override only supports the fixed 1296 STFT")
        return original_stft(input.double(), *args, window=window64, **kwargs).to(torch.complex64)

    def inverse(input, *args, window=None, **kwargs):
        return original_istft(input, *args, window=window64.float(), **kwargs)

    with patch("torch.stft", forward), patch("torch.istft", inverse):
        yield
