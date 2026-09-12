"""Reference complete VR audio using the pinned network and explicit DSP baseline."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch
import soundfile as sf

from audit_vr import ROOT, sha256
from generate_sequence_probe import Fixtures
from generate_vr_dsp import analyze, config, reference, signal
from generate_vr_model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["5hp", "6hp", "deecho"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-file", type=Path, help="use one local music excerpt instead of synthetic cases")
    parser.add_argument("--start-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    audit_path = ROOT / f"benchmarks/artifacts/vr-reference/{args.variant}.json"
    model, audited, sources = load_model(audit_path)
    mp, config_path = config(args.variant)
    if sha256(config_path) != audited["config_sha256"]:
        raise ValueError("configuration differs from verified weights")
    ref = reference()
    modern = args.variant == "deecho"
    window = 512
    roi = window - 2 * model.offset
    hop = mp.param["band"][len(mp.param["band"])] ["hl"]
    definitions = [("signal", 4097, 2, 44100), ("impulses", 2 * hop + 7, 2, 44100),
                   ("short", 7, 1, 22050), ("silence", 503, 2, 48000),
                   ("exact_roi", (roi - 1) * hop, 2, 44100),
                   ("partial_roi", (roi + 11) * hop + 17, 2, 44100)]
    fixtures = Fixtures(args.output)
    parameters = {}
    local_audio = None
    if args.audio_file:
        info = sf.info(args.audio_file)
        if args.start_seconds < 0 or not args.duration_seconds > 0:
            raise ValueError("invalid excerpt range")
        first = round(args.start_seconds * info.samplerate)
        count = round(args.duration_seconds * info.samplerate)
        if count < 1 or first + count > info.frames:
            raise ValueError("excerpt exceeds local audio")
        local_audio, rate = sf.read(args.audio_file, start=first, frames=count, dtype="float32", always_2d=True)
        local_audio = local_audio.T.copy()
        definitions = [("local_audio", count, info.channels, rate)]
        fixtures.sources.append({"local_audio": args.audio_file.name, "sha256": sha256(args.audio_file),
                                 "start_sample": first, "samples": count, "sample_rate": rate,
                                 "decoder": f"soundfile {sf.__version__}, libsndfile {sf.__libsndfile_version__}"})
    for kind, count, channels, rate in definitions:
        wave = local_audio if local_audio is not None else signal(kind, count, channels)
        spectrum, output_samples = analyze(ref, mp, modern, wave, rate)
        magnitude = np.abs(spectrum)
        peak = magnitude.max()
        if peak == 0:
            primary = residual = np.zeros((2, output_samples), dtype=np.float32)
            patches = 0
        else:
            frames = spectrum.shape[-1]
            left = model.offset
            right = roi - frames % roi + left
            padded = np.pad(magnitude, ((0,0), (0,0), (left,right)))
            padded /= padded.max()
            patches = (padded.shape[-1] - 2 * model.offset) // roi
            masks = []
            for patch in range(patches):
                x = torch.from_numpy(padded[:, :, patch * roi:patch * roi + window][None].copy())
                with torch.inference_mode():
                    masks.append(model.predict_mask(x)[0].numpy())
            mask = np.concatenate(masks, axis=2)[:, :, :frames]
            phase = np.exp(1.j * np.angle(spectrum))
            primary = ref.cmb_spectrogram_to_wave(mask * magnitude * phase, mp, is_v51_model=modern)[:, :output_samples]
            residual = ref.cmb_spectrogram_to_wave((1 - mask) * magnitude * phase, mp, is_v51_model=modern)[:, :output_samples]
        expected = np.stack([primary, residual]).astype(np.float32)
        fixtures.cases.append({"name": kind, "input": fixtures.save(kind + "-input", torch.from_numpy(wave)),
                               "expected": fixtures.save(kind + "-expected", torch.from_numpy(expected)), "steps": []})
        parameters[kind] = {"sample_rate": rate, "window_frames": window, "patches": patches,
                            "spectrum_frames": spectrum.shape[-1], "output_samples": output_samples}
        print(f"{args.variant}/{kind}: {patches} network windows, {output_samples} output samples", flush=True)
    fixtures.sources.append({"file": audited["file"], "sha256": audited["sha256"], "upstream_commit": sources["uvr_commit"],
                             "config_sha256": audited["config_sha256"], "audit_sha256": sha256(audit_path),
                             "generator_sha256": sha256(Path(__file__)),
                             "dsp_generator_sha256": sha256(ROOT / "tools/reference/generate_vr_dsp.py"),
                             "uv_lock_sha256": sha256(ROOT / "tools/reference/uv.lock"), "preset": "docs/baseline.md v1"})
    fixtures.write()
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["audio"] = {"variant": args.variant, "cases": parameters}
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pass .* as keyword args", category=FutureWarning)
        warnings.filterwarnings("ignore", message="n_fft=.* is too small for input signal")
        main()
