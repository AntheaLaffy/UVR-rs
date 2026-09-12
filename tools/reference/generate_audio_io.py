"""Generate self-authored codec fixtures with libsndfile, independently of Rust I/O."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "core/tests/fixtures/audio"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    definitions = [
        ("mono-u8.wav", "WAV", "PCM_U8", 11025, 513, 1),
        ("stereo-i16.wav", "WAV", "PCM_16", 44100, 1025, 2),
        ("stereo-i24.wav", "WAV", "PCM_24", 48000, 2049, 2),
        ("mono-i32.wav", "WAV", "PCM_32", 32000, 1113, 1),
        ("stereo-float.wav", "WAV", "FLOAT", 44100, 1337, 2),
        ("stereo.flac", "FLAC", "PCM_16", 48000, 2113, 2),
        ("mono.mp3", "MP3", "MPEG_LAYER_III", 44100, 5007, 1),
        ("stereo.mp3", "MP3", "MPEG_LAYER_III", 48000, 7173, 2),
    ]
    manifest = {"soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__, "numpy": np.__version__,
                "generator_sha256": digest(Path(__file__)), "uv_lock_sha256": digest(ROOT / "tools/reference/uv.lock"), "cases": []}
    for name, container, subtype, rate, count, channels in definitions:
        t = np.arange(count, dtype=np.float64)
        wave = np.stack([0.4 * np.sin(t * 0.113) + 0.17 * np.cos(t * 0.317),
                         0.2 * np.cos(t * 0.153) - 0.11 * np.sin(t * 0.237)], axis=-1)[:, :channels].astype(np.float32)
        if subtype == "FLOAT":
            wave[0] = [1.5, -2.0]
        path = DESTINATION / name
        sf.write(path, wave, rate, format=container, subtype=subtype)
        expected, actual_rate = sf.read(path, dtype="float32", always_2d=True)
        if actual_rate != rate or expected.shape != wave.shape or not np.isfinite(expected).all():
            raise ValueError("reference codec failed length/channel/finite checks")
        expected_path = path.with_suffix(path.suffix + ".f32")
        expected_path.write_bytes(expected.T.astype("<f4").tobytes())
        manifest["cases"].append({"file": name, "sha256": digest(path), "reference": expected_path.name,
                                  "reference_sha256": digest(expected_path), "sample_rate": rate,
                                  "channels": channels, "samples_per_channel": count,
                                  "absolute_tolerance": 2e-6 if container == "MP3" else 1e-7,
                                  "relative_tolerance": 2e-5 if container == "MP3" else 1e-7})
    sf.write(DESTINATION / "three-channels.wav", np.zeros((16, 3), dtype=np.float32), 44100, subtype="FLOAT")
    sf.write(DESTINATION / "nonfinite.wav", np.array([0.0, np.nan], dtype=np.float32), 44100, subtype="FLOAT")
    sf.write(DESTINATION / "empty.wav", np.array([], dtype=np.float32), 44100, subtype="FLOAT")
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(definitions)} codec reference cases and three invalid audio files")


if __name__ == "__main__":
    main()
