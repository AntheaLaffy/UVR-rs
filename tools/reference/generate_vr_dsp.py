"""Generate DSP goldens from pinned SciPy/librosa and extracted UVR functions."""

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import types
import warnings

import librosa
import numpy as np
import scipy
from scipy.signal import resample_poly

# librosa 0.9.2's dtype tables still use these removed aliases. They were exactly
# the Python builtins; restore only those names in this isolated reference process.
np.float = float
np.complex = complex

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "core/tests/fixtures/vr-dsp"
SOURCE = ROOT / "upstream/lib_v5/spec_utils.py"
FUNCTIONS = {
    "combine_spectrograms", "wave_to_spectrogram", "convert_channels",
    "spectrogram_to_wave", "cmb_spectrogram_to_wave", "get_lp_filter_mask",
    "get_hp_filter_mask", "fft_lp_filter", "fft_hp_filter",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AllocationControl:
    """Only replace raw ndarray allocation; every numerical operation remains NumPy."""
    def __init__(self, fill):
        self.fill = fill

    def ndarray(self, shape, dtype):
        return np.full(shape=shape, fill_value=self.fill, dtype=dtype)

    def __getattr__(self, key):
        return getattr(np, key)


def reference(fill=0):
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected} != FUNCTIONS:
        raise ValueError("fixed UVR DSP functions are missing")
    namespace = {"np": AllocationControl(fill), "librosa": librosa, "math": math,
                 "wav_resolution": "polyphase"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def config(name):
    names = {"5hp": "4band_v2_sn", "6hp": "3band_44100_msb2", "deecho": "4band_v3"}
    path = ROOT / "upstream/lib_v5/vr_network/modelparams" / (names[name] + ".json")
    parameters = json.loads(path.read_text())
    parameters["band"] = {int(k): v for k, v in parameters["band"].items()}
    for key in ["reverse", "mid_side", "mid_side_b2"]:
        parameters.setdefault(key, False)
    return types.SimpleNamespace(param=parameters), path


def signal(kind, count, channels):
    t = np.arange(count, dtype=np.float64)
    x = np.stack([0.3 * np.sin(t * 0.031) + 0.12 * np.cos(t * 0.173),
                  0.23 * np.cos(t * 0.041) - 0.1 * np.sin(t * 0.237)]).astype(np.float32)
    if kind == "silence":
        x.fill(0)
    elif kind == "impulses":
        x.fill(0)
        x[0, [0, count // 2, count - 1]] = [0.75, -0.5, 0.25]
        x[1, [0, count // 3, count - 1]] = [-0.3, 0.4, 0.7]
    return x[:channels]


def analyze(ref, mp, modern, input_wave, rate):
    wave = librosa.resample(input_wave, orig_sr=rate, target_sr=44100, res_type="polyphase")
    if wave.shape[0] == 1:
        wave = np.repeat(wave, 2, axis=0)
    output_samples = wave.shape[-1]
    bands = mp.param["band"]
    hop = bands[len(bands)]["hl"]
    padded = (output_samples + hop - 1) // hop * hop
    wave = np.pad(wave, ((0, 0), (0, padded - output_samples)))
    spectra = {}
    rate = 44100
    for index in range(len(bands), 0, -1):
        band = bands[index]
        wave = librosa.resample(wave, orig_sr=rate, target_sr=band["sr"], res_type="polyphase")
        rate = band["sr"]
        spectra[index] = ref.wave_to_spectrogram(wave, band["hl"], band["n_fft"], mp, index, modern)
    return ref.combine_spectrograms(spectra, mp, modern), output_samples


def save(name, arrays):
    raw = np.concatenate([np.asarray(a, dtype="<f4").reshape(-1) for a in arrays]).tobytes()
    (DESTINATION / name).write_bytes(raw)
    return {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    targets = json.loads((ROOT / "references/targets.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    if commit != targets["uvr_commit"]:
        raise ValueError("UVR commit mismatch")
    subprocess.run(["git", "-C", str(ROOT / "upstream"), "diff", "--exit-code", commit, "--", "lib_v5"], check=True)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ref, poisoned = reference(), reference(1.0)
    manifest = {"librosa": librosa.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
                "uvr_commit": commit, "source_sha256": digest(SOURCE), "generator_sha256": digest(Path(__file__)),
                "uv_lock_sha256": digest(ROOT / "tools/reference/uv.lock"),
                "parameters": {"resampling": "polyphase", "pad_mode": "constant", "uncovered_bins": 0,
                               "numpy_compatibility": "np.float=float; np.complex=complex for librosa 0.9.2 dtype tables",
                               "length": "pad to complete highest-band hop then crop to resampled input"},
                "allocation_experiment": [], "resampling": [], "cases": []}
    pairs = [(44100, 14700), (14700, 7350), (7350, 14700), (14700, 44100),
             (44100, 22050), (22050, 11025), (11025, 22050), (22050, 44100),
             (48000, 44100), (44100, 48000), (32000, 44100), (44100, 44100)]
    for index, (source_rate, target_rate) in enumerate(pairs):
        x = signal("signal", 513 + index * 17, 1)[0]
        expected = resample_poly(x, target_rate, source_rate)
        case = save(f"resample-{source_rate}-{target_rate}.f32", [x, expected])
        case.update(source_rate=source_rate, target_rate=target_rate, samples=len(x), output_samples=len(expected))
        manifest["resampling"].append(case)
    # Isolate filter boundary alignment at the smallest possible input.
    for source_rate, target_rate in [(48000, 44100), (7350, 44100)]:
        x = np.array([0.75], dtype=np.float32)
        expected = resample_poly(x, target_rate, source_rate)
        case = save(f"single-{source_rate}-{target_rate}.f32", [x, expected])
        case.update(source_rate=source_rate, target_rate=target_rate, samples=1, output_samples=len(expected))
        manifest["resampling"].append(case)
    definitions = [("silence", 503, 2, 44100), ("impulses", 1537, 2, 44100),
                   ("signal", 4097, 2, 44100), ("short", 7, 1, 22050),
                   ("single", 1, 2, 44100), ("resampled", 3073, 2, 48000)]
    for variant in ["5hp", "6hp", "deecho"]:
        mp, config_path = config(variant)
        modern = variant == "deecho"
        empty = np.zeros((2, mp.param["bins"] + 1, 4), dtype=np.complex64)
        clean = ref.cmb_spectrogram_to_wave(empty, mp, is_v51_model=modern)
        dirty = poisoned.cmb_spectrogram_to_wave(empty, mp, is_v51_model=modern)
        difference = float(np.max(np.abs(dirty - clean)))
        if np.any(clean) or not difference > 0 or not np.isfinite(dirty).all():
            raise ValueError("allocation experiment did not isolate uncovered bins")
        manifest["allocation_experiment"].append({"variant": variant, "uncovered_fill": [0, 1],
                                                    "maximum_waveform_difference": difference})
        for kind, count, channels, rate in definitions:
            x = signal(kind, count, channels)
            spectrum, output_samples = analyze(ref, mp, modern, x, rate)
            mask = np.linspace(0.05, 0.95, spectrum.size, dtype=np.float32).reshape(spectrum.shape)
            primary = ref.cmb_spectrogram_to_wave(spectrum * mask, mp, is_v51_model=modern)[:, :output_samples]
            residual = ref.cmb_spectrogram_to_wave(spectrum * (1 - mask), mp, is_v51_model=modern)[:, :output_samples]
            if not all(np.isfinite(v).all() for v in [spectrum, primary, residual]):
                raise ValueError("nonfinite reference output")
            complex_pairs = np.stack([spectrum.real, spectrum.imag], axis=-1)
            case = save(f"{variant}-{kind}.f32", [x, complex_pairs, mask, primary, residual])
            case.update(variant=variant, kind=kind, sample_rate=rate, channels=channels, samples=count,
                        output_samples=output_samples, bins=spectrum.shape[1], frames=spectrum.shape[2],
                        config_sha256=digest(config_path))
            manifest["cases"].append(case)
    with np.errstate(invalid="ignore"):
        silence = np.zeros((2, 673, 512), dtype=np.float32)
        silence /= silence.max()
    manifest["unmodified_silence_normalization_is_nonfinite"] = bool(not np.isfinite(silence).any())
    if not manifest["unmodified_silence_normalization_is_nonfinite"]:
        raise ValueError("expected zero-division behavior changed")
    (DESTINATION / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(manifest['resampling'])} resampling and {len(manifest['cases'])} multi-band cases")


if __name__ == "__main__":
    # Fixed upstream uses positional librosa arguments and intentionally short STFT inputs.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
        warnings.filterwarnings("ignore", message="n_fft=.* is too small for input signal")
        main()
