"""Verify the fixed RoFormer PCM baseline using original scheduling methods."""

import argparse
import ast
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import signal
import torch

from audit_vr import ROOT, sha256
from generate_roformer_model import load_model
from generate_sequence_probe import Fixtures
from roformer_precision import precise_stft

CHUNK, HOP, RATE = 352800, 441, 44100


def scheduler_class():
    sources = json.loads((ROOT / "references/roformer-source.json").read_text())
    source = next(f for f in sources["files"] if f["local"].endswith("mdxc_separator.py"))
    path = ROOT / source["local"]
    if sha256(path) != source["sha256"]:
        raise ValueError("scheduler source checksum mismatch")
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MDXCSeparator")
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_roformer_chunk_starts", "overlap_add")]
    if len(methods) != 2:
        raise ValueError("missing original scheduling methods")
    cls = ast.ClassDef(name="Scheduler", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Scheduler"](), source


def separate(model, scheduler, audio, rate):
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        audio = signal.resample_poly(audio, RATE // divisor, rate // divisor, axis=-1)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, axis=0)
    original = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    samples = audio.shape[1]
    padded = math.ceil(max(samples, 1025) / HOP) * HOP
    mix = torch.nn.functional.pad(original, (0, padded - samples))
    starts = scheduler._roformer_chunk_starts(padded, CHUNK, CHUNK // 4) if torch.any(original != 0) else []
    window = torch.tensor(signal.windows.hamming(CHUNK), dtype=torch.float32)
    result = torch.zeros_like(mix)
    counter = torch.zeros_like(mix)
    for index, offset in enumerate(starts):
        part = mix[:, offset:offset + CHUNK]
        with torch.inference_mode():
            output = model(part.unsqueeze(0))[0]
        if list(output.shape) != list(part.shape):
            raise ValueError("padded reference output must keep its length")
        scheduler.overlap_add(result, output, window, offset, part.shape[-1])
        counter[:, offset:offset + part.shape[-1]] += window[:part.shape[-1]]
        print(f"window {index + 1}/{len(starts)}", flush=True)
    vocals = (result / counter.clamp(min=1e-10))[:, :samples]
    return torch.stack((vocals, original - vocals)), len(starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boundary", action="store_true")
    parser.add_argument("--audio-file", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=3.0)
    parser.add_argument("--stft64", action="store_true", help="engineering baseline v1.1 forward STFT")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model, _, sources = load_model()
    scheduler, source = scheduler_class()
    fixtures = Fixtures(args.output)
    fixtures.sources = sources + [source, {"file": "tools/reference/generate_roformer_audio.py", "sha256": sha256(Path(__file__))}]
    if args.stft64:
        fixtures.sources.append({"stft_precision": "float64 forward, complex64 network input, float32 inverse",
                                 "precision_source_sha256": sha256(ROOT / "tools/reference/roformer_precision.py")})
    if args.audio_file:
        import soundfile as sf
        with sf.SoundFile(args.audio_file) as file:
            if file.channels not in (1, 2) or not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0:
                raise ValueError("expected positive duration and mono/stereo input")
            file.seek(60 * file.samplerate)
            rate = file.samplerate
            audio = file.read(round(args.duration_seconds * rate), dtype="float32", always_2d=True).T.copy()
        fixtures.sources.append({"local_audio_sha256": sha256(args.audio_file), "start_seconds": 60,
                                 "input_samples": audio.shape[1], "sample_rate": rate})
        cases = [("local_audio", audio, rate)]
    elif args.boundary:
        t = np.arange(CHUNK + 441, dtype=np.float32)
        audio = np.stack((np.sin(t * np.float32(0.073)), np.cos(t * np.float32(0.117)))) * np.float32(0.2)
        audio[0, 0], audio[1, -1] = 0.8, -0.7
        cases = [("overlapping_tail", audio, RATE)]
    else:
        t = np.arange(4097, dtype=np.float32)
        cases = [("signal", np.stack((np.sin(t * np.float32(0.073)), np.cos(t * np.float32(0.117)))) * np.float32(0.2), RATE),
                 ("single", np.array([[0.5]], dtype=np.float32), RATE),
                 ("short_resampled", np.array([[0.0, 0.3, -0.1, 0.2, 0.0, -0.5, 0.7]], dtype=np.float32), 22050),
                 ("silence", np.zeros((1, 503), dtype=np.float32), 48000)]
    parameters = {}
    for name, audio, rate in cases:
        start = time.perf_counter()
        with precise_stft(args.stft64):
            expected, windows = separate(model, scheduler, audio, rate)
        parameters[name] = {"sample_rate": rate, "windows": windows, "output_samples": expected.shape[-1],
                            "single_execution_seconds": time.perf_counter() - start}
        fixtures.cases.append({"name": name, "input": fixtures.save(name + "-input", torch.from_numpy(audio)),
                               "expected": fixtures.save(name + "-expected", expected), "steps": []})
        print(f"1296/{name}: {list(expected.shape)}", flush=True)
    fixtures.write()
    path = args.output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["audio"] = {"variant": "1296", "cases": parameters}
    path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
