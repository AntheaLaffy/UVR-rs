"""Drive the real Tauri window through WebKit WebDriver; verification only."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

from verify_vr_cli import load_tensor

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


class Browser:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}"
        self.session = None
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=40) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
        value = result.get("value")
        if isinstance(value, dict) and "error" in value:
            raise RuntimeError(value)
        return value

    def command(self, method, path, payload=None):
        return self.request(method, f"/session/{self.session}" + path, payload)

    def execute(self, script, *args):
        return self.command("POST", "/execute/sync", {"script": script, "args": args})

    def wait(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.2)
        raise TimeoutError("GUI condition timed out")

    def snapshot(self):
        return self.execute("""
            const byId = id => document.getElementById(id);
            return {phase: byId('phase').textContent, status: byId('task-status').textContent,
              statusClass: byId('task-status').className, startDisabled: byId('start').disabled,
              cancelDisabled: byId('cancel').disabled, percentage: byId('percentage').textContent,
              progress: byId('progress').value, log: byId('task-log').textContent,
              outputs: Array.from(document.querySelectorAll('#outputs code'), node => node.textContent)};
        """)

    def screenshot(self, path):
        path.write_bytes(base64.b64decode(self.command("GET", "/screenshot")))

    def click(self, selector):
        element = self.command("POST", "/element", {"using": "css selector", "value": selector})
        key = "element-6066-11e4-a52e-4f735466cecf"
        self.command("POST", f"/element/{element[key]}/click", {})

    def pick(self, kind, path):
        def xdotool(*args):
            result = subprocess.run(["/usr/sbin/xdotool", *args], capture_output=True, text=True, timeout=10)
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr)
            return result.stdout.strip()
        def windows():
            return set(xdotool("search", "--onlyvisible", "--class", ".*").splitlines())
        before = windows()
        self.click(f"[data-pick='{kind}']")
        def dialog():
            candidates = []
            for window in windows() - before:
                geometry = dict(line.split("=", 1) for line in
                                xdotool("getwindowgeometry", "--shell", window).splitlines() if "=" in line)
                if int(geometry.get("WIDTH", 0)) >= 300 and int(geometry.get("HEIGHT", 0)) >= 200:
                    candidates.append(window)
            if len(candidates) > 1:
                raise ValueError(f"More than one native dialog is visible: {candidates}")
            return candidates[0] if candidates else None
        window = self.wait(dialog)
        xdotool("windowfocus", "--sync", window)
        if path is None:
            xdotool("key", "--window", window, "Escape")
        else:
            xdotool("key", "--window", window, "ctrl+l")
            time.sleep(0.2)
            xdotool("type", "--window", window, "--clearmodifiers", "--delay", "3", str(path))
            xdotool("key", "--window", window, "Return")
            time.sleep(1)
            if window in windows():
                xdotool("key", "--window", window, "Return")
            if kind != "input" and window in windows():
                xdotool("key", "--window", window, "Tab", "Down", "alt+o")
        self.wait(lambda: self.execute("return !document.querySelector(arguments[0]).disabled;", f"[data-pick='{kind}']"))
        field = {"input": "input-path", "models": "models-path", "output": "output-path"}[kind]
        selected = self.execute("return document.getElementById(arguments[0]).value;", field)
        if path is not None and Path(selected) != path:
            raise ValueError(f"Native picker returned the wrong path: {selected}")
        return selected

    def start(self, model, audio, models, output):
        self.execute("""
            for (const [id, value] of Object.entries(arguments[0])) {
              const field = document.getElementById(id);
              if (field.disabled) throw new Error('Task control is still disabled: ' + id);
              field.value = value;
              field.dispatchEvent(new Event('change', {bubbles: true}));
            }
        """, {"model": model, "input-path": str(audio), "models-path": str(models),
              "output-path": str(output)})
        self.wait(lambda: not self.snapshot()["startDisabled"])
        self.click("#start")

    def configure_runtime(self, backend=None, threads=None):
        fields = {}
        if backend is not None:
            fields["runtime-backend"] = backend
        if threads is not None:
            fields["runtime-threads"] = str(threads)
        self.execute("""
            for (const [id, value] of Object.entries(arguments[0])) {
                const field = document.getElementById(id);
                field.value = value;
                field.dispatchEvent(new Event('input', {bubbles: true}));
                if (field.value !== value) throw new Error('Runtime option is unavailable: ' + value);
            }
        """, fields)

    def terminal(self, timeout):
        def finished():
            state = self.snapshot()
            return state if not state["startDisabled"] and state["status"] != "等待开始" else None
        return self.wait(finished, timeout)


def compare_outputs(paths, expected):
    if len(paths) != 2:
        raise ValueError("GUI did not publish two output paths")
    checks = []
    for index, path in enumerate(map(Path, paths)):
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        actual = audio.T
        if rate != 44100 or actual.shape != expected[index].shape or not np.isfinite(actual).all():
            raise ValueError("GUI output shape/rate/finite check failed")
        delta = actual.astype(np.float64) - expected[index].astype(np.float64)
        rmse = float(np.sqrt(np.mean(delta**2)))
        rms = float(np.sqrt(np.mean(expected[index].astype(np.float64)**2)))
        if np.any(np.abs(delta) > 2e-4 + 2e-3 * np.abs(expected[index])) or rmse > 1e-3 * max(rms, 1e-4):
            raise ValueError("GUI waveform exceeds the unchanged baseline tolerance")
        if not np.any(expected[index]) and np.any(actual):
            raise ValueError("GUI silence is not exactly zero")
        checks.append({"path": str(path), "max_absolute_error": float(np.max(np.abs(delta))),
                       "rmse": rmse, "reference_rms": rms,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return checks


def verify_runtime(browser):
    browser.execute("""
        window.__runtimeChecks = null;
        (async () => {
            const invoke = window.__TAURI__.core.invoke;
            const defaults = await invoke('defaults');
            const rejected = [];
            for (const [model, runtime] of [
                ['5hp', {backend: 'burn', threads: 0}],
                ['5hp', {backend: 'openvino-cpu', threads: 8}],
                ['1296', {backend: 'burn', threads: 8, roformer: {
                    timeBatch: 0, frequencyBatch: 301, windowParallelism: 1, linearLayout: 'flattened'}}]
            ]) {
                try {
                    await invoke('start_task', {request: {id: crypto.randomUUID(), model,
                        input: '/nonexistent/runtime.wav', modelsDir: '/nonexistent/models',
                        outputDir: '/nonexistent/output', runtime}});
                    throw new Error('Invalid runtime was accepted');
                } catch (error) {
                    if (String(error).includes('Invalid runtime was accepted')) throw error;
                    rejected.push(String(error));
                }
            }
            window.__runtimeChecks = {defaults: defaults.runtime, rejected};
        })().catch(error => { window.__runtimeChecks = {error: String(error)}; });
    """)
    checks = browser.wait(lambda: browser.execute("return window.__runtimeChecks;"))
    if "error" in checks:
        raise ValueError(checks["error"])
    controls = browser.execute("""
        const byId = id => document.getElementById(id);
        const set = (id, value, event = 'input') => {
            byId(id).value = value;
            byId(id).dispatchEvent(new Event(event, {bubbles:true}));
        };
        const defaults = window.__runtimeChecks.defaults;
        const choices = Array.from(byId('runtime-backend').options, node => node.value);
        if (JSON.stringify(choices) !== JSON.stringify(defaults.backends)) throw new Error('Backend capability mismatch');
        if (byId('runtime-backend').value !== defaults.roformerBackend) throw new Error('Recommended backend not selected');
        set('runtime-backend', 'burn');
        set('roformer-time-batch', '7');
        set('roformer-frequency-batch', '37');
        set('model', '5hp', 'change');
        set('vr-inference-batch', '2');
        if (!byId('vr-window-parallelism').disabled) throw new Error('Batch concurrency conflict');
        if (byId('runtime-backend').options.length !== 1) throw new Error('VR offers unsupported backend');
        set('model', 'deecho', 'change');
        if (!byId('vr-inference-batch').disabled || !byId('vr-window-parallelism').disabled || byId('vr-window-frames').min !== '144') throw new Error('DeEcho constraints missing');
        set('model', '1296', 'change');
        if (byId('roformer-time-batch').value !== '7' || byId('roformer-frequency-batch').value !== '37') throw new Error('Model switch lost settings');
        byId('runtime-reset').click();
        return {choices, recommended: defaults.roformerBackend, modelSwitch: true, schedulingConstraints: true};
    """)
    checks.update(controls)
    return checks


def verify_model_library(browser, output, live_download, portable):
    def states():
        return browser.execute("return Object.fromEntries(Array.from(document.querySelectorAll('[data-model]'), row => [row.dataset.model, row.dataset.availability]));")
    def scanned():
        return browser.execute("return !document.getElementById('check-models').disabled;")
    def directory(path):
        browser.execute("""
            const field = document.getElementById('models-path');
            field.value = arguments[0];
            field.dispatchEvent(new Event('input', {bubbles: true}));
            field.dispatchEvent(new Event('change', {bubbles: true}));
        """, str(path))
        browser.wait(scanned)
        return states()
    def refresh():
        browser.click("#check-models")
        browser.wait(scanned)
        return states()
    def download_state():
        return browser.execute("""
            const message = document.getElementById('download-message');
            return {status: message.dataset.status, message: message.textContent,
              progress: document.getElementById('download-progress').value};
        """)
    browser.execute("document.getElementById('model-library').open = true;")
    initial = browser.execute("return document.getElementById('models-path').value;")
    if portable and Path(initial) != output / "portable/models":
        raise ValueError(f"Portable model directory was not selected: {initial}")
    if set(states().values()) != {"ready"} or len(states()) != 4:
        raise ValueError("Existing models did not pass automatic detection")
    linked = output / "custom models link"
    linked.symlink_to(ROOT / "models", target_is_directory=True)
    if set(directory(linked).values()) != {"ready"}:
        raise ValueError("Custom model directory symlink was not followed")
    storage = output / "downloaded models"
    storage.mkdir()
    if set(directory(storage).values()) != {"missing"} or not browser.snapshot()["startDisabled"]:
        raise ValueError("Empty directory did not disable unavailable inference")
    orphan = storage / "5_HP-Karaoke-UVR.pth.part"
    orphan.write_bytes(b"incomplete download")
    if refresh()["5hp"] != "missing":
        raise ValueError("Partial download was misidentified as a model")
    target = storage / "5_HP-Karaoke-UVR.pth"
    target.write_bytes(b"invalid model")
    if refresh()["5hp"] != "invalid":
        raise ValueError("Corrupt model was not detected")
    browser.execute("document.getElementById('model').value = '5hp'; document.getElementById('model').dispatchEvent(new Event('change')); document.getElementById('download-proxy').value = 'not-a-proxy';")
    browser.click("[data-download='5hp']")
    failed = browser.wait(lambda: (value if (value := download_state())["status"] == "failed" else None))
    browser.wait(scanned)
    if "代理" not in failed["message"] or target.read_bytes() != b"invalid model" or list(storage.glob(".uvr-download-*")):
        raise ValueError("Invalid proxy recovery lost the existing model or temporary cleanup")
    browser.execute("document.getElementById('download-proxy').value = '';")
    checks = {"portable_default": initial, "directory_symlink": str(linked), "missing_and_partial": True,
              "corrupt_file": True, "proxy_validation": failed, "live_download": False}
    if live_download:
        browser.click("[data-download='5hp']")
        def receiving():
            state = download_state()
            if state["status"] == "failed":
                raise ValueError(state["message"])
            return state if "正在下载模型" in state["message"] and 0 < state["progress"] < 1 else None
        entered = browser.wait(receiving, 180)
        start = time.monotonic()
        browser.click("#cancel-download")
        cancelled = browser.wait(lambda: (value if (value := download_state())["status"] == "cancelled" else None))
        cancel_seconds = time.monotonic() - start
        browser.wait(scanned)
        if target.read_bytes() != b"invalid model" or list(storage.glob(".uvr-download-*")):
            raise ValueError("Download cancellation changed an existing model or left temporary data")
        print(f"GUI real model download cancellation passed in {cancel_seconds:.3f}s", flush=True)
        browser.click("[data-download='5hp']")
        def completed():
            state = download_state()
            if state["status"] == "failed":
                raise ValueError(state["message"])
            return state if state["status"] == "completed" else None
        result = browser.wait(completed, 900)
        browser.wait(scanned)
        with target.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        verified = json.loads((ROOT / "references/verified-weights.json").read_text())
        identity = next(model for model in verified["models"] if model["file"] == target.name)
        if target.stat().st_size != identity["size_bytes"] or digest != identity["sha256"] or states()["5hp"] != "ready":
            raise ValueError("Downloaded model identity or GUI availability is incorrect")
        if browser.snapshot()["startDisabled"]:
            raise ValueError("Downloaded selected model did not enable inference")
        checks.update({"live_download": True, "download": result, "sha256": digest,
                       "cancel_seconds": cancel_seconds, "cancel_entered": entered, "cancelled": cancelled})
        print("GUI Hugging Face download and full-file identity verification passed", flush=True)
    browser.command("POST", "/refresh", {})
    browser.wait(scanned)
    saved = browser.execute("return document.getElementById('models-path').value;")
    if Path(saved) != storage:
        raise ValueError("Custom model directory was not restored after reloading")
    browser.execute("document.getElementById('model-library').open = true;")
    browser.screenshot(output / "model-library.png")
    checks["custom_directory_restored"] = saved
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/uvr-gui")
    parser.add_argument("--driver", type=Path, default=ROOT / ".local/tauri-driver/bin/tauri-driver")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "benchmarks/artifacts/backend-probe/1296-f64-audio")
    parser.add_argument("--case", default="signal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=360)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--native-picker-check", action="store_true")
    parser.add_argument("--model-library-check", action="store_true")
    parser.add_argument("--download-check", action="store_true")
    parser.add_argument("--portable-check", action="store_true")
    parser.add_argument("--empty-library-check", action="store_true")
    parser.add_argument("--runtime-check", action="store_true")
    parser.add_argument("--backend", choices=["burn", "openvino-cpu"])
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.portable_check:
        portable = args.output / "portable"
        portable.mkdir()
        binary = portable / "uvr-gui"
        os.link(args.binary.resolve(), binary)
        (portable / "models").symlink_to(ROOT / "models", target_is_directory=True)
        args.binary = binary
    if args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if args.threads is not None and args.threads <= 0:
        parser.error("threads must be positive")
    manifest_raw = (args.fixtures / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    case = next(case for case in manifest["cases"] if case["name"] == args.case)
    waveform = load_tensor(args.fixtures, manifest["tensors"][case["input"]])
    expected = load_tensor(args.fixtures, manifest["tensors"][case["expected"]])
    variant = manifest["audio"]["variant"]
    audio = args.output / "声音 input.wav"
    sf.write(audio, waveform.T, manifest["audio"]["cases"][args.case]["sample_rate"], subtype="FLOAT")
    environment = os.environ.copy()
    environment.pop("WAYLAND_DISPLAY", None)
    environment.update({"GDK_BACKEND": "x11", "RAYON_NUM_THREADS": "2", "PATH": "/nonexistent",
                        "GIO_USE_VFS": "local", "GSETTINGS_BACKEND": "memory", "GTK_USE_PORTAL": "0",
                        "NO_AT_BRIDGE": "1"})
    for key in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR"):
        directory = args.output / key.lower()
        directory.mkdir(mode=0o700)
        environment[key] = str(directory)
    # GTK's SVG icon loader uses bubblewrap. Allow this system helper while
    # keeping Python and other development tools absent from the application's PATH.
    if args.native_picker_check:
        helper_directory = args.output / "native-helpers"
        helper_directory.mkdir()
        (helper_directory / "bwrap").symlink_to("/usr/sbin/bwrap")
        environment["PATH"] = str(helper_directory)
    port, native_port = free_port(), free_port()
    while native_port == port:
        native_port = free_port()
    browser = Browser(port)
    report = {"mode": "native_gui_verification_only", "variant": variant, "case": args.case,
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(manifest_raw).hexdigest(),
              "runtime_path": environment["PATH"], "checks": []}
    with (args.output / "driver.log").open("w") as log:
        process = subprocess.Popen([str(args.driver.resolve()), "--port", str(port), "--native-port", str(native_port),
                                    "--native-driver", "/usr/sbin/WebKitWebDriver"],
                                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            def ready():
                if process.poll() is not None:
                    raise RuntimeError("tauri-driver exited before becoming ready")
                try:
                    return browser.request("GET", "/status")
                except (OSError, urllib.error.URLError):
                    return None
            browser.wait(ready)
            session = browser.request("POST", "/session", {"capabilities": {"alwaysMatch": {
                "tauri:options": {"application": str(args.binary.resolve())}}}})
            browser.session = session["sessionId"]
            browser.wait(lambda: browser.execute("return !!window.__TAURI__ && !document.getElementById('check-models').disabled;"))
            browser.execute("""
                const language = document.getElementById('language');
                language.value = 'zh-CN';
                language.dispatchEvent(new Event('change', {bubbles:true}));
            """)
            if browser.execute("return !document.getElementById('preview-note').hidden;"):
                raise ValueError("Native window incorrectly shows the browser preview notice")
            models = browser.execute("return Array.from(document.querySelectorAll('#model option'), node => node.value);")
            if models != ["1296", "5hp", "6hp", "deecho"]:
                raise ValueError("GUI model choices differ from supported models")
            browser.screenshot(args.output / "ready.png")
            report["checks"].append({"startup": True, "models": models})
            if args.runtime_check:
                report["checks"].append({"runtime_options": verify_runtime(browser)})
                browser.execute("document.getElementById('runtime-advanced').open = true;")
                browser.screenshot(args.output / "runtime.png")
            browser.execute("""
                const model = document.getElementById('model');
                model.value = arguments[0];
                model.dispatchEvent(new Event('change', {bubbles:true}));
            """, variant)
            browser.configure_runtime(args.backend, args.threads)
            report["runtime"] = browser.execute("""
                return {backend: document.getElementById('runtime-backend').value,
                    threads: Number(document.getElementById('runtime-threads').value)};
            """)
            if args.empty_library_check:
                selected = browser.execute("return document.getElementById('models-path').value;")
                missing = browser.execute("return Array.from(document.querySelectorAll('[data-model]'), row => row.dataset.availability);")
                if Path(selected) != args.binary.resolve().parent / "models" or missing != ["missing"] * 4 or not browser.snapshot()["startDisabled"]:
                    raise ValueError("Light archive did not discover its empty models directory")
                buttons = browser.execute("return Array.from(document.querySelectorAll('[data-download]'), button => !button.disabled);")
                if buttons != [True] * 4:
                    raise ValueError("Light archive did not offer on-demand model downloads")
                browser.execute("document.getElementById('model-library').scrollIntoView();")
                browser.screenshot(args.output / "empty-library.png")
                report["checks"].append({"empty_portable_library": selected, "available_downloads": 4})
                report["passed"] = True
                print("Extracted light archive startup and on-demand model controls passed", flush=True)
                return
            if args.model_library_check:
                report["checks"].append({"model_library": verify_model_library(browser, args.output,
                                                                              args.download_check, args.portable_check)})
            if args.native_picker_check:
                picker_audio = args.output / "picker.wav"
                picker_audio.symlink_to(audio)
                selected = browser.pick("input", picker_audio)
                previous = browser.execute("return document.getElementById('models-path').value;")
                if browser.pick("models", None) != previous:
                    raise ValueError("Cancelling a native picker changed the selected path")
                output = browser.pick("output", args.output)
                report["checks"].append({"native_file_picker": selected, "native_folder_picker": output,
                                          "picker_cancel_preserves_path": True})
                print("Native file/folder selection and dialog cancellation passed", flush=True)
            browser.start(variant, args.output / "missing.wav", ROOT / "models", args.output / "missing")
            failed = browser.terminal(20)
            if failed["status"] != "处理失败" or failed["outputs"] or not failed["cancelDisabled"]:
                raise ValueError(f"Missing input was not reported correctly: {failed}")
            report["checks"].append({"missing_input": failed})
            print("GUI startup, model choices, and missing-input recovery passed", flush=True)
            if not args.smoke_only:
                destination = args.output / "分离 outputs"
                browser.start(variant, audio, ROOT / "models", destination)
                completed = browser.terminal(args.timeout_seconds)
                if completed["status"] != "已完成" or completed["progress"] != 1 or not completed["cancelDisabled"]:
                    raise ValueError(f"GUI task failed: {completed}")
                checks = compare_outputs(completed["outputs"], expected)
                browser.screenshot(args.output / "completed.png")
                report["checks"].append({"separation": completed, "waveforms": checks})
                before = [Path(path).read_bytes() for path in completed["outputs"]]
                browser.start(variant, audio, ROOT / "models", destination)
                refused = browser.terminal(20)
                if refused["status"] != "处理失败" or before != [Path(path).read_bytes() for path in completed["outputs"]]:
                    raise ValueError("GUI did not preserve existing output files")
                report["checks"].append({"overwrite_refusal": refused})
                print(f"GUI {variant} WAV outputs match reference; overwrite refusal passed", flush=True)
                cancelled_directory = args.output / "cancelled"
                browser.start(variant, audio, ROOT / "models", cancelled_directory)
                def inference_started():
                    state = browser.snapshot()
                    if not state["startDisabled"]:
                        raise ValueError(f"Task ended before cancellation check: {state}")
                    if "分离音频" in state["phase"] and (variant != "1296" or state["progress"] > 0.05):
                        return state
                    return None
                entered = browser.wait(inference_started, args.timeout_seconds)
                start = time.monotonic()
                browser.execute("document.getElementById('cancel').click();")
                cancelled = browser.terminal(90)
                seconds = time.monotonic() - start
                if cancelled["status"] != "已取消" or cancelled["outputs"] or list(cancelled_directory.iterdir()):
                    raise ValueError(f"GUI cancellation left incomplete files: {cancelled}")
                report["checks"].append({"cancel_seconds": seconds, "entered": entered, "cancelled": cancelled})
                browser.screenshot(args.output / "cancelled.png")
                print(f"GUI cancellation passed in {seconds:.3f}s", flush=True)
            report["passed"] = True
        except Exception as error:
            report["passed"] = False
            report["error"] = str(error)
            subprocess.run(["/usr/sbin/import", "-window", "root", str(args.output / "native-failure.png")],
                           capture_output=True, timeout=10)
            if browser.session:
                try:
                    report["last_state"] = browser.snapshot()
                    browser.screenshot(args.output / "failure.png")
                except Exception as capture_error:
                    report["capture_error"] = str(capture_error)
            raise
        finally:
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if browser.session:
                try:
                    browser.command("DELETE", "")
                except (OSError, RuntimeError):
                    pass
            # All processes here belong to this private driver process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
