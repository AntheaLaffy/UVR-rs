import "./style.css";
import "./theme";
import type { Bridge, UpdateResult } from "./bridge";
import { getLanguage, initializeLanguage, onLanguageChange, phaseText, t, type TranslatedPhase } from "./i18n";
import { ModelLibrary } from "./models";
import { RuntimeSettings, type RuntimeDefaults } from "./runtime";
import { APP_VERSION } from "./version";

type Status = "running" | "completed" | "cancelled" | "failed";
interface TaskEvent extends TranslatedPhase { id: string; status: Status; fraction: number | null; elapsedSeconds: number; outputs: string[] }
declare global { interface Window { __TAURI__?: Bridge } }

initializeLanguage();

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element: ${id}`);
  return node as T;
}

const form = element<HTMLFormElement>("task-form");
const fields = {
  input: element<HTMLInputElement>("input-path"),
  models: element<HTMLInputElement>("models-path"),
  output: element<HTMLInputElement>("output-path"),
};
const model = element<HTMLSelectElement>("model");
const startButton = element<HTMLButtonElement>("start");
const cancelButton = element<HTMLButtonElement>("cancel");
const progressBar = element<HTMLProgressElement>("progress");
const phase = element("phase");
const status = element("task-status");
const outputs = element("outputs");
const taskLog = element("task-log");
const appVersion = element("app-version");
const checkUpdate = element<HTMLButtonElement>("check-update");
const updateStatus = element("update-status");
const releaseLink = element<HTMLAnchorElement>("release-link");
const bridge = window.__TAURI__;
const runtime = new RuntimeSettings();
let connected = false;
let running = false;
let picking = false;
let taskId = "";
let taskModel = "1296";
let startedAt = 0;
let lines: Array<{ text: () => string; seconds: number }> = [];
let lastPhase = "";
let latestEvent: TaskEvent | null = null;
let latestError: string | null = null;
let outputPaths: string[] = [];
let cancelling = false;
let modelsDirectoryCustomized = false;
let defaultModelsDirectory = "";

appVersion.textContent = t("version.current", { version: APP_VERSION });

function renderUpdate(result: UpdateResult): void {
  appVersion.textContent = t("version.current", { version: result.current });
  releaseLink.hidden = true;
  if (result.hasUpdate && result.latest) {
    updateStatus.textContent = t("version.latest", { version: result.latest });
    updateStatus.className = "update-available";
    if (result.releaseUrl) {
      releaseLink.hidden = false;
      releaseLink.href = result.releaseUrl;
      releaseLink.textContent = t("version.release");
    }
  } else {
    updateStatus.textContent = t("version.upToDate");
    updateStatus.className = "update-current";
    releaseLink.hidden = true;
  }
}

checkUpdate.addEventListener("click", async () => {
  if (!bridge || checkUpdate.disabled) return;
  checkUpdate.disabled = true;
  updateStatus.textContent = t("version.checking");
  try {
    renderUpdate(await bridge.core.invoke<UpdateResult>("check_update", { lang: getLanguage() }));
  } catch {
    updateStatus.textContent = t("version.unavailable");
    updateStatus.className = "update-unavailable";
    releaseLink.hidden = true;
  } finally {
    checkUpdate.disabled = false;
  }
});

const models: Record<string, string> = {
  "1296": "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
  "5hp": "5_HP-Karaoke-UVR.pth",
  "6hp": "6_HP-Karaoke-UVR.pth",
  "deecho": "UVR-DeEcho-DeReverb.pth",
};

function updateModel(): void {
  element("weights-name").textContent = t("settings.weights", { filename: models[model.value] });
  element("model-note").textContent = t(`model.${model.value}.note`);
  updateControls();
}

function updateControls(): void {
  const busy = running || picking || !!library?.downloading;
  form.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select").forEach((node) => { node.disabled = busy; });
  runtime.update(model.value, busy);
  form.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((node) => { node.disabled = busy || !connected; });
  startButton.disabled = busy || !connected || !library?.ready(model.value);
  startButton.textContent = t(running ? "task.starting" : "task.start");
  cancelButton.disabled = !running || cancelling;
  cancelButton.textContent = t(cancelling ? "task.cancelling" : "task.cancel");
  form.setAttribute("aria-busy", String(running));
  library?.updateControls();
}

function elapsed(seconds: number): string {
  const count = Math.max(0, Math.floor(seconds));
  return `${Math.floor(count / 60).toString().padStart(2, "0")}:${(count % 60).toString().padStart(2, "0")}`;
}

function renderLog(): void {
  taskLog.textContent = lines.length ? lines.map(line => `${elapsed(line.seconds)}  ${line.text()}`).join("\n") : t("task.noLog");
  lastPhase = lines.at(-1)?.text() ?? "";
}

function log(text: () => string, seconds: number): void {
  const message = text();
  if (message === lastPhase) return;
  lastPhase = message;
  lines.push({ text, seconds });
  lines = lines.slice(-80);
  renderLog();
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  latestError = message;
  phase.textContent = t("task.phase.error", { error: message });
  status.textContent = t("task.status.failed");
  status.className = "status failed";
  progressBar.value = 0;
  element("percentage").textContent = "";
  log(() => t("task.phase.error", { error: message }), running ? (performance.now() - startedAt) / 1000 : 0);
}

function savePaths(): void {
  try { localStorage.setItem("uvr.paths.v2", JSON.stringify({ models: modelsDirectoryCustomized ? fields.models.value : null, output: fields.output.value })); } catch { /* Remembering paths is optional. */ }
}

function showOutputs(paths: string[]): void {
  outputPaths = paths;
  outputs.replaceChildren();
  element("output-count").textContent = t("task.outputCount", { count: paths.length });
  const names = taskModel === "1296" ? ["track.vocals", "track.instrumental"] : taskModel === "deecho" ? ["track.dry", "track.reverb"] : ["track.karaoke", "track.removed"];
  if (!paths.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = t("task.empty");
    outputs.append(empty);
  }
  paths.forEach((path, index) => {
    const track = document.createElement("div");
    track.className = "output-track";
    const name = document.createElement("strong");
    name.textContent = t(names[index] ?? "track.output");
    const location = document.createElement("code");
    location.textContent = path;
    track.append(name, location);
    outputs.append(track);
  });
}

function receive(event: TaskEvent): void {
  if (event.id !== taskId) return;
  latestEvent = event;
  latestError = null;
  phase.textContent = phaseText(event);
  element("elapsed").textContent = elapsed(event.elapsedSeconds);
  if (event.fraction === null && event.status === "running") {
    progressBar.removeAttribute("value");
    element("percentage").textContent = "";
  } else {
    progressBar.value = event.fraction ?? 0;
    element("percentage").textContent = event.fraction === null ? "" : `${Math.round(event.fraction * 100)}%`;
  }
  status.textContent = t(`task.status.${event.status}`);
  status.className = `status ${event.status}`;
  log(() => phaseText(event), event.elapsedSeconds);
  if (event.status !== "running") {
    running = false;
    cancelling = false;
    updateControls();
    showOutputs(event.outputs);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!bridge || !connected || running || picking || library?.downloading || !library?.ready(model.value) || !runtime.reportValidity() || !form.reportValidity()) return;
  const runtimeRequest = runtime.request();
  running = true;
  cancelling = false;
  taskId = crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  taskModel = model.value;
  startedAt = performance.now();
  lines = [];
  lastPhase = "";
  showOutputs([]);
  updateControls();
  savePaths();
  log(() => t("task.runtime", { configuration: runtime.describe(runtimeRequest) }), 0);
  receive({ id: taskId, status: "running", phase: "", phaseKey: "task.phase.prepare", fraction: null, elapsedSeconds: 0, outputs: [] });
  try {
    await bridge.core.invoke("start_task", { request: { id: taskId, model: taskModel, input: fields.input.value, modelsDir: fields.models.value, outputDir: fields.output.value, runtime: runtimeRequest, lang: getLanguage() } });
  } catch (error) {
    showError(error);
    running = false;
    updateControls();
  }
});

cancelButton.addEventListener("click", async () => {
  if (!bridge || !running) return;
  cancelling = true;
  updateControls();
  try { await bridge.core.invoke("cancel_task", { id: taskId, lang: getLanguage() }); }
  catch (error) { showError(error); cancelling = false; updateControls(); }
});

document.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!bridge || picking || running) return;
    const kind = button.dataset.pick as keyof typeof fields;
    picking = true;
    updateControls();
    try {
      const selected = await bridge.core.invoke<string | null>("choose_path", { kind, lang: getLanguage() });
      if (selected !== null) {
        fields[kind].value = selected;
        if (kind === "models") modelsDirectoryCustomized = true;
        savePaths();
      }
    } catch (error) { showError(error); }
    finally { picking = false; updateControls(); if (kind === "models") void library?.scan(); }
  });
});

model.addEventListener("change", updateModel);
fields.models.addEventListener("input", updateControls);
fields.models.addEventListener("change", () => { modelsDirectoryCustomized = true; savePaths(); void library?.scan(); });
fields.output.addEventListener("change", savePaths);
setInterval(() => { if (running) element("elapsed").textContent = elapsed((performance.now() - startedAt) / 1000); }, 500);

async function connect(): Promise<void> {
  try {
    const saved = JSON.parse(localStorage.getItem("uvr.paths.v2") ?? localStorage.getItem("uvr.paths.v1") ?? "{}");
    if (typeof saved.models === "string") { fields.models.value = saved.models; modelsDirectoryCustomized = true; }
    if (typeof saved.output === "string") fields.output.value = saved.output;
  } catch { /* Ignore outdated or unavailable local preferences. */ }
  updateModel();
  updateControls();
  if (!bridge) { element("preview-note").hidden = false; checkUpdate.disabled = true; return; }
  try {
    await bridge.event.listen<TaskEvent>("task-progress", (event) => receive(event.payload));
    const defaults = await bridge.core.invoke<{ modelsDir: string | null; runtime: RuntimeDefaults }>("defaults", { lang: getLanguage() });
    runtime.configure(defaults.runtime);
    defaultModelsDirectory = defaults.modelsDir ?? "";
    if (!fields.models.value && defaults.modelsDir) fields.models.value = defaults.modelsDir;
    connected = true;
    await library?.connect();
    updateControls();
  } catch (error) { showError(error); }
}

const library = bridge ? new ModelLibrary(bridge, {
  directory: () => fields.models.value,
  selected: () => model.value,
  busy: () => running || picking,
  changed: updateControls,
  resetDirectory: () => {
    fields.models.value = defaultModelsDirectory;
    modelsDirectoryCustomized = false;
    savePaths();
  },
}) : null;

function refreshLanguage(): void {
  appVersion.textContent = t("version.current", { version: APP_VERSION });
  if (!releaseLink.hidden) releaseLink.textContent = t("version.release");
  updateModel();
  showOutputs(outputPaths);
  renderLog();
  if (latestError !== null) {
    phase.textContent = t("task.phase.error", { error: latestError });
    status.textContent = t("task.status.failed");
  } else if (latestEvent) {
    phase.textContent = phaseText(latestEvent);
    status.textContent = t(`task.status.${latestEvent.status}`);
  } else {
    phase.textContent = t("task.ready");
    status.textContent = t("task.status.waiting");
  }
}
onLanguageChange(refreshLanguage);
refreshLanguage();
void connect();
