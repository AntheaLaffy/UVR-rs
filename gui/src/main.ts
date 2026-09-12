import "./style.css";
import type { Bridge } from "./bridge";
import { ModelLibrary } from "./models";

type Status = "running" | "completed" | "cancelled" | "failed";
interface TaskEvent { id: string; status: Status; phase: string; fraction: number | null; elapsedSeconds: number; outputs: string[] }
declare global { interface Window { __TAURI__?: Bridge } }

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
const bridge = window.__TAURI__;
let connected = false;
let running = false;
let picking = false;
let taskId = "";
let taskModel = "1296";
let startedAt = 0;
let lines: string[] = [];
let lastPhase = "";
let modelsDirectoryCustomized = false;
let defaultModelsDirectory = "";

const models: Record<string, { filename: string; note: string }> = {
  "1296": { filename: "model_bs_roformer_ep_368_sdr_12.9628.ckpt", note: "输出人声与伴奏两轨，保留原始增益。" },
  "5hp": { filename: "5_HP-Karaoke-UVR.pth", note: "输出卡拉 OK 侧与移除侧两轨，可分别试听确认。" },
  "6hp": { filename: "6_HP-Karaoke-UVR.pth", note: "另一套 Karaoke 权重，输出卡拉 OK 侧与移除侧两轨。" },
  "deecho": { filename: "UVR-DeEcho-DeReverb.pth", note: "输出去混响音频，以及残余回声／混响两轨。" },
};

function updateModel(): void {
  const info = models[model.value];
  element("weights-name").textContent = `所需文件：${info.filename}`;
  element("model-note").textContent = info.note;
  updateControls();
}

function updateControls(): void {
  const busy = running || picking || !!library?.downloading;
  form.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select").forEach((node) => { node.disabled = busy; });
  form.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((node) => { node.disabled = busy || !connected; });
  startButton.disabled = busy || !connected || !library?.ready(model.value);
  startButton.textContent = running ? "正在分离…" : "开始分离 →";
  cancelButton.disabled = !running;
  form.setAttribute("aria-busy", String(running));
  library?.updateControls();
}

function elapsed(seconds: number): string {
  const count = Math.max(0, Math.floor(seconds));
  return `${Math.floor(count / 60).toString().padStart(2, "0")}:${(count % 60).toString().padStart(2, "0")}`;
}

function log(message: string, seconds: number): void {
  if (message === lastPhase) return;
  lastPhase = message;
  lines.push(`${elapsed(seconds)}  ${message}`);
  lines = lines.slice(-80);
  taskLog.textContent = lines.join("\n");
}

function showError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  phase.textContent = message;
  status.textContent = "处理失败";
  status.className = "status failed";
  progressBar.value = 0;
  element("percentage").textContent = "";
  log(message, running ? (performance.now() - startedAt) / 1000 : 0);
}

function savePaths(): void {
  try { localStorage.setItem("uvr.paths.v2", JSON.stringify({ models: modelsDirectoryCustomized ? fields.models.value : null, output: fields.output.value })); } catch { /* Remembering paths is optional. */ }
}

function showOutputs(paths: string[]): void {
  outputs.replaceChildren();
  element("output-count").textContent = `${paths.length} 个文件`;
  const names = taskModel === "1296" ? ["人声", "伴奏"] : taskModel === "deecho" ? ["去混响音频", "残余回声／混响"] : ["卡拉 OK 侧", "移除侧"];
  if (!paths.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "完成后，两条音轨的保存位置会显示在这里。";
    outputs.append(empty);
  }
  paths.forEach((path, index) => {
    const track = document.createElement("div");
    track.className = "output-track";
    const name = document.createElement("strong");
    name.textContent = names[index] ?? "输出音轨";
    const location = document.createElement("code");
    location.textContent = path;
    track.append(name, location);
    outputs.append(track);
  });
}

function receive(event: TaskEvent): void {
  if (event.id !== taskId) return;
  phase.textContent = event.phase;
  element("elapsed").textContent = elapsed(event.elapsedSeconds);
  if (event.fraction === null && event.status === "running") {
    progressBar.removeAttribute("value");
    element("percentage").textContent = "";
  } else {
    progressBar.value = event.fraction ?? 0;
    element("percentage").textContent = event.fraction === null ? "" : `${Math.round(event.fraction * 100)}%`;
  }
  status.textContent = { running: "正在处理", completed: "已完成", cancelled: "已取消", failed: "处理失败" }[event.status];
  status.className = `status ${event.status}`;
  log(event.phase, event.elapsedSeconds);
  if (event.status !== "running") {
    running = false;
    cancelButton.textContent = "取消任务";
    updateControls();
    showOutputs(event.outputs);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!bridge || !connected || running || picking || library?.downloading || !library?.ready(model.value) || !form.reportValidity()) return;
  running = true;
  taskId = crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  taskModel = model.value;
  startedAt = performance.now();
  lines = [];
  lastPhase = "";
  showOutputs([]);
  updateControls();
  savePaths();
  receive({ id: taskId, status: "running", phase: "准备任务", fraction: null, elapsedSeconds: 0, outputs: [] });
  try {
    await bridge.core.invoke("start_task", { request: { id: taskId, model: taskModel, input: fields.input.value, modelsDir: fields.models.value, outputDir: fields.output.value } });
  } catch (error) {
    showError(error);
    running = false;
    updateControls();
  }
});

cancelButton.addEventListener("click", async () => {
  if (!bridge || !running) return;
  cancelButton.disabled = true;
  cancelButton.textContent = "正在取消…";
  try { await bridge.core.invoke("cancel_task", { id: taskId }); }
  catch (error) { showError(error); cancelButton.disabled = false; cancelButton.textContent = "取消任务"; }
});

document.querySelectorAll<HTMLButtonElement>("[data-pick]").forEach((button) => {
  button.addEventListener("click", async () => {
    if (!bridge || picking || running) return;
    const kind = button.dataset.pick as keyof typeof fields;
    picking = true;
    updateControls();
    try {
      const selected = await bridge.core.invoke<string | null>("choose_path", { kind });
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
  if (!bridge) { element("preview-note").hidden = false; return; }
  try {
    await bridge.event.listen<TaskEvent>("task-progress", (event) => receive(event.payload));
    const defaults = await bridge.core.invoke<{ modelsDir: string | null }>("defaults");
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

void connect();
