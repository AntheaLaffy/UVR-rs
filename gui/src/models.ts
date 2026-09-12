import type { Bridge } from "./bridge";

interface Entry {
  key: string; name: string; filename: string; sizeBytes: number;
  status: "ready" | "missing" | "invalid" | "error"; message: string;
}
interface DownloadEvent {
  id: string; model: string; status: "running" | "completed" | "cancelled" | "failed";
  phase: string; received: number; total: number; path: string | null;
}
interface Context {
  directory(): string; selected(): string; busy(): boolean;
  changed(): void; resetDirectory(): void;
}
function node<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing model control: ${id}`);
  return element as T;
}
function size(bytes: number): string { return `${(bytes / 1024 / 1024).toFixed(1)} MiB`; }

export class ModelLibrary {
  checking = false;
  downloading = false;
  private connected = false;
  private entries = new Map<string, Entry>();
  private directory = "";
  private generation = 0;
  private downloadId = "";
  private readonly panel = node<HTMLDetailsElement>("model-library");
  private readonly list = node("model-list");
  private readonly refresh = node<HTMLButtonElement>("check-models");
  private readonly reset = node<HTMLButtonElement>("reset-models");
  private readonly cancel = node<HTMLButtonElement>("cancel-download");
  private readonly source = node<HTMLSelectElement>("download-source");
  private readonly proxy = node<HTMLInputElement>("download-proxy");
  private readonly message = node("download-message");
  private readonly bar = node<HTMLProgressElement>("download-progress");

  constructor(private readonly bridge: Bridge, private readonly context: Context) {
    this.refresh.addEventListener("click", () => { void this.scan(); });
    this.reset.addEventListener("click", () => { context.resetDirectory(); void this.scan(); });
    this.cancel.addEventListener("click", async () => {
      this.cancel.disabled = true;
      try { await bridge.core.invoke("cancel_task", { id: this.downloadId }); }
      catch (error) { this.message.textContent = String(error); this.cancel.disabled = false; }
    });
    window.addEventListener("focus", () => {
      if (this.connected && !this.checking && !this.downloading && !context.busy()) void this.scan();
    });
  }

  async connect(): Promise<void> {
    await this.bridge.event.listen<DownloadEvent>("model-download", event => this.receive(event.payload));
    this.connected = true;
    await this.scan();
  }

  ready(key: string): boolean {
    return !this.checking && this.directory === this.context.directory()
      && this.entries.get(key)?.status === "ready";
  }

  updateControls(): void {
    const busy = this.context.busy() || this.downloading;
    this.refresh.disabled = !this.connected || busy || this.checking;
    this.reset.disabled = !this.connected || busy;
    this.source.disabled = busy;
    this.proxy.disabled = busy;
    this.cancel.disabled = !this.downloading;
    this.list.querySelectorAll<HTMLButtonElement>("[data-download]").forEach(button => {
      const entry = this.entries.get(button.dataset.download ?? "");
      button.disabled = !this.connected || busy || this.checking || !entry
        || entry.status === "ready" || entry.status === "error";
    });
    const selected = this.context.selected();
    this.list.querySelectorAll<HTMLElement>("[data-model]").forEach(row => {
      row.classList.toggle("selected", row.dataset.model === selected);
    });
    node("model-availability").textContent = this.checking ? "正在校验模型文件…"
      : this.ready(selected) ? "当前模型已就绪"
      : "当前模型尚未就绪，可在下方模型管理中检测或下载。";
  }

  async scan(): Promise<void> {
    if (!this.connected || this.downloading || this.context.busy()) return;
    const directory = this.context.directory();
    const generation = ++this.generation;
    this.checking = true;
    node("library-summary").textContent = "正在检测模型…";
    this.context.changed();
    try {
      const entries = await this.bridge.core.invoke<Entry[]>("inspect_models", { directory });
      if (generation !== this.generation || directory !== this.context.directory()) return;
      this.directory = directory;
      this.entries = new Map(entries.map(entry => [entry.key, entry]));
      this.render();
      const count = entries.filter(entry => entry.status === "ready").length;
      node("library-summary").textContent = `${count} / ${entries.length} 个模型已就绪`;
      if (this.entries.get(this.context.selected())?.status !== "ready") this.panel.open = true;
    } catch (error) {
      if (generation !== this.generation) return;
      this.entries.clear();
      this.directory = "";
      this.list.replaceChildren();
      node("library-summary").textContent = String(error);
      this.panel.open = true;
    } finally {
      if (generation === this.generation) { this.checking = false; this.context.changed(); }
    }
  }

  private render(): void {
    this.list.replaceChildren();
    for (const entry of this.entries.values()) {
      const row = document.createElement("div");
      row.className = "model-row";
      row.dataset.model = entry.key;
      row.dataset.availability = entry.status;
      const info = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${entry.name} · ${size(entry.sizeBytes)}`;
      const filename = document.createElement("code");
      filename.textContent = entry.filename;
      const availability = document.createElement("span");
      availability.className = `model-state ${entry.status}`;
      availability.textContent = entry.message;
      info.append(title, filename, availability);
      const action = document.createElement("button");
      action.type = "button";
      action.dataset.download = entry.key;
      action.textContent = entry.status === "ready" ? "已就绪" : entry.status === "invalid" ? "重新下载" : "下载";
      action.addEventListener("click", () => { void this.download(entry); });
      row.append(info, action);
      this.list.append(row);
    }
  }

  private async download(entry: Entry): Promise<void> {
    if (this.downloading || this.checking || this.context.busy() || !this.connected) return;
    this.downloading = true;
    this.downloadId = crypto.randomUUID?.() ?? `${Date.now().toString(36)}-download`;
    this.panel.open = true;
    this.message.textContent = `准备下载 ${entry.name}`;
    this.bar.removeAttribute("value");
    this.context.changed();
    try {
      await this.bridge.core.invoke("download_model", { request: {
        id: this.downloadId, model: entry.key, directory: this.context.directory(),
        source: this.source.value, proxy: this.proxy.value.trim() || null,
        replaceInvalid: entry.status === "invalid",
      } });
    } catch (error) {
      this.message.textContent = String(error);
      this.downloading = false;
      this.bar.value = 0;
      this.context.changed();
    }
  }

  private receive(event: DownloadEvent): void {
    if (event.id !== this.downloadId) return;
    this.message.textContent = event.status === "running"
      ? `${event.phase} · ${size(event.received)} / ${size(event.total)}` : event.phase;
    this.message.dataset.status = event.status;
    if (event.status === "running" && !event.received) this.bar.removeAttribute("value");
    else this.bar.value = event.total ? event.received / event.total : 0;
    if (event.status !== "running") {
      this.downloading = false;
      this.context.changed();
      void this.scan();
    }
  }
}
