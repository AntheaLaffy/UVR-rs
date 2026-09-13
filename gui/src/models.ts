import type { Bridge } from "./bridge";
import { getLanguage, onLanguageChange, phaseText, t, type TranslatedPhase } from "./i18n";

interface Entry {
  key: string; name: string; filename: string; sizeBytes: number;
  status: "ready" | "missing" | "invalid" | "error"; message: string;
}
interface DownloadEvent extends TranslatedPhase {
  id: string; model: string; status: "running" | "completed" | "cancelled" | "failed";
  received: number; total: number; path: string | null;
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
  private summaryText: () => string = () => t("library.initial");
  private messageText: () => string = () => t("download.initial");
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
      try { await bridge.core.invoke("cancel_task", { id: this.downloadId, lang: getLanguage() }); }
      catch (error) { this.setMessage(() => t("download.phase.error", { error: String(error) })); this.cancel.disabled = false; }
    });
    window.addEventListener("focus", () => {
      if (this.connected && !this.checking && !this.downloading && !context.busy()) void this.scan();
    });
    onLanguageChange(() => {
      this.render();
      node("library-summary").textContent = this.checking ? t("library.checking") : this.summaryText();
      this.message.textContent = this.messageText();
      this.updateControls();
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
    node("model-availability").textContent = t(this.checking ? "library.verifying"
      : this.ready(selected) ? "library.available" : "library.unavailable");
  }

  async scan(): Promise<void> {
    if (!this.connected || this.downloading || this.context.busy()) return;
    const directory = this.context.directory();
    const generation = ++this.generation;
    this.checking = true;
    node("library-summary").textContent = t("library.checking");
    this.context.changed();
    try {
      const entries = await this.bridge.core.invoke<Entry[]>("inspect_models", { directory, lang: getLanguage() });
      if (generation !== this.generation || directory !== this.context.directory()) return;
      this.directory = directory;
      this.entries = new Map(entries.map(entry => [entry.key, entry]));
      this.render();
      const count = entries.filter(entry => entry.status === "ready").length;
      this.summaryText = () => t("library.count", { count, total: entries.length });
      node("library-summary").textContent = this.summaryText();
      if (this.entries.get(this.context.selected())?.status !== "ready") this.panel.open = true;
    } catch (error) {
      if (generation !== this.generation) return;
      this.entries.clear();
      this.directory = "";
      this.list.replaceChildren();
      this.summaryText = () => String(error);
      node("library-summary").textContent = this.summaryText();
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
      availability.textContent = t(`library.${entry.status}`);
      if (entry.status === "invalid" || entry.status === "error") availability.title = entry.message;
      info.append(title, filename, availability);
      const action = document.createElement("button");
      action.type = "button";
      action.dataset.download = entry.key;
      action.textContent = t(entry.status === "ready" ? "library.actionReady" : entry.status === "invalid" ? "library.redownload" : "library.download");
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
    this.setMessage(() => t("download.prepare", { model: entry.name }));
    this.bar.removeAttribute("value");
    this.context.changed();
    try {
      await this.bridge.core.invoke("download_model", { request: {
        id: this.downloadId, model: entry.key, directory: this.context.directory(),
        source: this.source.value, proxy: this.proxy.value.trim() || null,
        replaceInvalid: entry.status === "invalid", lang: getLanguage(),
      } });
    } catch (error) {
      this.setMessage(() => t("download.phase.error", { error: String(error) }));
      this.downloading = false;
      this.bar.value = 0;
      this.context.changed();
    }
  }

  private receive(event: DownloadEvent): void {
    if (event.id !== this.downloadId) return;
    this.setMessage(() => event.status === "running"
      ? `${phaseText(event)} · ${size(event.received)} / ${size(event.total)}` : phaseText(event));
    this.message.dataset.status = event.status;
    if (event.status === "running" && !event.received) this.bar.removeAttribute("value");
    else this.bar.value = event.total ? event.received / event.total : 0;
    if (event.status !== "running") {
      this.downloading = false;
      this.context.changed();
      void this.scan();
    }
  }

  private setMessage(text: () => string): void {
    this.messageText = text;
    this.message.textContent = text();
  }
}
