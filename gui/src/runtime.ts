export type RuntimeBackend = "burn" | "openvino-cpu";
type LinearLayout = "flattened" | "batched";
interface VrOptions { windowFrames: number; inferenceBatch: number; windowParallelism: number }
interface RoformerOptions { timeBatch: number; frequencyBatch: number; windowParallelism: number; linearLayout: LinearLayout }
export interface RuntimeDefaults {
  backends: RuntimeBackend[];
  roformerBackend: RuntimeBackend;
  threads: number;
  vr: VrOptions;
  roformer: RoformerOptions;
}
export interface RuntimeRequest {
  backend: RuntimeBackend;
  threads: number;
  vr?: VrOptions;
  roformer?: RoformerOptions;
}
interface Preferences {
  backend: RuntimeBackend;
  threads: number;
  vr: Record<string, VrOptions>;
  roformer: RoformerOptions;
}

// Desktop defaults come from the shared runtime; these values also make web previews usable.
const previewDefaults: RuntimeDefaults = {
  backends: ["burn"],
  roformerBackend: "burn",
  threads: 2,
  vr: { windowFrames: 512, inferenceBatch: 1, windowParallelism: 4 },
  roformer: { timeBatch: 62, frequencyBatch: 301, windowParallelism: 1, linearLayout: "flattened" },
};
const storageKey = "uvr.runtime.v1";

function element<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing element: ${id}`);
  return node as T;
}

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" ? value as Record<string, unknown> : {};
}

function integer(value: unknown, fallback: number, min = 1, max = Number.MAX_SAFE_INTEGER, step = 1): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= min && value <= max && value % step === 0 ? value : fallback;
}

function preferences(defaults: RuntimeDefaults, saved: unknown): Preferences {
  const stored = object(saved);
  const vr = object(stored.vr);
  const roformer = object(stored.roformer);
  const result: Preferences = {
    backend: (stored.backend === "burn" || stored.backend === "openvino-cpu") && defaults.backends.includes(stored.backend) ? stored.backend : defaults.roformerBackend,
    threads: integer(stored.threads, defaults.threads),
    vr: {},
    roformer: {
      timeBatch: integer(roformer.timeBatch, defaults.roformer.timeBatch),
      frequencyBatch: integer(roformer.frequencyBatch, defaults.roformer.frequencyBatch),
      windowParallelism: integer(roformer.windowParallelism, defaults.roformer.windowParallelism, 1, 8),
      linearLayout: roformer.linearLayout === "batched" || roformer.linearLayout === "flattened" ? roformer.linearLayout : defaults.roformer.linearLayout,
    },
  };
  for (const model of ["5hp", "6hp", "deecho"]) {
    const options = object(vr[model]);
    const deecho = model === "deecho";
    result.vr[model] = {
      windowFrames: integer(options.windowFrames, defaults.vr.windowFrames, deecho ? 144 : 272, 2048, 16),
      inferenceBatch: deecho ? 1 : integer(options.inferenceBatch, defaults.vr.inferenceBatch, 1, 4),
      windowParallelism: deecho ? 1 : integer(options.windowParallelism, defaults.vr.windowParallelism, 1, 8),
    };
  }
  return result;
}

export class RuntimeSettings {
  private readonly panel = element("runtime-settings");
  private readonly backend = element<HTMLSelectElement>("runtime-backend");
  private readonly threads = element<HTMLInputElement>("runtime-threads");
  private readonly advanced = element<HTMLDetailsElement>("runtime-advanced");
  private readonly reset = element<HTMLButtonElement>("runtime-reset");
  private readonly vr = {
    windowFrames: element<HTMLInputElement>("vr-window-frames"),
    inferenceBatch: element<HTMLInputElement>("vr-inference-batch"),
    windowParallelism: element<HTMLInputElement>("vr-window-parallelism"),
  };
  private readonly roformer = {
    timeBatch: element<HTMLInputElement>("roformer-time-batch"),
    frequencyBatch: element<HTMLInputElement>("roformer-frequency-batch"),
    windowParallelism: element<HTMLInputElement>("roformer-window-parallelism"),
    linearLayout: element<HTMLSelectElement>("roformer-linear-layout"),
  };
  private defaults = previewDefaults;
  private stored: unknown;
  private state: Preferences;
  private model = "1296";
  private busy = false;

  constructor() {
    try { this.stored = JSON.parse(localStorage.getItem(storageKey) ?? "{}"); } catch { this.stored = {}; }
    this.state = preferences(this.defaults, this.stored);
    this.panel.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select").forEach((field) => {
      field.addEventListener("input", () => {
        if (field === this.backend) this.state.backend = this.backend.value as RuntimeBackend;
        this.updateAvailability();
        this.save();
        this.updateSummary();
      });
    });
    this.reset.addEventListener("click", () => {
      const recommended = preferences(this.defaults, {});
      this.state.threads = recommended.threads;
      if (this.model === "1296") {
        this.state.backend = recommended.backend;
        this.state.roformer = recommended.roformer;
      } else {
        this.state.vr[this.model] = recommended.vr[this.model];
      }
      this.populate();
      this.updateAvailability();
      this.save();
      this.updateSummary();
    });
    this.populate();
    this.update("1296", false);
    onLanguageChange(() => {
      const recommended = this.model === "1296" ? this.defaults.roformerBackend : "burn";
      for (const option of this.backend.options) {
        option.textContent = (option.value === "burn" ? "Burn CPU" : "OpenVINO CPU") + (option.value === recommended ? t("runtime.default") : "");
      }
      this.updateAvailability();
      this.updateSummary();
    });
  }

  configure(defaults: RuntimeDefaults): void {
    this.defaults = defaults;
    this.state = preferences(defaults, this.stored);
    this.populate();
    this.update(this.model, this.busy);
  }

  update(model: string, busy: boolean): void {
    if (model !== this.model) {
      this.model = model;
      this.populate();
    }
    this.busy = busy;
    this.updateAvailability();
    this.updateSummary();
  }

  reportValidity(): boolean {
    const invalid = this.controls().find((field) => !field.disabled && !field.validity.valid);
    if (!invalid) return true;
    if (this.advanced.contains(invalid)) this.advanced.open = true;
    invalid.reportValidity();
    return false;
  }

  request(): RuntimeRequest {
    const request: RuntimeRequest = { backend: this.backend.value as RuntimeBackend, threads: this.threads.valueAsNumber };
    if (request.backend === "openvino-cpu") return request;
    if (this.model === "1296") {
      request.roformer = {
        timeBatch: this.roformer.timeBatch.valueAsNumber,
        frequencyBatch: this.roformer.frequencyBatch.valueAsNumber,
        windowParallelism: this.roformer.windowParallelism.valueAsNumber,
        linearLayout: this.roformer.linearLayout.value as LinearLayout,
      };
    } else {
      const batch = this.model === "deecho" ? 1 : this.vr.inferenceBatch.valueAsNumber;
      request.vr = {
        windowFrames: this.vr.windowFrames.valueAsNumber,
        inferenceBatch: batch,
        windowParallelism: this.model === "deecho" || batch > 1 ? 1 : this.vr.windowParallelism.valueAsNumber,
      };
    }
    return request;
  }

  describe(request: RuntimeRequest = this.request()): string {
    const parts = [request.backend === "burn" ? "Burn CPU" : "OpenVINO CPU", t("runtime.summary.threads", { count: request.threads })];
    if (request.vr) {
      parts.push(t("runtime.summary.frames", { count: request.vr.windowFrames }), t("runtime.summary.batch", { count: request.vr.inferenceBatch }), t("runtime.summary.parallel", { count: request.vr.windowParallelism }));
    }
    if (request.roformer) {
      parts.push(t("runtime.summary.time", { count: request.roformer.timeBatch }), t("runtime.summary.frequency", { count: request.roformer.frequencyBatch }), t("runtime.summary.parallel", { count: request.roformer.windowParallelism }), t(`runtime.${request.roformer.linearLayout}`));
    }
    return parts.join(" · ");
  }

  private controls(): Array<HTMLInputElement | HTMLSelectElement> {
    return Array.from(this.panel.querySelectorAll<HTMLInputElement | HTMLSelectElement>("input, select"));
  }

  private populate(): void {
    this.backend.replaceChildren();
    for (const value of this.defaults.backends) {
      if (value === "openvino-cpu" && this.model !== "1296") continue;
      const option = document.createElement("option");
      option.value = value;
      const recommended = this.model === "1296" ? this.defaults.roformerBackend : "burn";
      option.textContent = (value === "burn" ? "Burn CPU" : "OpenVINO CPU") + (value === recommended ? t("runtime.default") : "");
      this.backend.append(option);
    }
    this.backend.value = this.model === "1296" ? this.state.backend : "burn";
    this.threads.value = String(this.state.threads);
    const vr = this.state.vr[this.model] ?? this.state.vr["5hp"];
    for (const key of ["windowFrames", "inferenceBatch", "windowParallelism"] as const) this.vr[key].value = String(vr[key]);
    for (const key of ["timeBatch", "frequencyBatch", "windowParallelism", "linearLayout"] as const) this.roformer[key].value = String(this.state.roformer[key]);
  }

  private updateAvailability(): void {
    const roformer = this.model === "1296";
    const openvino = this.backend.value === "openvino-cpu";
    const deecho = this.model === "deecho";
    this.controls().forEach((field) => { field.disabled = this.busy; });
    this.backend.disabled = this.busy || this.backend.options.length < 2;
    this.reset.disabled = this.busy;
    this.advanced.hidden = openvino;
    element("runtime-vr-options").hidden = roformer;
    element("runtime-roformer-options").hidden = !roformer;
    for (const field of Object.values(this.vr)) field.disabled = this.busy || roformer;
    for (const field of Object.values(this.roformer)) field.disabled = this.busy || !roformer || openvino;
    this.vr.windowFrames.min = deecho ? "144" : "272";
    this.vr.inferenceBatch.disabled ||= deecho;
    this.vr.windowParallelism.disabled ||= deecho || this.vr.inferenceBatch.valueAsNumber > 1;
    element("vr-parallelism-note").textContent = t(deecho ? "runtime.deechoHint" : this.vr.inferenceBatch.valueAsNumber > 1 ? "runtime.batchHint" : "runtime.parallelHint");
    element("vr-window-note").textContent = t("runtime.windowHint", { min: deecho ? 144 : 272 });
    element("runtime-note").textContent = t(openvino ? "runtime.openvinoHint" : roformer ? "runtime.burnHint" : "runtime.vrHint");
  }

  private save(): void {
    if (this.controls().some((field) => !field.disabled && !field.validity.valid)) return;
    this.state.threads = this.threads.valueAsNumber;
    const request = this.request();
    if (request.roformer) this.state.roformer = request.roformer;
    if (request.vr) {
      // Keep the user's window concurrency when batching temporarily makes it inactive.
      this.state.vr[this.model] = { ...request.vr, windowParallelism: integer(this.vr.windowParallelism.valueAsNumber, this.state.vr[this.model].windowParallelism, 1, 8) };
    }
    this.stored = structuredClone(this.state);
    try { localStorage.setItem(storageKey, JSON.stringify(this.state)); } catch { /* Remembering settings is optional. */ }
  }

  private updateSummary(): void {
    const valid = this.controls().every((field) => field.disabled || field.validity.valid);
    element("runtime-summary").textContent = valid ? this.describe() : t("runtime.invalid");
  }
}
import { onLanguageChange, t } from "./i18n";
