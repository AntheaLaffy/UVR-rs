export {};

const root = document.documentElement;
const mode = document.getElementById("color-mode") as HTMLSelectElement;
const theme = document.getElementById("color-theme") as HTMLSelectElement;
const toggle = document.getElementById("appearance-toggle") as HTMLButtonElement;
const menu = document.getElementById("appearance-menu") as HTMLDivElement;
const system = matchMedia("(prefers-color-scheme: dark)");

function apply(): void {
  root.dataset.colorMode = mode.value;
  root.dataset.colorScheme = mode.value === "system" ? (system.matches ? "dark" : "light") : mode.value;
  root.dataset.theme = theme.value;
}

function remember(): void {
  apply();
  try { localStorage.setItem("uvr.appearance.v1", JSON.stringify({ mode: mode.value, theme: theme.value })); } catch { /* Remembering appearance is optional. */ }
}

function close(): void {
  menu.hidden = true;
  toggle.setAttribute("aria-expanded", "false");
}

mode.value = root.dataset.colorMode ?? "system";
theme.value = root.dataset.theme ?? "violet";
apply();
mode.addEventListener("change", remember);
theme.addEventListener("change", remember);
system.addEventListener("change", apply);
toggle.addEventListener("click", () => {
  menu.hidden = !menu.hidden;
  toggle.setAttribute("aria-expanded", String(!menu.hidden));
  if (!menu.hidden) mode.focus();
});
document.addEventListener("pointerdown", event => {
  if (event.target instanceof Node && !menu.contains(event.target) && !toggle.contains(event.target)) close();
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !menu.hidden) {
    close();
    toggle.focus();
  }
});
