// Apply saved appearance before the app stylesheet loads to avoid a theme flash.
(() => {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem("uvr.appearance.v1") ?? "{}") ?? {}; } catch { /* Optional preferences. */ }
  const mode = ["system", "light", "dark"].includes(saved.mode) ? saved.mode : "system";
  const theme = ["violet", "blue", "teal"].includes(saved.theme) ? saved.theme : "violet";
  const root = document.documentElement;
  root.dataset.colorMode = mode;
  root.dataset.colorScheme = mode === "system" ? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light") : mode;
  root.dataset.theme = theme;
})();
