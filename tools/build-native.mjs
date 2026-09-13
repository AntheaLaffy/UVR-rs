import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { cpus } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Keep host ISA artifacts separate so a later portable build cannot reuse them.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const burnOnly = process.argv.slice(2).includes("--burn-only");
// Separate outputs keep a Burn-only distribution free of previously staged libraries.
const target = join(root, "target", burnOnly ? "native-burn" : "native");
const temporary = join(target, "tmp");
mkdirSync(temporary, { recursive: true });
if (process.argv.slice(2).some((arg) => arg !== "--burn-only")) {
  throw new Error("Usage: pnpm build:native [--burn-only]");
}
if (process.platform !== "linux" || process.arch !== "x64") {
  throw new Error("The measured native build targets Linux x86_64. Use pnpm gui:build for a portable build.");
}
const library = resolve(process.env.UVR_OPENVINO_LIB_DIR ?? join(root, ".local", "openvino-2026.3.1", "lib"));
if (!burnOnly && !existsSync(join(library, "libopenvino_c.so"))) {
  throw new Error("Set UVR_OPENVINO_LIB_DIR to the OpenVINO CPU library directory (including libopenvino_c.so), or pass --burn-only.");
}
const environment = {
  ...process.env,
  CARGO_TARGET_DIR: target,
  TMPDIR: temporary,
  CARGO_PROFILE_RELEASE_STRIP: "symbols",
  RUSTFLAGS: `${process.env.RUSTFLAGS ?? ""} -C target-cpu=native`.trim(),
};
if (process.env.CARGO_ENCODED_RUSTFLAGS) {
  throw new Error("Unset CARGO_ENCODED_RUSTFLAGS so target-cpu=native can take effect.");
}
function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env: environment, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
const features = burnOnly ? [] : ["--features", "openvino"];
run("cargo", ["build", "--release", "--locked", "-p", "uvr-cli", ...features]);
run("pnpm", ["--filter", "@uvr/gui", "tauri", "build", "--no-bundle", ...features, "--", "--locked"]);

if (!burnOnly) {
  const destination = join(target, "release", "lib");
  mkdirSync(destination, { recursive: true });
  // The wheel's $ORIGIN rpaths let both executables load this directory directly.
  const cpuLibrary = /^(libopenvino(?:_c|_ir_frontend)?\.so(?:\..*)?|libopenvino_intel_cpu_plugin\.so|libtbb[^/]*\.so(?:\..*)?|libhwloc\.so(?:\..*)?)$/;
  for (const name of readdirSync(library).filter((name) => cpuLibrary.test(name))) {
    copyFileSync(join(library, name), join(destination, name));
  }
}
console.log(`Native CLI: ${join(target, "release", "uvr")}`);
console.log(`Native GUI: ${join(target, "release", "uvr-gui")}`);
console.log(burnOnly ? "Built for this CPU; no OpenVINO libraries are required."
  : "Built for this CPU; keep the lib directory beside the executables when using OpenVINO.");
const artifacts = ["uvr", "uvr-gui"].map((name) => {
  const contents = readFileSync(join(target, "release", name));
  return { name, bytes: contents.length, sha256: createHash("sha256").update(contents).digest("hex") };
});
const metadata = {
  builtAt: new Date().toISOString(), platform: process.platform, architecture: process.arch,
  cpu: cpus()[0]?.model, rustflags: environment.RUSTFLAGS, strip: environment.CARGO_PROFILE_RELEASE_STRIP,
  openvino: !burnOnly, artifacts,
};
writeFileSync(join(target, "release", "build-info.json"), JSON.stringify(metadata, null, 2) + "\n");
