import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const [platform, binaryArgument, wheelArgument, ...extra] = process.argv.slice(2);
if (!["linux-x86_64", "windows-x86_64"].includes(platform) || !binaryArgument || extra.length || (wheelArgument && platform !== "linux-x86_64")) {
  throw new Error("Usage: node tools/package-ci.mjs <linux-x86_64|windows-x86_64> <release-directory> [openvino-wheel-directory]");
}
if (/target-cpu[=\s]+native/.test(`${process.env.RUSTFLAGS ?? ""} ${process.env.CARGO_ENCODED_RUSTFLAGS ?? ""}`)) {
  throw new Error("CI artifacts must not be compiled with target-cpu=native.");
}
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const binaryDirectory = resolve(binaryArgument);
const output = join(root, "target", "ci-artifacts");
mkdirSync(join(root, "target"), { recursive: true });
const temporary = mkdtempSync(join(root, "target", "ci-package-"));
const openvinoVersion = "2026.3.1";

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} exited with ${result.status}`);
}
function digest(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}
function files(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? files(path) : [path];
  }).sort();
}
function archive(stage, name, category, manifestName = "SHA256SUMS") {
  const destination = join(output, category);
  mkdirSync(destination, { recursive: true });
  const manifest = files(stage).map(path => `${digest(path)}  ${relative(stage, path).replaceAll("\\", "/")}`).join("\n") + "\n";
  writeFileSync(join(stage, manifestName), manifest);
  const filename = `${name}.tar.gz`;
  const archivePath = join(destination, filename);
  if (existsSync(archivePath)) throw new Error(`Refusing to replace an existing artifact: ${archivePath}`);
  // Tar keeps Linux executable modes intact inside the Actions artifact ZIP.
  run("tar", ["-czf", archivePath, "-C", dirname(stage), "UVR"]);
  writeFileSync(join(destination, "SHA256SUMS"), `${digest(archivePath)}  ${filename}\n`);
  console.log(`Created ${archivePath}`);
}

try {
  const stage = join(temporary, "base", "UVR");
  mkdirSync(join(stage, "models"), { recursive: true });
  const extension = platform.startsWith("windows") ? ".exe" : "";
  for (const name of [`uvr${extension}`, `uvr-gui${extension}`]) {
    const source = join(binaryDirectory, name);
    if (!statSync(source).isFile()) throw new Error(`Missing executable: ${source}`);
    copyFileSync(source, join(stage, name));
    if (!extension) chmodSync(join(stage, name), 0o755);
  }
  copyFileSync(join(root, "references", "model-downloads.json"), join(stage, "model-downloads.json"));
  writeFileSync(join(stage, "models", "README.txt"), "Put original model weights here, or select/download them in the GUI. Model weights are not included.\n");
  const systemRequirements = extension
    ? "Windows x86_64: Microsoft Edge WebView2 Runtime and Microsoft Visual C++ 2015-2022 x64 Redistributable are required. This Burn-only build has not completed real-device Windows audio acceptance testing."
    : "Linux x86_64: built on Ubuntu 24.04 (glibc 2.39). The GUI requires WebKitGTK 4.1 and GTK 3 (Ubuntu packages: libwebkit2gtk-4.1-0, libgtk-3-0t64, libayatana-appindicator3-1). Other distributions and older glibc versions are not validated.";
  const optionalRuntime = extension
    ? "Inference uses Burn CPU for all four models."
    : `The lightweight package works with Burn CPU. To enable OpenVINO CPU for 1296, extract uvr-openvino-cpu-${openvinoVersion}-linux-x86_64.tar.gz into the same parent directory; it adds UVR/lib beside both executables. VR continues to use Burn. Keep lib beside the executables.`;
  writeFileSync(join(stage, "README.txt"), `UVR CLI + GUI (${platform})\n\nExtract this tar.gz first, then launch uvr-gui${extension}, or run uvr${extension} --help.\n\n${systemRequirements}\n\n${optionalRuntime}\n\nThese binaries use the generic x86-64 CPU target, not the hosted runner's native ISA. No model weights or Python interpreter are included. CI checks builds and ordinary tests; it does not replace real-model audio acceptance testing.\n\nSHA256SUMS inside this directory covers the included files. The outer SHA256SUMS covers the archive.\n`);
  writeFileSync(join(stage, "build-info.json"), JSON.stringify({
    platform, commit: process.env.GITHUB_SHA ?? "local", runId: process.env.GITHUB_RUN_ID ?? null,
    rustflags: process.env.RUSTFLAGS ?? null, runtime: extension ? "burn" : "burn + optional OpenVINO CPU",
    modelWeightsIncluded: false, pythonIncluded: false, windowsAudioAcceptance: "not completed",
  }, null, 2) + "\n");
  archive(stage, `uvr-${platform}`, platform);

  if (wheelArgument) {
    const wheel = resolve(wheelArgument);
    const library = join(wheel, "openvino", "libs");
    const runtimeStage = join(temporary, "openvino", "UVR");
    const destination = join(runtimeStage, "lib");
    mkdirSync(destination, { recursive: true });
    const cpuLibrary = /^(libopenvino(?:_c|_ir_frontend)?\.so(?:\..*)?|libopenvino_intel_cpu_plugin\.so|libtbb[^/]*\.so(?:\..*)?|libhwloc\.so(?:\..*)?)$/;
    for (const name of readdirSync(library).filter(name => cpuLibrary.test(name))) copyFileSync(join(library, name), join(destination, name));
    if (!existsSync(join(destination, "libopenvino_c.so"))) {
      const candidates = readdirSync(destination).filter(name => /^libopenvino_c\.so\./.test(name));
      if (candidates.length !== 1) throw new Error("Expected exactly one OpenVINO C API library in the wheel.");
      copyFileSync(join(destination, candidates[0]), join(destination, "libopenvino_c.so"));
    }
    for (const pattern of [/^libopenvino\.so/, /^libopenvino_ir_frontend\.so/, /^libopenvino_intel_cpu_plugin\.so$/, /^libtbb\.so/]) {
      if (!readdirSync(destination).some(name => pattern.test(name))) throw new Error(`Missing CPU runtime dependency: ${pattern}`);
    }
    const distribution = join(wheel, `openvino-${openvinoVersion}.dist-info`);
    const licensePaths = readdirSync(distribution).filter(name => /^(licen[cs]es?|notices?)(\.|$)/i.test(name));
    if (!licensePaths.length) throw new Error("The OpenVINO wheel must provide its license and third-party notices.");
    const licenses = join(runtimeStage, "licenses", `openvino-${openvinoVersion}`);
    mkdirSync(licenses, { recursive: true });
    for (const name of licensePaths) cpSync(join(distribution, name), join(licenses, name), { recursive: true });
    writeFileSync(join(runtimeStage, "OPENVINO-README.txt"), `Optional OpenVINO ${openvinoVersion} CPU runtime for UVR on Linux x86_64.\n\nExtract alongside the lightweight archive so these libraries become UVR/lib/. The application discovers this directory automatically; Python and environment activation are unnecessary. Only the CPU plugin and its native dependencies are included. Model weights are external. Third-party notices are in licenses/. OPENVINO-SHA256SUMS verifies these additions without replacing the base package's SHA256SUMS.\n`);
    archive(runtimeStage, `uvr-openvino-cpu-${openvinoVersion}-linux-x86_64`, "openvino-cpu-linux-x86_64", "OPENVINO-SHA256SUMS");
  }
} finally {
  // This exact directory was created by mkdtemp above and contains only our staging files.
  rmSync(temporary, { recursive: true, force: true });
}
