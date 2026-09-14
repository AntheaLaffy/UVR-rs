#!/bin/bash
set -euo pipefail

uvr_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
uvr_output=""
uvr_models=""
while (($#)); do
    case "$1" in
        --output|--models-dir)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            if [[ "$1" == --output ]]; then uvr_output=$2; else uvr_models=$2; fi
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash tools/package-portable.sh --output <archive.zip> [--models-dir <directory>]"
            echo "Builds the Linux GUI. Models stay external; omitting --models-dir creates a light archive."
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$uvr_output" || "$uvr_output" != *.zip ]]; then
    echo "Specify a new .zip archive with --output." >&2; exit 2
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "This first portable package is validated for Linux x86_64." >&2; exit 1
fi
for uvr_tool in cargo pnpm jq zip sha256sum; do command -v "$uvr_tool" >/dev/null; done
uvr_output=$(realpath -m -- "$uvr_output")
if [[ -e "$uvr_output" || -L "$uvr_output" ]]; then echo "Archive already exists: $uvr_output" >&2; exit 1; fi
if [[ -n "$uvr_models" ]]; then
    uvr_models=$(realpath -e -- "$uvr_models")
    if [[ ! -d "$uvr_models" ]]; then echo "Model directory does not exist." >&2; exit 1; fi
fi
uvr_parent=$(dirname -- "$uvr_output")
mkdir -p -- "$uvr_parent"
uvr_stage=$(mktemp -d "$uvr_parent/.uvr-package.XXXXXX")
trap 'rm -rf -- "$uvr_stage"' EXIT
mkdir -p -- "$uvr_stage/UVR/models"
cd -- "$uvr_root"
pnpm build
cargo build --release --locked -p uvr-gui --features custom-protocol
cp --preserve=mode -- target/release/uvr-gui "$uvr_stage/UVR/uvr-gui"
cp -- docs/portable-readme.txt "$uvr_stage/UVR/README.txt"
cp -- LICENSE THIRD_PARTY_NOTICES.md "$uvr_stage/UVR/"
cp -- references/model-downloads.json "$uvr_stage/UVR/model-downloads.json"
jq -r '.models[] | [.file, .sha256, .size_bytes] | @tsv' references/verified-weights.json > "$uvr_stage/models.tsv"
uvr_count=0
while IFS=$'\t' read -r uvr_name uvr_sha uvr_size; do
    if [[ -z "$uvr_models" || ! -e "$uvr_models/$uvr_name" ]]; then continue; fi
    if [[ ! -f "$uvr_models/$uvr_name" ]]; then echo "Not a model file: $uvr_name" >&2; exit 1; fi
    # Dereference model links so the resulting archive works on another machine.
    cp -L -- "$uvr_models/$uvr_name" "$uvr_stage/UVR/models/$uvr_name"
    if [[ $(stat -c %s -- "$uvr_stage/UVR/models/$uvr_name") != "$uvr_size" ]]; then
        echo "Model size mismatch: $uvr_name" >&2; exit 1
    fi
    printf '%s  %s\n' "$uvr_sha" "$uvr_name" >> "$uvr_stage/UVR/models/SHA256SUMS"
    uvr_count=$((uvr_count + 1))
done < "$uvr_stage/models.tsv"
if ((uvr_count)); then (cd -- "$uvr_stage/UVR/models" && sha256sum --check --strict SHA256SUMS); fi
printf '此目录用于存放外部模型。可在 GUI 中按需下载，也可选择自定义目录或目录软链接。\n' > "$uvr_stage/UVR/models/README.txt"
(cd -- "$uvr_stage/UVR" && sha256sum uvr-gui > SHA256SUMS)
(cd -- "$uvr_stage" && zip -q -r package.zip UVR)
# Both paths share a filesystem. Hard-link publication refuses an existing archive.
ln -- "$uvr_stage/package.zip" "$uvr_output"
echo "Created $uvr_output (external models: $uvr_count / 4)"
