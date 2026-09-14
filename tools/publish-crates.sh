#!/usr/bin/env bash
set -euo pipefail

# Publish workspace library crates in dependency order. Versions are read from
# Cargo.toml; this script never changes versions or bypasses cargo's checks.

usage() {
    cat <<'EOF'
Usage: tools/publish-crates.sh [--dry-run] [--allow-dirty]

Publishes the crates.io packages that are maintained in this workspace:
uvr-core, uvr-dsp, uvr-models, uvr-vr, uvr-roformer, uvr-runtime.
Already-published versions are reported and skipped.
EOF
}

dry_run=0
allow_dirty=0
while (($#)); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --allow-dirty) allow_dirty=1 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if ((allow_dirty == 0)) && [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' 'working tree is dirty; commit changes or pass --allow-dirty' >&2
    exit 1
fi

version=$(sed -n 's/^version = "\([^"]*\)"$/\1/p' Cargo.toml | head -1)
[[ -n "$version" ]] || { printf '%s\n' 'cannot read workspace version' >&2; exit 1; }

mapfile -t packages < <(python3 tools/scan-workspace.py --packages)
for package in "${packages[@]}"; do
    printf '== %s@%s ==\n' "$package" "$version"
    if curl --fail --silent --show-error \
        "https://crates.io/api/v1/crates/$package/$version" >/dev/null 2>&1; then
        printf 'already published; skipping\n'
        continue
    fi

    args=(cargo publish -p "$package" --locked)
    ((dry_run)) && args+=(--dry-run)
    ((allow_dirty)) && args+=(--allow-dirty)
    "${args[@]}"
done
