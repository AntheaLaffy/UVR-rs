#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { printf 'Usage: %s VERSION\n' "$(basename "$0")" >&2; exit 2; }
exec python3 "$(dirname "${BASH_SOURCE[0]}")/scan-workspace.py" --version "$1"
