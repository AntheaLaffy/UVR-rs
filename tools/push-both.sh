#!/usr/bin/env bash
set -euo pipefail

branch=$(git branch --show-current)
[[ -n "$branch" ]] || { printf '%s\n' 'detached HEAD; cannot choose a branch' >&2; exit 1; }

git push origin "$branch"
git push upstream "$branch"
