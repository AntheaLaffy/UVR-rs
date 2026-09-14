#!/usr/bin/env python3
"""Scan workspace crates and documentation, optionally synchronize a release version."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r'^(version = ")([^"\n]+)("\s*)$', re.MULTILINE)


def metadata() -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def workspace_version() -> str:
    text = (ROOT / "Cargo.toml").read_text()
    match = re.search(r'^version = "([^"]+)"\s*$', text, re.MULTILINE)
    if not match:
        raise SystemExit("workspace version not found in Cargo.toml")
    return match.group(1)


def files() -> dict[str, list[str]]:
    groups = {
        "docs": ["README*", "docs/**/*.md", "docs/**/*.txt"],
        "config": ["**/*.toml", "**/*.json", ".github/**/*.yml", ".github/**/*.yaml"],
        "tools": ["tools/*.sh", "tools/*.mjs", "tools/*.py"],
        "build": ["Makefile"],
    }
    result: dict[str, list[str]] = {}
    for name, patterns in groups.items():
        found = {
            str(path.relative_to(ROOT))
            for pattern in patterns
            for path in ROOT.glob(pattern)
            if path.is_file()
            and not {".git", "target", "node_modules", "agent", ".venv"}.intersection(path.parts)
        }
        result[name] = sorted(found)
    return result


def crates(data: dict) -> list[dict[str, str | bool]]:
    result = []
    for package in data["packages"]:
        manifest = Path(package["manifest_path"]).resolve().relative_to(ROOT)
        publish = package.get("publish") not in (False, [])
        result.append({"name": package["name"], "version": package["version"], "manifest": str(manifest), "publish": publish})
    return sorted(result, key=lambda item: str(item["name"]))


def update_version(old: str, new: str, package_list: list[dict]) -> int:
    changed = 0
    root = ROOT / "Cargo.toml"
    text = root.read_text()
    updated, count = re.subn(rf'^(version = "){re.escape(old)}("\s*)$', rf'\g<1>{new}\g<2>', text, count=1, flags=re.MULTILINE)
    if count:
        root.write_text(updated)
        changed += 1
    text = root.read_text()
    updated, count = re.subn(rf'(uvr-core\s*=\s*\{{[^}}]*?version\s*=\s*"){re.escape(old)}', rf'\g<1>{new}', text, count=1)
    if count:
        root.write_text(updated)
        changed += 1

    for package in package_list:
        manifest = ROOT / str(package["manifest"])
        readme = manifest.parent / "README.md"
        if not readme.is_file():
            continue
        text = readme.read_text()
        updated = text.replace(f'version = "{old}"', f'version = "{new}"')
        if updated != text:
            readme.write_text(updated)
            changed += 1
    subprocess.run(["cargo", "check", "--workspace", "--all-targets", "--offline"], cwd=ROOT, check=True)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="update the workspace and crate README references to VERSION")
    parser.add_argument("--json", action="store_true", help="emit machine-readable inventory")
    parser.add_argument("--packages", action="store_true", help="print publishable crate names, one per line")
    args = parser.parse_args()

    package_list = crates(metadata())
    old = workspace_version()
    if args.version:
        if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", args.version):
            raise SystemExit(f"invalid SemVer: {args.version}")
        if args.version != old:
            changed = update_version(old, args.version, package_list)
            old = args.version
            print(f"updated {changed} files: {old}")
    if args.packages:
        for package in package_list:
            if package["publish"]:
                print(package["name"])
        return
    inventory = {"version": old, "crates": package_list, "files": files()}
    if args.json:
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return
    print(f"workspace version: {old}")
    print(f"crates: {len(package_list)} ({sum(bool(p['publish']) for p in package_list)} publishable)")
    print(f"documentation files: {len(inventory['files']['docs'])}")
    print(f"configuration files: {len(inventory['files']['config'])}")
    print(f"tool scripts: {len(inventory['files']['tools'])}")
    print(f"build entrypoints: {len(inventory['files']['build'])}")
    for package in package_list:
        marker = "publish" if package["publish"] else "private"
        print(f"  {package['name']} {package['version']} [{marker}] -> {package['manifest']}")


if __name__ == "__main__":
    main()
