# Workspace maintenance

This repository keeps release metadata and maintenance entry points under version control so that adding a crate does not require editing several independent lists. The workspace manifest remains the source of truth for the release version; scripts discover crates through `cargo metadata` and scan the tracked documentation, configuration, tools and Makefile entry points.

## Routine workflow

Run the inventory before and after a structural change:

```sh
make scan-workspace
python3 tools/scan-workspace.py --json
```

The scanner reports workspace crates, publishability, documentation files, configuration files, tool scripts and build entry points. Publishable crates are emitted in dependency order for the publishing script.

To prepare a release, update the version once at the workspace root:

```sh
make bump-version VERSION=0.1.3
```

`tools/bump-version.sh` delegates to the scanner. It updates the workspace manifest, internal `uvr-core` dependency references and crate README examples, then runs an offline workspace check so the lockfile and manifests stay consistent. Keep the working tree clean unless the command is deliberately being used during an in-progress change.

The same command also synchronizes the Tauri manifest, the GUI npm package and the frontend version constant. Run `make check-version` (or `python3 tools/scan-workspace.py --check-version`) in CI or before packaging to reject version drift. The CLI exposes the build version with `uvr --version`; `uvr update-check` and the GUI footer provide an explicit, non-automatic GitHub Releases check.

Publish and synchronize only after reviewing the diff:

```sh
tools/publish-crates.sh --dry-run
make publish-crates
make push-both
```

These commands are repository-owner release operations. The configured `origin`/`upstream` remotes and the crates.io namespace belong to the maintainer; other contributors should not run `make publish-crates` or `make push-both` against their own fork unless they have explicitly configured an equivalent release destination and credentials.

Publishing queries crates.io and skips versions that already exist. `make push-both` pushes the current branch to both configured remotes (`origin` and `upstream`). Both operations are explicit so a release cannot be created accidentally by a scan.

## Change policy

Performance-sensitive inference code must not be changed merely to satisfy a CI MSRV check. Update the declared toolchain and maintenance documentation only when that is the intended compatibility decision, and record the validation evidence in the relevant project notes. When adding crates, docs, configuration or tools, rerun the scanner and commit its accompanying script or documentation changes together with the structural change.
