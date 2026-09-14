# Local developer entry points. The default target is the measured native build.
.DEFAULT_GOAL := native

.PHONY: native native-burn web check check-version fmt lint test scan-workspace bump-version publish-crates push-both help

native:
	 pnpm build:native

native-burn:
	 pnpm build:native -- --burn-only

web:
	 pnpm build

check:
	 cargo check --workspace --all-targets --locked

check-version:
	 python3 tools/scan-workspace.py --check-version

fmt:
	 cargo fmt --all -- --check

lint:
	 cargo clippy --workspace --all-targets --locked --all-features -- -D warnings

test:
	 cargo test --workspace --all-features --locked

scan-workspace:
	 python3 tools/scan-workspace.py

bump-version:
	 @test -n "$(VERSION)" || (printf '%s\n' 'usage: make bump-version VERSION=x.y.z' >&2; exit 2)
	 tools/bump-version.sh "$(VERSION)"

publish-crates:
	 tools/publish-crates.sh

push-both:
	 tools/push-both.sh

help:
	 @printf '%s\n' \
	  'make                 Build the CPU-optimized native CLI and GUI' \
	  'make native-burn      Build the CPU-optimized Burn-only variant' \
	  'make web              Build the frontend only' \
	  'make check            Run workspace checks' \
	  'make check-version    Verify CLI and GUI versions are synchronized' \
	  'make fmt              Verify Rust formatting' \
	  'make lint             Run strict Clippy' \
	  'make test             Run the full workspace test suite' \
	  'make scan-workspace    Scan crates, docs, tools and configuration files' \
	  'make bump-version     Update the workspace crate version and lockfile' \
	  'make publish-crates   Publish library crates to crates.io in dependency order' \
	  'make push-both        Push the current branch to origin and upstream'
