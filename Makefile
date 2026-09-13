# Local developer entry points. The default target is the measured native build.
.DEFAULT_GOAL := native

.PHONY: native native-burn web check fmt lint test help

native:
	 pnpm build:native

native-burn:
	 pnpm build:native -- --burn-only

web:
	 pnpm build

check:
	 cargo check --workspace --all-targets --locked

fmt:
	 cargo fmt --all -- --check

lint:
	 cargo clippy --workspace --all-targets --locked --all-features -- -D warnings

test:
	 cargo test --workspace --all-features --locked

help:
	 @printf '%s\n' \
	  'make                 Build the CPU-optimized native CLI and GUI' \
	  'make native-burn      Build the CPU-optimized Burn-only variant' \
	  'make web              Build the frontend only' \
	  'make check            Run workspace checks' \
	  'make fmt              Verify Rust formatting' \
	  'make lint             Run strict Clippy' \
	  'make test             Run the full workspace test suite'
