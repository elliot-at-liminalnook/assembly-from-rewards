#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
"${RUSTC:-rustc}" --edition=2021 --test src/lib.rs -o build/core-tests
./build/core-tests
node tests/wasm.mjs
