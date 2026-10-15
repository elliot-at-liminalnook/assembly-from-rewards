#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
"${RUSTC:-rustc}" --edition=2021 --test src/lib.rs -o build/core-tests
./build/core-tests
node --check web/app.js
node --check web/studio.js
node --check web/comparison.js
node tests/wasm.mjs
