#!/usr/bin/env sh
# Compiler only; no Cargo packages, npm packages, or downloaded runtime assets.
set -eu
cd "$(dirname "$0")"
RUSTC="${RUSTC:-rustc}"
mkdir -p dist
cp web/index.html web/style.css web/app.js web/studio.js dist/
cp src/lib.rs dist/source.rs
cp README.md dist/README.md
touch dist/.nojekyll
"$RUSTC" --edition=2021 --crate-type=cdylib --target=wasm32-unknown-unknown \
  -C opt-level=2 -C panic=abort -C strip=symbols src/lib.rs -o dist/assembly.wasm
echo "Built dist/ — serve this directory over HTTP."
