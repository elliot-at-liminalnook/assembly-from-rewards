#!/usr/bin/env sh
# Compiler only; no Cargo packages, npm packages, or downloaded runtime assets.
set -eu
cd "$(dirname "$0")"
RUSTC="${RUSTC:-rustc}"
mkdir -p dist
cp web/index.html web/style.css web/app.js web/studio.js web/comparison.js dist/
cp src/lib.rs dist/source.rs
cp src/transfer.rs dist/transfer-source.rs
cp README.md dist/README.md
touch dist/.nojekyll
"$RUSTC" --edition=2021 --crate-type=cdylib --target=wasm32-unknown-unknown \
  -C opt-level=2 -C panic=abort -C strip=symbols src/lib.rs -o dist/assembly.wasm

# A new HTML page must not pick up an old cached script. Content-derived names
# make every page version refer to the exact renderer, source, and WASM it needs.
# cksum, awk, and sed are standard POSIX tools; no bundler/package is required.
studio_hash=$(cksum web/studio.js | awk '{print $1}')
style_hash=$(cksum web/style.css | awk '{print $1}')
wasm_hash=$(cksum dist/assembly.wasm | awk '{print $1}')
transfer_hash=$(cksum src/transfer.rs | awk '{print $1}')
cp src/transfer.rs "dist/transfer-source-$transfer_hash.rs"
source_hash=$(cksum src/lib.rs | awk '{print $1}')
cp web/studio.js "dist/studio-$studio_hash.js"
cp web/style.css "dist/style-$style_hash.css"
cp dist/assembly.wasm "dist/assembly-$wasm_hash.wasm"
cp src/lib.rs "dist/source-$source_hash.rs"
sed -e "s|./studio.js|./studio-$studio_hash.js|g" \
    -e "s|./transfer-source.rs|./transfer-source-$transfer_hash.rs|g" web/comparison.js > dist/comparison.js
comparison_hash=$(cksum dist/comparison.js | awk '{print $1}')
cp dist/comparison.js "dist/comparison-$comparison_hash.js"
sed -e "s|./studio.js|./studio-$studio_hash.js|g" \
    -e "s|./comparison.js|./comparison-$comparison_hash.js|g" \
    -e "s|./assembly.wasm|./assembly-$wasm_hash.wasm|g" \
    -e "s|./source.rs|./source-$source_hash.rs|g" web/app.js > dist/app.js
app_hash=$(cksum dist/app.js | awk '{print $1}')
cp dist/app.js "dist/app-$app_hash.js"
sed -e "s|./style.css|./style-$style_hash.css|g" \
    -e "s|./app.js|./app-$app_hash.js|g" \
    -e "s|./studio.js|./studio-$studio_hash.js|g" web/index.html > dist/index.html
echo "Built dist/ — serve this directory over HTTP."
