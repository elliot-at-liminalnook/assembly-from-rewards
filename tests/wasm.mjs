// Node's built-in WebAssembly runtime, filesystem, and assertions. No packages.
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
const bytes = await readFile(new URL('../dist/assembly.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(bytes, {});
const w = instance.exports;
assert.equal(WebAssembly.Module.imports(new WebAssembly.Module(bytes)).length, 0);
w.reset(7, 0);
assert.equal(w.evaluate(), 0, 'An empty table should not already solve the task.');
w.begin(4, 1);
for (let i = 0; i < 40 && !w.read(21); i++) {
  w.step();
  assert.equal(w.read(26), w.read(3) === 2 ? 1 : 0);
}
const results = [];
for (const mode of [0, 1, 2]) {
  for (const seed of [1, 7, 42]) {
    w.reset(seed, mode);
    w.train(3000);
    const steps = w.evaluate();
    results.push({ mode, seed, fullStartSteps: steps });
    if (mode < 2) assert.ok(steps > 0 && steps <= 40);
    const episodes = w.read(23), updates = w.read(33);
    w.begin(0, 0);
    while (!w.read(21)) w.step();
    assert.equal(w.read(23), episodes);
    assert.equal(w.read(33), updates);
    const probabilities = Array.from({ length: 5 }, (_, i) => w.reset_stat(i, 1));
    assert.ok(Math.abs(probabilities.reduce((a, b) => a + b) - 1) < 1e-10);
  }
}
// Same seed + same interaction schedule must produce the same learner.
const fingerprint = () => [w.read(24), w.read(33), w.evaluate(), ...Array.from({ length: 5 }, (_, i) => w.reset_stat(i, 0))];
w.reset(42, 0); w.train(500); const first = fingerprint();
w.reset(42, 0); w.train(500); assert.deepEqual(fingerprint(), first);
console.table(results);
console.log(`WebAssembly checks passed (${bytes.length.toLocaleString()} bytes, zero imports).`);
