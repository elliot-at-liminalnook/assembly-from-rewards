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

// Verify the comparison through the exact browser-facing WebAssembly interface.
const comparisonFingerprint = () => [0,1].flatMap(a => [8,9,10,11,12,20,21,22].map(f => w.compare_read(a,f)).concat(Array.from({length:9},(_,f)=>w.compare_probe(a,f))));
w.compare_reset(7,0); w.compare_train(300,1);
assert.equal(w.compare_read(0,12),0, 'No hard-task training in the easy-only intervention');
assert.ok(w.compare_probe(0,1)>.8, 'Easy practice gives the shared hard-task controller useful finishing values');
assert.equal(w.compare_probe(1,1),0, 'Hard specialist cannot receive easy-task updates');
assert.equal(w.compare_evaluate(0,2),0, 'A useful suffix does not teach the untouched hard initial state');
const probeBefore = comparisonFingerprint();
w.compare_watch(2); for(let i=0;i<40;i++) w.compare_step();
assert.deepEqual(comparisonFingerprint(),probeBefore, 'Watching must not affect either learner or RNG schedule');
w.compare_reset(7,0);w.compare_train(900,0);const comparisonFirst=comparisonFingerprint();
assert.equal(w.compare_evaluate(0,2),9);
assert.equal(w.compare_evaluate(1,2),0);
for(const arm of [0,1]) for(const field of [10,11,12]) assert.equal(w.compare_read(arm,field),300);
w.compare_reset(7,0);for(let i=0;i<60;i++)w.compare_train(15,0);
assert.deepEqual(comparisonFingerprint(),comparisonFirst, 'Batch size must not change the experiment');
for(const mode of [1,2]) {
 w.compare_reset(42,mode);w.compare_train(900,0);
 assert.ok(w.compare_evaluate(0,2)>0, 'Shared controller also learns with mixed starts');
}
const seedChecks=[];
for(let seed=1;seed<=24;seed++){
 w.compare_reset(seed,0);w.compare_train(900,0);
 seedChecks.push({seed,shared:w.compare_evaluate(0,2)>0,separate:w.compare_evaluate(1,2)>0});
}
console.log('Comparison at 900 matched attempts across seeds 1–24:', {
 sharedHardSolved:seedChecks.filter(r=>r.shared).length,
 separateHardSolved:seedChecks.filter(r=>r.separate).length,
 totalSeeds:24
});
console.log('Shared/separate WebAssembly, provenance, determinism and read-only replay checks passed.');
