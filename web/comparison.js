// The browser schedules experiments and draws their results. All task choices,
// actions, table updates, probes and evaluation trajectories come from Rust.
import { Studio } from './studio.js';
const $ = id => document.getElementById(id);
const TASKS = ['A · Short carry', 'B · Medium carry', 'C · Long carry'];
const ACTIONS = ['Left', 'Right', 'Lift', 'Lower', 'Close', 'Open'];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const format = n => n.toLocaleString('en-US');

export function mountComparison(w) {
  let timer = 0, playing = false, training = false, remaining = 0, schedule = 0;
  let points = [], lastRound = -1;
  const studios = [0, 1].map(arm => {
    try { return new Studio($(`compare-world-${arm}`), { detail: false, shadowSize: 1024, noteId: 'compare-render-note' }); }
    catch (error) {
      $(`compare-motion-${arm}`).textContent = '3D unavailable; task checks, learning and values still work.';
      $('compare-render-note').textContent = `3D unavailable: ${error.message}. The numeric experiment still runs.`;
      return null;
    }
  });
  for (let arm = 0; arm < 2; arm++) {
    $(`compare-results-${arm}`).innerHTML = TASKS.map((name, task) => `<div class="task-result" id="task-result-${arm}-${task}"><span>${name}<small id="task-attempts-${arm}-${task}">0 training attempts</small></span><b id="task-check-${arm}-${task}">—</b></div>`).join('');
  }
  $('compare-probe').innerHTML = ACTIONS.map((name, a) => `<div class="probe-row"><b>${name}</b><span><i id="probe-bar-0-${a}"></i><span id="probe-0-${a}">0.000</span></span><span><i id="probe-bar-1-${a}"></i><span id="probe-1-${a}">0.000</span></span></div>`).join('');

  function controls() {
    for (const id of ['compare-run', 'compare-batch', 'compare-easy', 'compare-hard', 'compare-mode', 'compare-seed', 'compare-reset', 'compare-watch', 'compare-step', 'compare-task']) $(id).disabled = training && id !== 'compare-run';
    $('compare-run').textContent = training ? 'Ⅱ Pause comparison' : '▶ Compare 900 attempts';
    $('compare-watch').textContent = playing ? 'Ⅱ Pause playback' : '▷ Watch both policies';
  }
  function stop() { clearTimeout(timer); training = false; playing = false; controls(); }
  function drawChart() {
    const max = Math.max(90, points.at(-1)?.n || 0);
    let svg = '';
    for (let i = 0; i <= 3; i++) {
      const y = 160 - i * 43;
      svg += `<line x1="40" x2="975" y1="${y}" y2="${y}" stroke="#dfe6d8"/><text x="10" y="${y + 4}">${i}</text>`;
    }
    for (const arm of [1, 0]) {
      let path = '';
      points.forEach((p, i) => {
        const x = 40 + 935 * p.n / max, y = 160 - 43 * p.solved[arm];
        path += i ? `H${x}V${y}` : `M${x},${y}`;
      });
      svg += `<path d="${path}" stroke="${arm ? '#8f78a6' : '#187f70'}" stroke-width="${arm ? 4 : 2}" ${arm ? 'stroke-dasharray="6 4"' : ''} fill="none"/>`;
    }
    svg += `<text x="40" y="192">0</text><text x="975" y="192" text-anchor="end">${format(max)} attempts per side</text><text x="40" y="15">Full-start tasks solved (out of 3)</text>`;
    $('compare-chart').innerHTML = svg;
  }
  function render() {
    const n = w.compare_read(0, 9), solved = [0, 0];
    $('compare-rounds').textContent = format(n);
    for (let arm = 0; arm < 2; arm++) {
      $(`compare-updates-${arm}`).textContent = format(w.compare_read(arm, 8));
      for (let task = 0; task < 3; task++) {
        const steps = w.compare_evaluate(arm, task);
        solved[arm] += steps > 0;
        $(`task-result-${arm}-${task}`).classList.toggle('solved', steps > 0);
        $(`task-check-${arm}-${task}`).textContent = steps ? `✓ ${steps} actions` : '— Not yet';
        $(`task-attempts-${arm}-${task}`).textContent = `${format(w.compare_read(arm, 10 + task))} training attempts`;
      }
      for (let a = 0; a < 6; a++) {
        const v = w.compare_probe(arm, a);
        $(`probe-${arm}-${a}`).textContent = v.toFixed(3);
        $(`probe-bar-${arm}-${a}`).style.width = `${v * 36}px`;
      }
    }
    const counts = [0, 1, 2].map(t => w.compare_probe(0, t + 6));
    $('compare-provenance').textContent = `Updates to this shared row: ${format(counts[0])} from A · ${format(counts[1])} from B · ${format(counts[2])} from C. The hard specialist’s row received ${format(w.compare_probe(1, 8))} updates, all from C.`;
    if (n !== lastRound) {
      points.push({ n, solved }); lastRound = n;
      if (points.length > 1200) points = points.filter((_, i) => i % 2 === 0 || i === points.length - 1);
      drawChart();
    }
  }
  function drawPlayback(animate = false) {
    const task = Number($('compare-task').value);
    for (let arm = 0; arm < 2; arm++) {
      const s = { x: w.compare_read(arm, 0), y: w.compare_read(arm, 1), peg: w.compare_read(arm, 2), depth: w.compare_read(arm, 3), held: !!w.compare_read(arm, 4) };
      studios[arm]?.setState(s, animate && !reduced, 210);
      const steps = w.compare_read(arm, 5), action = w.compare_read(arm, 7), done = w.compare_read(arm, 6);
      const status = s.depth === 2 ? 'Assembled · reward +1' : done ? 'Time limit · not assembled' : `${TASKS[task]} · action ${steps}/40`;
      $(`compare-overlay-${arm}`).textContent = status;
      $(`compare-overlay-${arm}`).classList.toggle('solved', s.depth === 2);
      if (studios[arm]) $(`compare-motion-${arm}`).textContent = action < 0 ? 'Full start. No exploration or learning during this test.' : `${ACTIONS[action]} · ${s.depth === 2 ? 'complete' : s.depth ? 'partly inserted' : s.held ? 'peg held' : 'peg on table'} · ${steps} actions`;
    }
  }
  function resetPlayback() { w.compare_watch(Number($('compare-task').value)); drawPlayback(); }
  function replayTick() {
    if (!playing) return;
    w.compare_step(); drawPlayback(true);
    if (w.compare_read(0, 6) && w.compare_read(1, 6)) { playing = false; controls(); return; }
    timer = setTimeout(replayTick, reduced ? 60 : 280);
  }
  function watch() {
    if (playing) { stop(); return; }
    stop(); resetPlayback(); playing = true; controls(); timer = setTimeout(replayTick, 300);
  }
  function fresh() {
    stop();
    let seed = Math.floor(Number($('compare-seed').value));
    if (!Number.isFinite(seed) || seed < 1 || seed > 4294967295) seed = 7;
    $('compare-seed').value = seed;
    const mode = Number($('compare-mode').value);
    w.compare_reset(seed, mode); points = []; lastRound = -1;
    $('compare-start-note').textContent = mode === 2 ? 'Same tasks; each side adapts its own reset probabilities.' : 'Same task, same start, paired random seeds.';
    $('compare-status').textContent = 'Both memories start empty.';
    render(); resetPlayback(); controls();
  }
  function trainingTick() {
    if (!training) return;
    const n = Math.min(15, remaining);
    w.compare_train(n, schedule); remaining -= n; render();
    const purpose = schedule === 1 ? 'Easy-only practice' : schedule === 2 ? 'Hard-only practice' : 'A, B, C in rotation';
    $('compare-status').textContent = `${purpose} · ${format(w.compare_read(0, 9))} attempts per side${remaining ? ` · ${remaining} left` : ' · complete'}`;
    if (remaining) { timer = setTimeout(trainingTick, 65); return; }
    training = false; controls();
    // The causal probe is the result of easy-only practice. Don't claim it has
    // solved the whole hard task. Other batches finish with an actual replay.
    if (schedule !== 1) watch();
  }
  function train(rounds, which = 0) {
    stop(); remaining = rounds; schedule = which; training = true; controls();
    resetPlayback(); timer = setTimeout(trainingTick, 20);
  }
  $('compare-run').addEventListener('click', () => {
    if (training) { stop(); $('compare-status').textContent = 'Paused. Learned values are kept.'; return; }
    train(900);
  });
  $('compare-batch').addEventListener('click', () => train(90));
  $('compare-easy').addEventListener('click', () => { fresh(); train(300, 1); });
  $('compare-hard').addEventListener('click', () => train(90, 2));
  $('compare-reset').addEventListener('click', fresh);
  $('compare-mode').addEventListener('change', fresh);
  $('compare-seed').addEventListener('change', fresh);
  $('compare-task').addEventListener('change', () => { stop(); resetPlayback(); });
  $('compare-watch').addEventListener('click', watch);
  $('compare-step').addEventListener('click', () => {
    stop(); if (w.compare_read(0, 6) && w.compare_read(1, 6)) resetPlayback();
    w.compare_step(); drawPlayback(true);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  fresh();
  fetch('./transfer-source.rs').then(r => { if (!r.ok) throw new Error('Source unavailable'); return r.text(); })
    .then(text => { $('transfer-source').textContent = text; })
    .catch(() => { $('transfer-source').textContent = 'Read src/transfer.rs in the GitHub repository.'; });
}
