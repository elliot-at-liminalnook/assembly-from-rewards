// Presentation only. All world transitions, choices, learning, and reset sampling
// happen in Rust/WASM. The 3D rig interpolates between actual engine states.
import { Studio } from './studio.js';
import { mountComparison } from './comparison.js';
const $ = (id) => document.getElementById(id);
const ACTIONS = ['Left', 'Right', 'Lift', 'Lower', 'Close', 'Open'];
const STARTS = ['On the table', 'Ready to grasp', 'Already held', 'At the socket', 'Partly inserted'];
const COLORS = ['#187f70', '#627dbe', '#b37c40', '#9e6f9e', '#91a34e'];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let w, running = false, watching = false, busy = false, timer = 0;
let series = [], lastEpisode = -1;
const number = (n) => n.toLocaleString('en-US');
const fixed = (n) => n.toFixed(3);
let studio;
try {
  studio = new Studio($('world'));
} catch (error) {
  $('renderer-error').hidden = false;
  $('renderer-error').textContent = '3D graphics are unavailable in this browser. You can still train and inspect the learner below.';
  $('detail-frame').hidden = true;
  console.error('Studio renderer:', error);
}
for (const button of document.querySelectorAll('[data-view]')) {
  button.addEventListener('click', () => {
    studio?.camera(button.dataset.view);
    for (const other of document.querySelectorAll('[data-view]')) other.setAttribute('aria-pressed', String(other === button));
  });
}
$('detail-toggle').addEventListener('click', () => {
  if (!studio) return;
  studio.detail = !studio.detail;
  $('detail-frame').hidden = !studio.detail;
  $('detail-toggle').setAttribute('aria-pressed', String(studio.detail));
  $('detail-toggle').querySelector('span').textContent = studio.detail ? 'ON' : 'OFF';
  studio.requestDraw();
});

$('q-bars').innerHTML = ACTIONS.map((name, i) => `<div class="q-row" id="q-row-${i}"><span>${name}</span><div class="bar-track"><div class="bar-fill" id="q-fill-${i}"></div></div><span id="q-number-${i}">0.00</span></div>`).join('');
$('reset-rows').innerHTML = STARTS.map((name, i) => `<div class="reset-row" id="reset-${i}"><span class="reset-name"><span class="reset-number">${i + 1}</span>${name}</span><span class="reset-rate" id="rate-${i}">Untried</span><span class="probability"><span class="bar-track"><span class="bar-fill" id="prob-fill-${i}" style="display:block"></span></span><b id="prob-${i}">20%</b></span></div>`).join('');
$('chart-legend').innerHTML = STARTS.map((name, i) => `<span><i style="background:${COLORS[i]}"></i>${name}</span>`).join('');

function state(offset = 0) {
  return { x: w.read(offset), y: w.read(offset + 1), peg: w.read(offset + 2), depth: w.read(offset + 3), held: !!w.read(offset + 4) };
}
function snapshot() {
  return { state: state(), previous: state(10), steps: w.read(20), done: !!w.read(21),
    training: !!w.read(22), episodes: w.read(23), wins: w.read(24), action: w.read(25),
    reward: w.read(26), old: w.read(27), future: w.read(28), target: w.read(29),
    updated: w.read(30), start: w.read(31), exploratory: !!w.read(32), updates: w.read(33) };
}

// The renderer receives a state, never a target action sequence. Camera changes
// and inverse kinematics cannot write to the learner or change its outcomes.
function drawWorld(s, animate) {
  $('studio-success').hidden = s.state.depth !== 2;
  studio?.setState(s.state, animate && !reduceMotion, 430);
}

function record() {
  const n = w.read(23);
  if (n === lastEpisode) return;
  lastEpisode = n;
  series.push({ n, rates: STARTS.map((_, i) => w.reset_stat(i, 0)) });
  // Keep the complete horizontal range while thinning old points for long runs.
  if (series.length > 500) series = series.filter((_, i) => i % 2 === 0 || i === series.length - 1);
  drawChart();
}
function drawChart() {
  const max = Math.max(100, series.at(-1)?.n || 0);
  let html = '';
  for (const p of [0, 0.5, 1]) {
    const y = 145 - 120 * p;
    html += `<line x1="38" x2="975" y1="${y}" y2="${y}" stroke="#dfe6d8"/><text x="2" y="${y + 4}">${p * 100}%</text>`;
  }
  for (let i = 0; i < 5; i++) {
    const d = series.map((v, j) => `${j ? 'L' : 'M'}${38 + v.n / max * 937},${145 - v.rates[i] * 120}`).join(' ');
    html += `<path d="${d}" fill="none" stroke="${COLORS[i]}" stroke-width="2" opacity=".85"/>`;
  }
  html += `<text x="38" y="176">0</text><text x="975" y="176" text-anchor="end">${number(max)} episodes</text>`;
  if (series.length < 2) html += '<text x="500" y="80" text-anchor="middle">Training history will appear here.</text>';
  $('chart').innerHTML = html;
}

function render(animate = false) {
  const s = snapshot();
  drawWorld(s, animate);
  $('episodes').textContent = number(s.episodes);
  $('updates').textContent = number(s.updates);
  const steps = w.evaluate();
  $('eval-result').textContent = steps ? `Solved · ${steps} actions` : 'Not solved yet';
  $('step-count').textContent = `Action ${s.steps} / 40`;
  $('episode-label').textContent = `${s.training ? 'Training' : 'Policy test'} · ${STARTS[s.start]}`;
  $('state-label').textContent = s.state.depth === 2 ? 'Fully assembled. This is the only rewarded state.' : s.state.depth === 1 ? 'Partly inserted · one more lower action completes it' : s.state.held ? `Peg held · ${s.state.y ? 'raised' : 'at table height'}` : `Peg on the table at ${s.state.peg} · hand ${s.state.y ? 'raised' : 'lowered'}`;
  $('reward').textContent = s.reward ? '+1' : '0';
  if (s.action < 0) {
    $('action-title').textContent = s.training ? 'A new starting situation.' : 'Testing what it has learned.';
    $('action-story').textContent = s.training ? `Starting ${STARTS[s.start].toLowerCase()}. The robot chooses its own next move; the reset supplies no action instructions.` : 'The peg starts on the table. Every move follows the current best value; no learning or random exploration occurs.';
    $('update-equation').textContent = 'Waiting for an action…';
    $('target-equation').textContent = 'target = reward + 0.96 × best next value';
  } else {
    $('action-title').textContent = `${ACTIONS[s.action]}${s.reward ? ' → success!' : s.done ? ' → time limit' : '.'}`;
    if (!s.training) {
      $('action-story').textContent = s.reward ? 'The learned controller completed the task from the table. This test changed no learned values.' : s.done ? 'This policy did not finish within 40 actions. Train more, then test again.' : 'Following the learned table. This action is only being evaluated, so the table is unchanged.';
      $('update-equation').textContent = 'Evaluation only · no update';
      $('target-equation').textContent = `Chosen action value: ${fixed(s.old)}`;
    } else {
      const choice = s.exploratory ? 'Random exploration.' : 'Best-valued action (random tie-breaks).';
      const effect = s.reward ? 'Completion gives the reward that starts useful learning.' : s.updated > s.old + 0.000001 ? 'No reward now, but the next state has value. This earlier action improves.' : s.previous.x === s.state.x && s.previous.y === s.state.y && s.previous.held === s.state.held && s.previous.depth === s.state.depth ? 'This action left the world unchanged.' : 'No completion reward. The update uses the best value in the next state.';
      $('action-story').textContent = `${choice} ${effect}`;
      $('update-equation').textContent = `${fixed(s.old)} + 0.25 × (${fixed(s.target)} − ${fixed(s.old)}) = ${fixed(s.updated)}`;
      $('target-equation').textContent = `target = ${s.reward} + 0.96 × ${fixed(s.future)} = ${fixed(s.target)}`;
    }
  }
  const values = ACTIONS.map((_, i) => w.action_value(i));
  const max = Math.max(...values);
  for (let i = 0; i < 6; i++) {
    $(`q-fill-${i}`).style.width = `${values[i] * 100}%`;
    $(`q-number-${i}`).textContent = values[i].toFixed(2);
    $(`q-row-${i}`).classList.toggle('best', max > 0 && values[i] === max);
  }
  for (let i = 0; i < 5; i++) {
    const count = w.reset_stat(i, 3), p = w.reset_stat(i, 0), probability = w.reset_stat(i, 1);
    $(`rate-${i}`).textContent = count ? `${Math.round(p * 100)}%` : 'Untried';
    $(`rate-${i}`).title = `${count} completed attempts in the recent window; ${w.reset_stat(i, 2)} total starts`;
    $(`prob-${i}`).textContent = `${Math.round(probability * 100)}%`;
    $(`prob-fill-${i}`).style.width = `${probability * 100}%`;
    $(`reset-${i}`).classList.toggle('active', i === s.start);
  }
  record();
}

function stop() {
  running = false; watching = false; clearTimeout(timer);
  $('run').textContent = '▶ Run training';
  $('watch').textContent = '▷ Watch learned policy';
}
function advance() {
  if (w.read(21)) w.begin(-1, 1);
  w.step();
  render(true);
}
function loop() {
  if (!running) return;
  if (watching && w.read(21)) { stop(); return; }
  advance();
  timer = setTimeout(loop, w.read(21) ? 1200 : 550);
}
function fresh() {
  stop();
  let seed = Math.floor(Number($('seed').value));
  if (!Number.isFinite(seed) || seed < 1 || seed > 4294967295) seed = 7;
  $('seed').value = String(seed);
  w.reset(seed, Number($('mode').value));
  const mode = Number($('mode').value);
  $('sampling-description').textContent = [
    'A start with about 50% recent success gets more practice. Even a 0% or 100% start keeps a chance.',
    'All five starts get the same chance, regardless of success. This keeps the reset bank but removes adaptive sampling.',
    'Every attempt begins with the peg on the table. The learner must find a complete success before any action value can rise.'
  ][mode];
  $('sampling-rule').textContent = [
    'weight = 0.05 + √(p × (1 − p))',
    'probability = 1 / 5 for every start',
    'probability = 1 for the table start'
  ][mode];
  series = []; lastEpisode = -1;
  w.begin(-1, 1);
  render();
}
function disableControls(disabled) {
  for (const id of ['run', 'single', 'fast', 'watch', 'restart', 'seed', 'mode']) $(id).disabled = disabled;
}
$('run').addEventListener('click', () => {
  if (running) { stop(); return; }
  if (!w.read(22)) w.begin(-1, 1);
  running = true; watching = false;
  $('run').textContent = 'Ⅱ Pause training'; loop();
});
$('single').addEventListener('click', () => {
  stop(); if (!w.read(22)) w.begin(-1, 1); advance();
});
$('watch').addEventListener('click', () => {
  if (watching) { stop(); return; }
  stop(); w.begin(0, 0); render(); running = true; watching = true;
  $('watch').textContent = 'Ⅱ Stop watching';
  $('run').textContent = 'Ⅱ Pause';
  timer = setTimeout(loop, 400);
});
$('fast').addEventListener('click', async () => {
  if (busy) return;
  stop(); busy = true; disableControls(true);
  // Small chunks keep the UI responsive and sample the actual learning curve.
  for (let n = 0; n < 1000; n += 25) {
    w.train(25); render(); $('fast').textContent = `Training ${n + 25} / 1,000…`;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  $('fast').textContent = 'Train 1,000 episodes';
  busy = false; disableControls(false);
});
$('restart').addEventListener('click', fresh);
$('mode').addEventListener('change', fresh);
$('seed').addEventListener('change', fresh);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });

async function load() {
  try {
    const response = await fetch('./assembly.wasm');
    if (!response.ok) throw new Error(`WASM request returned ${response.status}`);
    // ArrayBuffer works even on simple servers without an application/wasm MIME.
    const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
    w = instance.exports;
    $('runtime').textContent = '● WebAssembly running · zero packages';
    disableControls(false); fresh(); mountComparison(w);
  } catch (error) {
    $('runtime').textContent = 'WebAssembly unavailable';
    $('error').hidden = false;
    $('error').textContent = `The interactive module could not load: ${error.message}. If running locally, build first and serve dist/ over HTTP; opening the HTML as a local file will not work. The written lesson below is still available.`;
  }
  try {
    const source = await fetch('./source.rs');
    if (!source.ok) throw new Error('Source unavailable');
    $('actual-source').textContent = await source.text();
  } catch { $('actual-source').textContent = 'Could not load the source here. Read src/lib.rs in the linked GitHub repository.'; }
}
drawChart();
load();
