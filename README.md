# Assembly from rewards

A runnable, dependency-free teaching example inspired by **AlphaNIST**. A tiny
Rust learner compiles directly to WebAssembly. Plain JavaScript draws its actual
state transitions and displays its actual learned values. No recorded solution,
pretrained weights, fake graph, external fonts, CDN, package manager, or backend.

**This is a conceptual model, not an AlphaNIST reproduction.** It replaces PPO and
neural networks with tabular Q-learning, contact physics with discrete rules, and
20 tasks with one peg and one socket. Its results say nothing about real robots.

## Run it

Visit the deployed page, then choose **One action** to inspect a learning step or
**Train 1,000 episodes** to speed through practice. **Watch learned policy** starts
with the peg on the table and runs the current policy without changing it.

To build locally, install Rust with its `wasm32-unknown-unknown` target. The pinned
version is in `rust-toolchain.toml`. There are **zero third-party library dependencies**;
the Rust compiler and target are build tools. Node is used only for tests. Python
is an optional local web server. A normal browser is the runtime.

```sh
rustup target add wasm32-unknown-unknown
sh build.sh
sh test.sh                          # requires Node, uses only built-in modules
python3 -m http.server 8000 --directory dist
# Open http://localhost:8000 (double-clicking index.html will not load WASM).
```

## The world: five positions, two heights, one peg

The hand moves along a five-position rail. The peg starts at position 0, the
socket is at 4, and the hand starts raised at 2. There are six actions: left,
right, lift, lower, close, open. A held peg must be lifted before sideways motion.
At the socket mouth, each lower action inserts it one unit; two units finishes.
Lifting can undo insertion. Invalid actions do nothing. Opening the hand drops
an uninserted peg onto the table at its current horizontal position.

The state is `(hand_x, hand_height, peg_x, insertion_depth, held)`. This is enough
to predict the next state in our deliberately simplified world. There are 300
possible encodings (many unreachable), each with six learned action values.

Reward is **1 only when fully inserted**, otherwise **0**. No progress reward,
grasp bonus, or motion penalty is used. AlphaNIST also uses control penalties;
we omit them to make credit assignment easier to see. Episodes stop after success
or 40 actions. The timeout is a collection cutoff; Q-learning still bootstraps on
that last transition, as time remaining is not part of the state.

## Where the starting states come from

`reset_state()` constructs five snapshots directly:

| Reset | Hand and peg arrangement | What remains |
|---|---|---|
| On the table | Peg at 0; hand raised at 2 | Reach, grasp, transport, insert |
| Ready to grasp | Hand at the peg on the table | Close, lift, transport, insert |
| Already held | Peg held, raised at 0 | Transport, insert |
| At the socket | Peg held at the socket mouth | Insert twice |
| Partly inserted | Peg held, inserted one unit | Insert once |

The last snapshot is made by defining the known completed geometry and subtracting
one insertion unit. No successful action sequence is needed. The cited OmniReset
method uses richer geometry: it spawns an object at the goal, perturbs it with small
random forces, samples grasps, and validates states through collision checks and
simulation. Our subtraction is an educational stand-in for that process.

These snapshots are not demonstrations, but they **are supplied task structure**.
The learner knows no sequence connecting them and starts with all action values at
zero. It must discover its own actions between situations.

## The algorithm in pseudocode

```text
Q[state, action] = 0
make a bank of starting snapshots

repeat:
    for each snapshot i:
        p[i] = success fraction in its last 32 completed attempts
        weight[i] = 0.05 + sqrt(p[i] * (1 - p[i]))
    choose a starting snapshot proportional to its weight

    until success or 40 actions:
        with 25% probability: choose a random action
        otherwise: choose an action with highest Q[state, action]
        next_state = apply the action
        reward = 1 if fully inserted, otherwise 0
        future = 0 if successful, otherwise max(Q[next_state, :])
        target = reward + 0.96 * future
        Q[state, action] += 0.25 * (target - Q[state, action])
        state = next_state

    record this attempt's success/failure for the starting snapshot
```

The table does not store the optimal answer ahead of time. Consider the first
successful `lower` from partly inserted. Its value goes from 0 to
`0 + 0.25 * (1 - 0) = 0.25`. Later, lowering from the socket mouth can get **zero
immediate reward** but reach that useful state. Its target becomes
`0 + 0.96 * 0.25 = 0.24`; its value can rise to `0.06`. Repetition carries value
back toward the pickup. Numbers here illustrate a particular update, not a promise
about the order in which a random run learns.

`0.25` (alpha) is the learning rate. `0.96` (gamma) discounts delayed reward, so a
shorter successful route is worth more. Q is discounted future return, not exactly
a probability. AlphaNIST uses PPO's actor/critic updates instead of this equation.

## Why choose intermediate success rates?

`sqrt(p * (1-p))` is largest at `p=0.5`. Those starts are neither already mastered
nor almost always unsuccessful. The `0.05` floor prevents any start from becoming
impossible to sample. Before a start has outcomes, we use `p=0`; all initially have
equal weight. Each has a rolling 32-outcome buffer. The UI shows both success
fraction and next-reset probability; these are different quantities.

The page also provides uniform mixed resets and table-only resets for experiments.
Changing mode or seed resets learning, history, and charts. The plot measures each
reset's recent **training** success, which includes exploratory actions. The full
start check is one deterministic greedy rollout. Table-only practice may take much
longer to find its first success. The demo does not assert a universal advantage
or reproduce the paper's ablation results. Compare several seeds, episode budgets,
AND action counts.

## Read the code

| File | Purpose |
|---|---|
| `src/lib.rs` | State, rules, resets, sampling, Q-learning, numeric browser interface, native tests |
| `web/app.js` | Calls WASM; forwards states to the renderer; shows metrics and the exact Q update |
| `web/studio.js` | Procedural 3D meshes, lighting, shadows, visual inverse kinematics, and cameras; no dependencies |
| `web/index.html` | The lesson, pseudocode, controls, source viewer |
| `web/style.css` | Responsive page styling; no framework |
| `tests/wasm.mjs` | Exercises the real compiled WASM using Node's built-in runtime |
| `build.sh` | Compiles Rust and copies static files into `dist/` |
| `.github/workflows/pages.yml` | On a main-branch push: build, test, publish |

Rust is used for the algorithm and simulator. JavaScript handles presentation and
calls a small exported interface: `reset`, `begin`, `step`, `train`, `evaluate`,
`read`, `reset_stat`, `action_value`. The module imports **nothing** from JavaScript.
An animation interpolates between two real discrete states; the interpolation has
no effect on the world or on learning. Watching the learned policy does not update
the Q table, the outcome histories, or the training random-number stream.

## The 3D studio: better pictures, the same small learner

The visual reference is the [AlphaNIST project site](https://alpha-nist.github.io/):
white robot shells, dark joints, metallic fixtures, gold parts, and a synchronized
contact close-up. Our meshes are original, generated in code; no assets or videos
from that site are copied. The renderer uses the browser's built-in **WebGL** API.

Keep this boundary in mind: the **discrete Rust world is the simulation**. The
articulated arm is an illustrative rendering rig. It is not a simulated Franka,
does not enforce real joint limits or contact dynamics, and is not the controller
being learned. The gripper and peg follow the actual toy state.

Here is how a frame is made:

1. `poseFor(state)` maps the hand's rail position, height, grip, and insertion depth
   into drawing coordinates. Partial insertion lowers the peg by one visual unit.
2. During an action, smooth interpolation moves from the previous drawing pose to
   the new one. A reset immediately places the objects in their new snapshot.
3. `scene()` places a two-link arm between a fixed shoulder and the moving wrist.
   It computes the elbow with the cosine rule. This is **visual inverse kinematics**:
   a way to draw the arm, not an extra learned action or motion planner.
4. Rounded-box meshes form the fixture and gripper. Revolving a radius/height
   profile creates the arm shells, joints, peg, socket collar, and fasteners.
5. A shadow pass renders depth from the light. The main pass uses that depth,
   surface normals, soft shadow filtering, diffuse light, and specular highlights.
6. The same scene is drawn again into the close-up rectangle with another camera.
   Both views therefore show the same instant of the running learner.

Drag the scene to orbit, use the Studio / Front / Overhead buttons, or focus the
canvas and use arrow keys (Home restores Studio). The close-up can be switched off.
Camera operations never call the learning API. Rendering stops when the scene is
still and resumes on an action, camera change, or resize. Reduced-motion settings
disable the movement interpolation. If WebGL is unavailable, learning and the
written lesson remain available with a visible rendering error message.

The renderer's comments follow the same order as this description. It is kept in
a separate file so the learning algorithm stays small and readable.

## GitHub Pages

Push this project to `main`. In the repository's **Settings → Pages**, select
**GitHub Actions** as the source (already configured for the original repository).
The included workflow compiles the WASM, runs native and WASM tests, uploads `dist/`,
and deploys it. Subsequent pushes repeat that process automatically. A fork needs
Pages/Actions enabled under its own settings. All asset URLs are relative, so a
repository subpath works. The published source viewer loads the same Rust file
that the workflow compiled into the module.

## Sources and scope

- [AlphaNIST, §§IV–VI](https://arxiv.org/html/2610.13354v1): mixed cached states,
  success-guided sampling, terminal assembly reward, shared PPO training, limitations.
- [OmniReset, §3.2](https://arxiv.org/html/2603.15789v3#S3.SS2): generating and validating
  grasped and near-goal reset configurations without demonstration trajectories.

This teaching project implements only a small analogy. It has no contact forces,
perception, continuous control, multiple tasks, neural generalization, or robot
hardware validation. The original papers' performance numbers are not demo results.

Original demo code: MIT license. The linked papers retain their own licenses.
