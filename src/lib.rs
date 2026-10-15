//! Assembly from rewards — an intentionally small teaching model.
//!
//! This is NOT a reproduction of AlphaNIST. It keeps three ideas: diverse reset
//! states, success-guided reset sampling, and a terminal success reward. We use
//! tabular Q-learning instead of PPO so every learned number is inspectable.
//! The transfer module compares one shared table with separate task tables.
//! No crates, neural networks, physics engine, demonstrations, or build framework.

use std::sync::Mutex;
mod transfer;

const ACTIONS: usize = 6; // left, right, lift, lower, close, open
const STATES: usize = 5 * 2 * 5 * 3 * 2;
const RESETS: usize = 5;
const HISTORY: usize = 32;
const HORIZON: u32 = 40;
const ALPHA: f64 = 0.25; // how much one observation changes a table entry
const GAMMA: f64 = 0.96; // preference for rewards reached sooner
const EPSILON: f64 = 0.25; // explore on 25% of training actions

// A state describes where things ARE; it does not prescribe what to do next.
// y=0 is table/socket-mouth height; y=1 is lifted. The socket is at x=4.
// depth=0 is outside, 1 is partly inserted, 2 is complete (terminal).
#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    x: u32,
    y: u32,
    peg: u32,
    depth: u32,
    held: bool,
}

impl State {
    fn id(self) -> usize {
        ((((self.x * 2 + self.y) * 5 + self.peg) * 3 + self.depth) * 2 + self.held as u32) as usize
    }
}

/// Generate snapshots directly from known geometry, not a demonstrated motion.
/// Our discrete substitute for OmniReset's physical perturbations is to start
/// with an inserted peg and subtract one unit of insertion depth.
fn reset_state(kind: usize) -> State {
    let complete = State {
        x: 4,
        y: 0,
        peg: 4,
        depth: 2,
        held: true,
    };
    match kind {
        0 => State {
            x: 2,
            y: 1,
            peg: 0,
            depth: 0,
            held: false,
        },
        1 => State {
            x: 0,
            y: 0,
            peg: 0,
            depth: 0,
            held: false,
        },
        2 => State {
            x: 0,
            y: 1,
            peg: 0,
            depth: 0,
            held: true,
        },
        3 => State {
            depth: 0,
            ..complete
        },
        _ => State {
            depth: complete.depth - 1,
            ..complete
        },
    }
}

/// The whole toy "physics engine". Invalid actions simply leave the state alone.
/// There is no reward for grasping, lifting, alignment, or partial insertion.
fn transition(mut s: State, action: usize) -> State {
    match action {
        0 | 1 => {
            // A held peg must be raised before it can travel sideways.
            if !s.held || (s.y == 1 && s.depth == 0) {
                if action == 0 {
                    s.x = s.x.saturating_sub(1);
                } else {
                    s.x = (s.x + 1).min(4);
                }
            }
        }
        2 => {
            if s.held && s.depth > 0 {
                s.depth -= 1;
            } else {
                s.y = 1;
            }
        }
        3 => {
            if s.y == 1 {
                s.y = 0;
            } else if s.held && s.x == 4 {
                s.depth = (s.depth + 1).min(2);
            }
        }
        4 => {
            if s.y == 0 && s.x == s.peg {
                s.held = true;
            }
        }
        5 => {
            s.held = false;
        } // an uninserted released peg falls to the table
        _ => {}
    }
    if s.held {
        s.peg = s.x;
    }
    s
}

/// Tiny seeded RNG. Reproducible teaching runs, NOT cryptography.
fn random(rng: &mut u32) -> f64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 17;
    *rng ^= *rng << 5;
    *rng as f64 / 4294967296.0
}

struct Lab {
    q: [[f64; ACTIONS]; STATES],
    // Each reset remembers its last 32 successes/failures, not its whole history.
    outcomes: [[u8; HISTORY]; RESETS],
    counts: [usize; RESETS],
    cursor: [usize; RESETS],
    visits: [u32; RESETS],
    rng: u32,
    mode: u32, // 0: success-guided mixed resets, 1: uniform mixed, 2: table only
    state: State,
    previous: State,
    reset_kind: usize,
    steps: u32,
    done: bool,
    training: bool,
    episodes: u32,
    wins: u32,
    updates: u32,
    action: i32,
    reward: f64,
    old_q: f64,
    next_value: f64,
    target: f64,
    new_q: f64,
    explored: bool,
}

impl Lab {
    const fn new() -> Self {
        let s = State {
            x: 2,
            y: 1,
            peg: 0,
            depth: 0,
            held: false,
        };
        Self {
            q: [[0.0; ACTIONS]; STATES],
            outcomes: [[0; HISTORY]; RESETS],
            counts: [0; RESETS],
            cursor: [0; RESETS],
            visits: [0; RESETS],
            rng: 7,
            mode: 0,
            state: s,
            previous: s,
            reset_kind: 0,
            steps: 0,
            done: true,
            training: true,
            episodes: 0,
            wins: 0,
            updates: 0,
            action: -1,
            reward: 0.0,
            old_q: 0.0,
            next_value: 0.0,
            target: 0.0,
            new_q: 0.0,
            explored: false,
        }
    }

    fn success_rate(&self, i: usize) -> f64 {
        if self.counts[i] == 0 {
            return 0.0;
        }
        self.outcomes[i].iter().map(|v| *v as f64).sum::<f64>() / self.counts[i] as f64
    }

    // AlphaNIST's sampling kernel. It peaks at p=0.5, not at the easiest reset.
    // The positive floor keeps untried, difficult, and mastered resets available.
    fn weight(&self, i: usize) -> f64 {
        match self.mode {
            1 => 1.0,
            2 => {
                if i == 0 {
                    1.0
                } else {
                    0.0
                }
            }
            _ => {
                let p = self.success_rate(i);
                0.05 + (p * (1.0 - p)).sqrt()
            }
        }
    }

    fn sample_reset(&mut self) -> usize {
        let total: f64 = (0..RESETS).map(|i| self.weight(i)).sum();
        let mut ticket = random(&mut self.rng) * total;
        for i in 0..RESETS {
            ticket -= self.weight(i);
            if ticket < 0.0 {
                return i;
            }
        }
        RESETS - 1
    }

    fn start(&mut self, kind: usize, training: bool) {
        self.state = reset_state(kind);
        self.previous = self.state;
        self.reset_kind = kind;
        self.steps = 0;
        self.done = false;
        self.training = training;
        self.action = -1;
        self.reward = 0.0;
        if training {
            self.visits[kind] += 1;
        }
    }

    fn best_value(&self, s: State) -> f64 {
        self.q[s.id()]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    // Evaluation uses deterministic tie-breaking. Training randomizes ties so
    // an all-zero table does not always choose "left" before it has learned.
    fn choose_action(&mut self) -> usize {
        self.explored = self.training && random(&mut self.rng) < EPSILON;
        if self.explored {
            return (random(&mut self.rng) * ACTIONS as f64) as usize;
        }
        let row = self.q[self.state.id()];
        let best = self.best_value(self.state);
        let choices: Vec<usize> = (0..ACTIONS).filter(|&a| row[a] == best).collect();
        let index = if self.training {
            (random(&mut self.rng) * choices.len() as f64) as usize
        } else {
            0
        };
        choices[index]
    }

    /// One REAL action and one REAL learning update. The browser animates this
    /// exact transition; it never supplies a scripted solution or fake score.
    fn step(&mut self) {
        if self.done {
            return;
        }
        let a = self.choose_action();
        let before = self.state;
        let after = transition(before, a);
        let success = after.depth == 2;
        let reward = if success { 1.0 } else { 0.0 };

        // Q(s,a) is an estimate of discounted future reward after taking a.
        // A zero-reward move can improve if its NEXT state already has value.
        let old = self.q[before.id()][a];
        let next = if success { 0.0 } else { self.best_value(after) };
        let target = reward + GAMMA * next;
        let updated = old + ALPHA * (target - old);
        if self.training {
            self.q[before.id()][a] = updated;
            self.updates += 1;
        }

        self.previous = before;
        self.state = after;
        self.action = a as i32;
        self.reward = reward;
        self.old_q = old;
        self.next_value = next;
        self.target = target;
        self.new_q = if self.training { updated } else { old };
        self.steps += 1;
        self.done = success || self.steps >= HORIZON;

        // A 40-action timeout truncates collection, not the physical task.
        // Thus its last Q update still bootstraps from next-state value.
        if self.done && self.training {
            let i = self.reset_kind;
            self.outcomes[i][self.cursor[i]] = success as u8;
            self.cursor[i] = (self.cursor[i] + 1) % HISTORY;
            self.counts[i] = (self.counts[i] + 1).min(HISTORY);
            self.episodes += 1;
            self.wins += success as u32;
        }
    }

    /// Read-only test from the hardest start. No exploration, updates, sampling,
    /// or RNG changes. This is one deterministic test, NOT a success-rate study.
    fn evaluation_steps(&self) -> u32 {
        let mut s = reset_state(0);
        for step in 1..=HORIZON {
            let row = self.q[s.id()];
            let mut a = 0;
            for candidate in 1..ACTIONS {
                if row[candidate] > row[a] {
                    a = candidate;
                }
            }
            s = transition(s, a);
            if s.depth == 2 {
                return step;
            }
        }
        0
    }
}

// A small numeric interface keeps the browser bridge dependency-free.
// The mutex also lets Rust's native test runner safely call these exports.
static LAB: Mutex<Lab> = Mutex::new(Lab::new());

#[no_mangle]
pub extern "C" fn reset(seed: u32, mode: u32) {
    let mut lab = LAB.lock().unwrap();
    *lab = Lab::new();
    lab.rng = seed.max(1);
    lab.mode = mode.min(2);
}

#[no_mangle]
pub extern "C" fn begin(kind: i32, training: u32) {
    let mut lab = LAB.lock().unwrap();
    let i = if kind < 0 {
        lab.sample_reset()
    } else {
        (kind as usize).min(RESETS - 1)
    };
    lab.start(i, training != 0);
}

#[no_mangle]
pub extern "C" fn step() {
    LAB.lock().unwrap().step();
}

#[no_mangle]
pub extern "C" fn train(episodes: u32) {
    let mut lab = LAB.lock().unwrap();
    for _ in 0..episodes.min(100_000) {
        let i = lab.sample_reset();
        lab.start(i, true);
        while !lab.done {
            lab.step();
        }
    }
}

#[no_mangle]
pub extern "C" fn evaluate() -> u32 {
    LAB.lock().unwrap().evaluation_steps()
}

/// Fields 0..5: current state; 10..15: previous state.
/// Remaining field meanings are mirrored in web/app.js's snapshot().
#[no_mangle]
pub extern "C" fn read(field: u32) -> f64 {
    let l = LAB.lock().unwrap();
    if field < 5 || (10..15).contains(&field) {
        let s = if field < 5 { l.state } else { l.previous };
        return match field % 10 {
            0 => s.x as f64,
            1 => s.y as f64,
            2 => s.peg as f64,
            3 => s.depth as f64,
            _ => s.held as u32 as f64,
        };
    }
    match field {
        20 => l.steps as f64,
        21 => l.done as u32 as f64,
        22 => l.training as u32 as f64,
        23 => l.episodes as f64,
        24 => l.wins as f64,
        25 => l.action as f64,
        26 => l.reward,
        27 => l.old_q,
        28 => l.next_value,
        29 => l.target,
        30 => l.new_q,
        31 => l.reset_kind as f64,
        32 => l.explored as u32 as f64,
        33 => l.updates as f64,
        _ => 0.0,
    }
}

#[no_mangle]
pub extern "C" fn reset_stat(kind: u32, field: u32) -> f64 {
    let l = LAB.lock().unwrap();
    let i = (kind as usize).min(RESETS - 1);
    match field {
        0 => l.success_rate(i),
        1 => l.weight(i) / (0..RESETS).map(|j| l.weight(j)).sum::<f64>(),
        2 => l.visits[i] as f64,
        _ => l.counts[i] as f64,
    }
}

#[no_mangle]
pub extern "C" fn action_value(action: u32) -> f64 {
    let l = LAB.lock().unwrap();
    l.q[l.state.id()][(action as usize).min(ACTIONS - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewards_require_complete_insertion() {
        let partial = reset_state(4);
        assert_eq!(partial.depth, 1);
        assert_eq!(transition(partial, 3).depth, 2);
        for a in [0, 1, 2, 4, 5] {
            assert_ne!(transition(partial, a).depth, 2);
        }
        // The engine can complete a full task; this sequence is ONLY a test.
        let mut s = reset_state(0);
        for a in [0, 0, 3, 4, 2, 1, 1, 1, 1, 3, 3, 3] {
            s = transition(s, a);
        }
        assert_eq!(s.depth, 2);
    }

    #[test]
    fn reset_sampling_is_normalized_and_never_starves_a_reset() {
        let mut l = Lab::new();
        l.counts = [32; RESETS];
        l.outcomes[1] = [1; HISTORY];
        for i in 0..16 {
            l.outcomes[2][i] = 1;
        }
        assert!(l.weight(2) > l.weight(0));
        assert_eq!(l.weight(0), l.weight(1));
        for i in 0..RESETS {
            assert!(l.weight(i) > 0.0);
        }
    }

    #[test]
    fn mixed_resets_learn_full_task_across_seeds() {
        for seed in 1..=12 {
            let mut l = Lab::new();
            l.rng = seed;
            for _ in 0..3000 {
                let i = l.sample_reset();
                l.start(i, true);
                while !l.done {
                    l.step();
                }
            }
            assert!(l.evaluation_steps() > 0, "failed seed {seed}");
            let q = l.q;
            let rng = l.rng;
            l.start(0, false);
            while !l.done {
                l.step();
            }
            assert_eq!(q, l.q, "watching must not train");
            assert_eq!(rng, l.rng, "evaluation must not consume training RNG");
            assert_eq!(l.state.depth, 2);
        }
    }

    #[test]
    fn all_reachable_states_remain_valid() {
        let mut frontier = vec![
            reset_state(0),
            reset_state(1),
            reset_state(2),
            reset_state(3),
            reset_state(4),
        ];
        let mut seen = [false; STATES];
        while let Some(s) = frontier.pop() {
            if seen[s.id()] {
                continue;
            }
            seen[s.id()] = true;
            assert!(s.x <= 4 && s.y <= 1 && s.peg <= 4 && s.depth <= 2);
            if s.held {
                assert_eq!(s.x, s.peg);
            }
            if s.depth > 0 {
                assert_eq!(s.peg, 4);
            }
            if s.depth == 2 {
                continue;
            }
            for a in 0..ACTIONS {
                frontier.push(transition(s, a));
            }
        }
    }
}
