//! Three assembly layouts, one shared memory or three separate memories.
//!
//! This deliberately makes transfer inspectable: all tasks use the same peg,
//! socket and actions, but begin 1, 2 or 4 positions away. Their paths overlap.
//! Shared Q rows let an easy task teach a useful ending to a longer task.
//! This is explicit state sharing, NOT learned neural generalization or PPO.
//! No action sequence, value, score or learning curve is supplied in advance.
use super::{random, transition, State, ACTIONS, ALPHA, EPSILON, GAMMA, HISTORY, HORIZON, STATES};
use std::sync::Mutex;

const TASKS: usize = 3;
const STARTS: usize = 3; // table, already held, partly inserted
const DISTANCES: [u32; TASKS] = [1, 2, 4];
type Table = [[f64; ACTIONS]; STATES];

fn start(task: usize, kind: usize) -> State {
    let x = 4 - DISTANCES[task];
    match kind {
        0 => State {
            x,
            y: 0,
            peg: x,
            depth: 0,
            held: false,
        },
        1 => State {
            x,
            y: 1,
            peg: x,
            depth: 0,
            held: true,
        },
        _ => State {
            x: 4,
            y: 0,
            peg: 4,
            depth: 1,
            held: true,
        },
    }
}

fn greedy(q: &Table, s: State) -> usize {
    let mut best = 0;
    for a in 1..ACTIONS {
        if q[s.id()][a] > q[s.id()][best] {
            best = a;
        }
    }
    best
}
fn value(q: &Table, s: State) -> f64 {
    q[s.id()][greedy(q, s)]
}
fn choose(q: &Table, s: State, rng: &mut u32) -> usize {
    if random(rng) < EPSILON {
        return (random(rng) * ACTIONS as f64) as usize;
    }
    let best = value(q, s);
    let mut ties = [0; ACTIONS];
    let mut count = 0;
    for a in 0..ACTIONS {
        if q[s.id()][a] == best {
            ties[count] = a;
            count += 1;
        }
    }
    ties[(random(rng) * count as f64) as usize]
}

#[derive(Clone, Copy)]
struct History {
    outcomes: [u8; HISTORY],
    count: usize,
    cursor: usize,
}
impl History {
    const fn new() -> Self {
        Self {
            outcomes: [0; HISTORY],
            count: 0,
            cursor: 0,
        }
    }
    fn rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.outcomes.iter().map(|x| *x as f64).sum::<f64>() / self.count as f64
        }
    }
    fn weight(&self) -> f64 {
        let p = self.rate();
        0.05 + (p * (1.0 - p)).sqrt()
    }
    fn record(&mut self, success: bool) {
        self.outcomes[self.cursor] = success as u8;
        self.cursor = (self.cursor + 1) % HISTORY;
        self.count = (self.count + 1).min(HISTORY);
    }
}

struct Learner {
    // Shared arm only uses table 0. Separate arm selects table = task.
    q: [Table; TASKS],
    history: [[History; STARTS]; TASKS],
    attempts: [u32; TASKS],
    updates: u32,
    // Provenance: which task actually updated a row? Counts, not credit estimates.
    row_updates: [[u32; STATES]; TASKS],
}
impl Learner {
    const fn new() -> Self {
        Self {
            q: [[[0.0; ACTIONS]; STATES]; TASKS],
            history: [[History::new(); STARTS]; TASKS],
            attempts: [0; TASKS],
            updates: 0,
            row_updates: [[0; STATES]; TASKS],
        }
    }
    fn sample(&self, task: usize, mode: u32, rng: &mut u32) -> usize {
        if mode == 0 {
            return 0;
        } // all full starts isolates sharing from resets
        if mode == 1 {
            return (random(rng) * STARTS as f64) as usize;
        }
        let total: f64 = self.history[task].iter().map(History::weight).sum();
        let mut ticket = random(rng) * total;
        for i in 0..STARTS {
            ticket -= self.history[task][i].weight();
            if ticket < 0.0 {
                return i;
            }
        }
        STARTS - 1
    }
    fn episode(&mut self, task: usize, shared: bool, kind: usize, rng: &mut u32) {
        let table = if shared { 0 } else { task };
        let mut s = start(task, kind);
        let mut success = false;
        for _ in 0..HORIZON {
            let a = choose(&self.q[table], s, rng);
            let next = transition(s, a);
            success = next.depth == 2;
            let target = if success {
                1.0
            } else {
                GAMMA * value(&self.q[table], next)
            };
            self.q[table][s.id()][a] += ALPHA * (target - self.q[table][s.id()][a]);
            self.row_updates[task][s.id()] += 1;
            self.updates += 1;
            s = next;
            if success {
                break;
            }
        }
        self.attempts[task] += 1;
        self.history[task][kind].record(success);
    }
    fn evaluate(&self, task: usize, shared: bool) -> u32 {
        let q = &self.q[if shared { 0 } else { task }];
        let mut s = start(task, 0);
        for i in 1..=HORIZON {
            s = transition(s, greedy(q, s));
            if s.depth == 2 {
                return i;
            }
        }
        0
    }
}

struct Comparison {
    learners: [Learner; 2],
    seed: u32,
    rounds: u32,
    mode: u32,
    // Read-only replay has separate state; never consumes training RNG/updates.
    replay: [State; 2],
    replay_steps: [u32; 2],
    replay_actions: [i32; 2],
    replay_task: usize,
}
impl Comparison {
    const fn new() -> Self {
        let s = State {
            x: 0,
            y: 0,
            peg: 0,
            depth: 0,
            held: false,
        };
        Self {
            learners: [Learner::new(), Learner::new()],
            seed: 7,
            rounds: 0,
            mode: 0,
            replay: [s; 2],
            replay_steps: [0; 2],
            replay_actions: [-1; 2],
            replay_task: 2,
        }
    }
    /// One episode per arm, same task and maximum horizon. Every episode gets
    /// a paired RNG seed so a difference in length cannot shift later seeds.
    /// Actual action counts can differ and are displayed alongside attempts.
    fn train(&mut self, rounds: u32, schedule: u32) {
        for _ in 0..rounds {
            let task = match schedule {
                1 => 0,
                2 => 2,
                _ => self.rounds as usize % TASKS,
            };
            let episode_seed = self
                .seed
                .wrapping_add(self.rounds.wrapping_mul(0x9e3779b9))
                .max(1);
            for arm in 0..2 {
                let mut rng = episode_seed;
                let kind = self.learners[arm].sample(task, self.mode, &mut rng);
                self.learners[arm].episode(task, arm == 0, kind, &mut rng);
            }
            self.rounds += 1;
        }
    }
    fn watch(&mut self, task: usize) {
        self.replay_task = task;
        self.replay = [start(task, 0); 2];
        self.replay_steps = [0; 2];
        self.replay_actions = [-1; 2];
    }
    fn watch_step(&mut self) {
        for arm in 0..2 {
            if self.replay[arm].depth == 2 || self.replay_steps[arm] >= HORIZON {
                continue;
            }
            let q = &self.learners[arm].q[if arm == 0 { 0 } else { self.replay_task }];
            let a = greedy(q, self.replay[arm]);
            self.replay[arm] = transition(self.replay[arm], a);
            self.replay_actions[arm] = a as i32;
            self.replay_steps[arm] += 1;
        }
    }
}
static COMPARISON: Mutex<Comparison> = Mutex::new(Comparison::new());

#[no_mangle]
pub extern "C" fn compare_reset(seed: u32, mode: u32) {
    let mut c = COMPARISON.lock().unwrap();
    *c = Comparison::new();
    c.seed = seed.max(1);
    c.mode = mode.min(2);
}
#[no_mangle]
pub extern "C" fn compare_train(rounds: u32, schedule: u32) {
    COMPARISON
        .lock()
        .unwrap()
        .train(rounds.min(10_000), schedule);
}
#[no_mangle]
pub extern "C" fn compare_evaluate(arm: u32, task: u32) -> u32 {
    let c = COMPARISON.lock().unwrap();
    c.learners[arm.min(1) as usize].evaluate(task.min(2) as usize, arm == 0)
}
#[no_mangle]
pub extern "C" fn compare_watch(task: u32) {
    COMPARISON.lock().unwrap().watch(task.min(2) as usize);
}
#[no_mangle]
pub extern "C" fn compare_step() {
    COMPARISON.lock().unwrap().watch_step();
}
/// Fields: 0..4 replay state; 5 steps; 6 done; 7 action; 8 update count;
/// 9 paired rounds; 10..12 per-task attempts; 20..22 full-start evaluation steps.
#[no_mangle]
pub extern "C" fn compare_read(arm: u32, field: u32) -> f64 {
    let c = COMPARISON.lock().unwrap();
    let a = arm.min(1) as usize;
    let s = c.replay[a];
    match field {
        0 => s.x as f64,
        1 => s.y as f64,
        2 => s.peg as f64,
        3 => s.depth as f64,
        4 => s.held as u32 as f64,
        5 => c.replay_steps[a] as f64,
        6 => (s.depth == 2 || c.replay_steps[a] >= HORIZON) as u32 as f64,
        7 => c.replay_actions[a] as f64,
        8 => c.learners[a].updates as f64,
        9 => c.rounds as f64,
        10..=12 => c.learners[a].attempts[(field - 10) as usize] as f64,
        20..=22 => c.learners[a].evaluate((field - 20) as usize, a == 0) as f64,
        _ => 0.0,
    }
}
/// The transfer microscope: inspect the shared finishing state, a held peg one
/// position from the socket. This same state is reachable in every task.
/// Fields 0..5 are action values under the HARD task's controller; 6..8 count
/// source-task updates to this row (in shared arm) or hard-only updates (separate).
#[no_mangle]
pub extern "C" fn compare_probe(arm: u32, field: u32) -> f64 {
    let c = COMPARISON.lock().unwrap();
    let a = arm.min(1) as usize;
    let l = &c.learners[a];
    let id = start(0, 1).id();
    if field < 6 {
        return l.q[if a == 0 { 0 } else { 2 }][id][field as usize];
    }
    if (6..9).contains(&field) {
        let task = (field - 6) as usize;
        return if a == 0 || task == 2 {
            l.row_updates[task][id] as f64
        } else {
            0.0
        };
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn easy_practice_changes_shared_hard_task_values_but_not_hard_specialist() {
        let mut c = Comparison::new();
        c.train(1000, 1);
        let id = start(0, 1).id();
        assert!(c.learners[0].q[0][id][1] > 0.8);
        assert_eq!(c.learners[0].attempts, [1000, 0, 0]);
        assert_eq!(c.learners[1].attempts, [1000, 0, 0]);
        assert_eq!(c.learners[1].q[2], [[0.0; ACTIONS]; STATES]);
        // With only easy-task training both active tables must be identical:
        // no hidden boost is supplied to the shared controller.
        assert_eq!(c.learners[0].q[0], c.learners[1].q[0]);
        assert_eq!(c.learners[0].updates, c.learners[1].updates);
        // It has learned a finishing suffix, not the unpracticed initial move.
        assert_eq!(c.learners[0].q[0][start(2, 0).id()], [0.0; ACTIONS]);
    }

    #[test]
    fn comparison_has_matched_attempts_and_read_only_evaluation() {
        for mode in 0..3 {
            let mut c = Comparison::new();
            c.mode = mode;
            c.train(90, 0);
            for l in &c.learners {
                assert_eq!(l.attempts, [30, 30, 30]);
            }
            let before = c.learners[0].q;
            let updates = c.learners[0].updates;
            let rounds = c.rounds;
            let outcomes = c.learners[0].history[0][0].outcomes;
            c.watch(2);
            for _ in 0..50 {
                c.watch_step();
            }
            assert_eq!(c.learners[0].q, before);
            assert_eq!(c.learners[0].updates, updates);
            assert_eq!(c.learners[0].history[0][0].outcomes, outcomes);
            assert_eq!(c.rounds, rounds);
            assert!(c.replay_steps[0] <= HORIZON);
        }
    }

    #[test]
    fn sharing_improves_hard_task_learning_over_many_fixed_seeds() {
        let mut wins = [0; 2];
        for seed in 1..=24 {
            let mut c = Comparison::new();
            c.seed = seed;
            c.train(900, 0);
            for arm in 0..2 {
                wins[arm] += (c.learners[arm].evaluate(2, arm == 0) > 0) as u32;
            }
        }
        // An aggregate regression, not a guarantee of improvement in every run.
        assert!(wins[0] >= 20, "shared solved {} of 24 seeds", wins[0]);
        assert!(wins[0] > wins[1] + 12, "shared/separate: {wins:?}");
    }
}
