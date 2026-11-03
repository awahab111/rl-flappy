//! The game. Same rules as env.py, same constants, same order of operations
//! inside a step.
//!
//! This port has one job, which is to behave exactly like the Python version.
//! Two simulators that disagree by a pixel make every comparison meaningless,
//! and comparing them is the whole point here. `--trace` plus
//! `python/parity_trace.py` checks it a frame at a time.
//!
//! No game engine, no random number crate. The RNG is twelve lines of
//! xorshift64* in a struct.

/// xorshift64*, deterministic on every platform.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Mix the seed first, otherwise seeds 1 and 2 start with very similar
        // bit patterns and produce nearly the same level opening.
        Rng { state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1), built from the top 53 bits like every other RNG.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.unit() * n as f64) as usize).min(n - 1)
    }

    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

// ------------------------------------------------------------------ constants
pub const W: f64 = 120.0;
pub const H: f64 = 90.0;
pub const GROUND_Y: f64 = H - 9.0;
pub const BIRD_X: f64 = 26.0;
pub const BIRD_HW: f64 = 3.2;
pub const BIRD_HH: f64 = 2.6;
pub const GRAVITY: f64 = 0.40;
pub const FLAP_V: f64 = -3.2;
pub const MAX_FALL: f64 = 5.5;
pub const PIPE_W: f64 = 11.0;
pub const PIPE_SPEED: f64 = 1.7;
pub const PIPE_GAP: f64 = 30.0;
pub const PIPE_SPACING: f64 = 56.0;
pub const MAX_STEPS: u32 = 2400;

pub const DX_EDGES: [f64; 5] = [8.0, 16.0, 26.0, 38.0, 52.0];
pub const DY_EDGES: [f64; 9] = [-18.0, -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0, 18.0];
pub const VY_EDGES: [f64; 4] = [-2.4, -0.8, 0.8, 2.4];
pub const Y_EDGES: [f64; 9] = [10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 66.0, 74.0];

pub const N_DX: usize = DX_EDGES.len() + 1;
pub const N_DY: usize = DY_EDGES.len() + 1;
pub const N_VY: usize = VY_EDGES.len() + 1;
pub const N_Y: usize = Y_EDGES.len() + 1;
pub const N_ACTIONS: usize = 2;
pub const N_STATES: usize = N_DX * N_DY * N_VY * N_Y;

pub const REWARD_ALIVE: f64 = 0.1;
pub const REWARD_PIPE: f64 = 5.0;
pub const REWARD_DEATH: f64 = -10.0;

/// The four bucket numbers. A plain array, because it is a value, not a thing.
pub type Obs = [usize; 4];

#[inline]
pub fn bucket(value: f64, edges: &[f64]) -> usize {
    let mut i = 0;
    for &e in edges {
        // `>=` not `>`, so a value sitting exactly on an edge belongs to the
        // bucket above it. The Python version documents the same choice.
        if value >= e {
            i += 1;
        } else {
            break;
        }
    }
    i
}

#[derive(Clone, Copy)]
pub struct Pipe {
    pub x: f64,
    pub gap: f64,
    counted: bool,
}

pub struct Flappy {
    pub y: f64,
    pub vy: f64,
    pub t: u32,
    pub passed: u32,
    pub timed_out: bool,
    pub pipes: Vec<Pipe>,
    rng: Rng,
    /// Set for the parity check. The scripted gap sequence removes the RNG from
    /// the equation so Rust and Python can be compared on the same level.
    scripted: bool,
    spawned: usize,
}

impl Flappy {
    pub fn new(seed: u64) -> Self {
        // No episode yet. Like the Python constructor, this must not draw a
        // random number, or a caller that also calls reset() starts on the
        // second level. That bug cost hours once already.
        Flappy {
            y: GROUND_Y * 0.45,
            vy: 0.0,
            t: 0,
            passed: 0,
            timed_out: false,
            pipes: Vec::with_capacity(8),
            rng: Rng::new(seed),
            scripted: false,
            spawned: 0,
        }
    }

    pub fn scripted(seed: u64) -> Self {
        let mut env = Flappy::new(seed);
        env.scripted = true;
        env
    }

    pub fn reset(&mut self) -> Obs {
        self.y = GROUND_Y * 0.45;
        self.vy = 0.0;
        self.t = 0;
        self.passed = 0;
        self.timed_out = false;
        self.pipes.clear();
        self.spawned = 0;
        self.spawn(W + 6.0);
        self.obs()
    }

    fn spawn(&mut self, x: f64) {
        let gap = if self.scripted {
            // 29.0 to 52.0, the same window the real sampler uses.
            29.0 + 23.0 * ((self.spawned * 37 % 100) as f64 / 100.0)
        } else {
            self.rng.uniform(29.0, 52.0)
        };
        self.spawned += 1;
        self.pipes.push(Pipe { x, gap, counted: false });
    }

    /// One frame. The order is load bearing and matches env.py:
    /// action, gravity, scroll, spawn, score, collide.
    pub fn step(&mut self, action: usize) -> (Obs, f64, bool) {
        if action == 1 {
            self.vy = FLAP_V;
        }
        self.vy = (self.vy + GRAVITY).min(MAX_FALL);
        self.y += self.vy;
        self.t += 1;

        let mut reward = REWARD_ALIVE;

        // Pipes move first. A pipe counts as passed when its right edge clears
        // the bird's left edge, not when it disappears.
        let mut newly_passed = 0;
        for p in self.pipes.iter_mut() {
            p.x -= PIPE_SPEED;
            if !p.counted && p.x + PIPE_W < BIRD_X - BIRD_HW {
                p.counted = true;
                newly_passed += 1;
                reward += REWARD_PIPE;
            }
        }
        self.passed += newly_passed;

        self.pipes.retain(|p| p.x > -PIPE_W - 12.0);
        let last_x = self.pipes.last().map(|p| p.x).unwrap_or(W);
        if self.pipes.is_empty() || last_x <= W - PIPE_SPACING {
            self.spawn(last_x + PIPE_SPACING);
        }

        let mut done = false;
        if self.y - BIRD_HH <= 0.0 {
            done = true; // ceiling
        } else if self.y + BIRD_HH >= GROUND_Y {
            done = true; // ground
        } else {
            for p in &self.pipes {
                if self.hits(p) {
                    done = true; // pipe body
                    break;
                }
            }
        }
        if self.t >= MAX_STEPS {
            done = true;
            self.timed_out = true;
        }
        if done {
            // Without this, a fresh table learns that dying pays: every step
            // pays, including the last one, so crashing beats trying.
            reward += REWARD_DEATH;
        }
        (self.obs(), reward, done)
    }

    fn hits(&self, p: &Pipe) -> bool {
        if BIRD_X + BIRD_HW < p.x || BIRD_X - BIRD_HW > p.x + PIPE_W {
            return false;
        }
        let top = p.gap - PIPE_GAP / 2.0;
        let bot = p.gap + PIPE_GAP / 2.0;
        self.y - BIRD_HH < top || self.y + BIRD_HH > bot
    }

    pub fn obs(&self) -> Obs {
        let mut target: Option<&Pipe> = None;
        for p in &self.pipes {
            if p.x + PIPE_W >= BIRD_X - BIRD_HW {
                target = Some(p);
                break;
            }
        }
        let (dx, dy) = match target {
            Some(p) => (p.x - (BIRD_X + BIRD_HW), self.y - p.gap),
            None => (60.0, 0.0),
        };
        [
            bucket(dx, &DX_EDGES),
            bucket(dy, &DY_EDGES),
            bucket(self.vy, &VY_EDGES),
            bucket(self.y, &Y_EDGES),
        ]
    }
}
