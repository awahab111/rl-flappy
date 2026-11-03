//! The game. Same rules as env.py, same constants, same order of operations
//! inside a step.
//!
//! Read the file in this order:
//!   1. `Rng`       the twelve lines that make the "random" numbers
//!   2. constants   the world size and the bucket edges
//!   3. `bucket`    one number turned into one bucket index
//!   4. `Flappy`    the bird, the pipes, and one step of physics
//!
//! Rust words used in this file, for a reader who knows Python or Go:
//!   struct      a group of named values. Like a Go struct.
//!   impl        the block that adds methods to a struct.
//!   pub         visible outside this file. Without it, the item is private.
//!   &mut self   "this method may change the struct". `&self` means read only.
//!   u64         unsigned 64 bit integer, 0 to about 1.8e19.
//!   f64         64 bit float. The same type as a Python float.
//!   usize       an integer used for indexes and lengths. Never negative.
//!   bool        true or false.
//!   Vec<T>      a growable array of T. Like a Go slice with append.
//!   Option<T>   either Some(value) or None. Replaces null.
//!   #[inline]   a hint: paste this code where it is called instead of calling.
//!
//! This port has one job, which is to behave exactly like the Python version.
//! Two simulators that disagree by a pixel make every comparison meaningless,
//! and comparing them is the whole point here. `--trace` plus
//! `python/parity_trace.py` checks it a frame at a time.

/// xorshift64*. One state number, three shifts, one multiply.
///
/// It is not random. It is a fixed rule that turns one number into the next.
/// The seed picks where in the very long sequence you start.
pub struct Rng {
    // The only thing the generator remembers. 64 bits.
    // Every later number comes from this one.
    state: u64,
}

impl Rng {
    /// Make a generator from a seed.
    pub fn new(seed: u64) -> Self {
        // wrapping_mul means "multiply, and let the result wrap if it overflows
        // 64 bits". A plain `*` panics in a debug build when it overflows.
        //
        // The constant is the fractional part of the golden ratio, scaled to 64
        // bits. It scatters nearby seeds. Without it, seed 1 and seed 2 would
        // start with almost the same bits.
        //
        // `| 1` forces the lowest bit to 1. xorshift must not start at zero,
        // because zero stays zero forever.
        Rng { state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    /// Produce the next 64 bit number and move the state forward.
    ///
    /// `&mut self` matters: this method changes `self.state`, so the caller
    /// must own a mutable value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // Copy the state into a local variable. The three shifts work on the
        // copy, then the copy goes back into `self.state`.
        let mut x = self.state;

        // `>>` shifts bits right, `<<` shifts left, `^` is XOR: a bit becomes 1
        // when the two inputs differ. `x ^= y` is short for `x = x ^ y`.
        //
        // The three shifts mix the high bits down into the low bits and back.
        // One bit of change spreads to many bits within three steps.
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;

        // Keep the mixed value as the next state. Same input, same output,
        // forever.
        self.state = x;

        // The multiply hides the pattern further and returns a value that does
        // not simply repeat the state. The low bits of xorshift alone are weak,
        // so the multiply fixes that.
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A float in [0, 1), built from the top 53 bits.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        // An f64 holds 53 bits of precision. Take the top 53 bits, then divide
        // by 2^53 to land between 0 and 1, never reaching 1.
        //
        // `as f64` converts the integer to a float. Rust never converts types
        // on its own, so the cast is explicit.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// An integer from 0 to n-1.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        // Multiply the 0..1 float by n, then `as usize` throws away the
        // fraction. `.min(n - 1)` guards the rare case where rounding pushes the
        // result up to n, which would be out of range.
        ((self.unit() * n as f64) as usize).min(n - 1)
    }

    /// A float in [lo, hi).
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        // Stretch the 0..1 number across the range. At 0 you get lo, at almost
        // 1 you get almost hi.
        lo + (hi - lo) * self.unit()
    }
}

// ------------------------------------------------------------------ constants
//
// `const` means the value is fixed at compile time. The world is 120 units wide
// and 90 tall. One step is one frame of the simulation.
//
// These must match env.py exactly. Change one side and the parity check fails,
// which is the point of the check.

pub const W: f64 = 120.0;
pub const H: f64 = 90.0;

// The ground sits 9 units above the bottom of the field.
pub const GROUND_Y: f64 = H - 9.0;

// The bird does not move forward. It sits at this x and the world scrolls.
pub const BIRD_X: f64 = 26.0;

// Half of the hitbox. The hitbox is 6.4 by 5.2 units, smaller than the sprite.
pub const BIRD_HW: f64 = 3.2;
pub const BIRD_HH: f64 = 2.6;

// Gravity adds this to the vertical speed on every step. A flap sets the speed
// to FLAP_V. Both are negative upward, because y grows downward.
pub const GRAVITY: f64 = 0.40;
pub const FLAP_V: f64 = -3.2;
pub const MAX_FALL: f64 = 5.5;

// Each pipe is 11 units wide and moves 1.7 units per step. New pipes appear
// every 56 units of travel.
pub const PIPE_W: f64 = 11.0;
pub const PIPE_SPEED: f64 = 1.7;
pub const PIPE_SPACING: f64 = 56.0;

// The hole in a pipe is 30 units tall.
pub const PIPE_GAP: f64 = 30.0;

// An episode stops here even if the bird is still alive. 2400 steps at 32.94
// steps per pipe allows 71 pipes, which is why every perfect run scores 71.
pub const MAX_STEPS: u32 = 2400;

// The bucket edges. Each list splits one number into ranges.
//
// `[f64; 5]` is a fixed size array of five floats, not a Vec. The length is part
// of the type, so it can never change.
//
// Five edges make six buckets, because the ranges outside the first and last
// edge count too. That is why N_DX below is `len + 1`.
pub const DX_EDGES: [f64; 5] = [8.0, 16.0, 26.0, 38.0, 52.0];
pub const DY_EDGES: [f64; 9] = [-18.0, -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0, 18.0];
pub const VY_EDGES: [f64; 4] = [-2.4, -0.8, 0.8, 2.4];
pub const Y_EDGES: [f64; 9] = [10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 66.0, 74.0];

// `.len()` on an array runs at compile time, so these are constants, not work
// done at runtime.
pub const N_DX: usize = DX_EDGES.len() + 1;
pub const N_DY: usize = DY_EDGES.len() + 1;
pub const N_VY: usize = VY_EDGES.len() + 1;
pub const N_Y: usize = Y_EDGES.len() + 1;

// Two actions: 0 is coast, 1 is flap.
pub const N_ACTIONS: usize = 2;

// 6 * 10 * 5 * 10 = 3000 states. Each state has two scores, so 6000 numbers.
pub const N_STATES: usize = N_DX * N_DY * N_VY * N_Y;

// Rewards. Small enough to stay readable, big enough to order the choices.
// Survival pays little, a pipe pays fifty survival steps, and death costs twice
// that. Without the death cost, a fresh table learns that crashing is better
// than trying an unknown action.
pub const REWARD_ALIVE: f64 = 0.1;
pub const REWARD_PIPE: f64 = 5.0;
pub const REWARD_DEATH: f64 = -10.0;

/// The four bucket numbers. A plain array, because it is a value, not a thing.
///
/// `[usize; 4]` copies freely. Passing an Obs to a function moves nothing and
/// costs nothing. Python passes a tuple here, with the same idea.
pub type Obs = [usize; 4];

/// Turn one number into a bucket index, from 0 to edges.len().
///
/// `&[f64]` is a slice: a view into an array that also knows its length. It lets
/// the same function accept DX_EDGES or any other list of edges.
#[inline]
pub fn bucket(value: f64, edges: &[f64]) -> usize {
    let mut i = 0;

    // `for &e in edges` walks the slice and copies each float out. The `&` in
    // the pattern is what copies it. Without the `&` you would get references.
    for &e in edges {
        // `>=` and not `>`, so a value sitting exactly on an edge belongs to
        // the bucket above it. env.py makes the same choice, on purpose.
        if value >= e {
            i += 1;
        } else {
            // The edges are sorted, so the first edge above the value ends the
            // search.
            break;
        }
    }
    i
}

/// One pipe. `x` is the left edge, `gap` is the centre of the hole.
#[derive(Clone, Copy)]
pub struct Pipe {
    pub x: f64,
    pub gap: f64,
    // Private: only this file may read or write it.
    counted: bool,
}
// `#[derive(Clone, Copy)]` above tells the compiler to copy this struct by
// value, like an integer. Without it, passing a Pipe to a function would move
// ownership and take the pipe away from the game.

/// The whole game state. One struct per running level.
pub struct Flappy {
    pub y: f64,          // height of the bird, 0 at the top
    pub vy: f64,         // vertical speed, negative means rising
    pub t: u32,          // steps taken in this episode
    pub passed: u32,     // pipes cleared
    pub timed_out: bool, // true when the run ended at the step cap, not a crash
    pub pipes: Vec<Pipe>, // pipes on screen, left to right
    rng: Rng,            // the level generator for this level
    scripted: bool,      // true only for the parity trace
    spawned: usize,      // how many pipes this level has created so far
}

impl Flappy {
    /// Create the game. This does not start a level.
    pub fn new(seed: u64) -> Self {
        // No episode yet. Like the Python constructor, this must not draw a
        // random number. A caller that also calls reset() would otherwise start
        // on the second level. That bug cost hours once already.
        Flappy {
            // Start at 45 percent of the ground height, roughly the middle.
            y: GROUND_Y * 0.45,
            vy: 0.0,
            t: 0,
            passed: 0,
            timed_out: false,
            // with_capacity(8) reserves room for eight pipes up front. It is a
            // performance detail: the array will not reallocate for the first
            // eight pushes.
            pipes: Vec::with_capacity(8),
            rng: Rng::new(seed),
            scripted: false,
            spawned: 0,
        }
    }

    /// Same game, but the gaps come from a fixed list instead of the RNG.
    /// Only the parity check uses this.
    pub fn scripted(seed: u64) -> Self {
        let mut env = Flappy::new(seed);
        // The struct is not mutable by default. `mut` allows the change below.
        env.scripted = true;
        env
    }

    /// Start a new level. Returns the first observation.
    pub fn reset(&mut self) -> Obs {
        self.y = GROUND_Y * 0.45;
        self.vy = 0.0;
        self.t = 0;
        self.passed = 0;
        self.timed_out = false;

        // `clear()` empties the array but keeps the memory for reuse.
        self.pipes.clear();
        self.spawned = 0;

        // The first pipe starts 6 units off the right edge, so the bird gets a
        // moment before it has to act.
        self.spawn(W + 6.0);

        // Return the observation of this first state. The last expression in a
        // function is its return value. No `return` keyword needed.
        self.obs()
    }

    /// Put one pipe on screen. `x` is its left edge.
    fn spawn(&mut self, x: f64) {
        // `if` can produce a value in Rust, so this whole block is one
        // expression assigned to `gap`.
        let gap = if self.scripted {
            // A fixed ladder from 29.0 to 52.0, which is the same window the
            // real sampler uses. `%` is the remainder operator.
            29.0 + 23.0 * ((self.spawned * 37 % 100) as f64 / 100.0)
        } else {
            // A gap centre between 14 units from the ceiling and 14 units from
            // the ground, minus half the hole.
            self.rng.uniform(29.0, 52.0)
        };

        self.spawned += 1;

        // `Pipe { x, gap, counted: false }` builds the struct. When a variable
        // has the same name as the field, the short form `x` works.
        self.pipes.push(Pipe { x, gap, counted: false });
    }

    /// One frame. The order matters and matches env.py:
    /// action, gravity, scroll, spawn, score, collide.
    ///
    /// Returns a tuple: the new observation, the reward, and whether the
    /// episode ended.
    pub fn step(&mut self, action: usize) -> (Obs, f64, bool) {
        // 1. The action. A flap sets the speed upward in one move, it does not
        // add to it.
        if action == 1 {
            self.vy = FLAP_V;
        }

        // 2. Gravity and movement. `.min(MAX_FALL)` caps the fall speed so the
        // bird cannot accelerate forever.
        self.vy = (self.vy + GRAVITY).min(MAX_FALL);
        self.y += self.vy;
        self.t += 1;

        // Every surviving step pays a little. This is the only reward that does
        // not need a pipe.
        let mut reward = REWARD_ALIVE;

        // 3. The world scrolls.
        //
        // `iter_mut()` walks the array and allows changes to each pipe. The
        // score counter is a local variable, then added after the loop, because
        // Rust will not let you borrow the array and change `self.passed` at the
        // same time.
        let mut newly_passed = 0;
        for p in self.pipes.iter_mut() {
            p.x -= PIPE_SPEED;

            // A pipe counts once, when its right edge clears the bird's left
            // edge. `!p.counted` stops it paying twice.
            if !p.counted && p.x + PIPE_W < BIRD_X - BIRD_HW {
                p.counted = true;
                newly_passed += 1;
                reward += REWARD_PIPE;
            }
        }
        self.passed += newly_passed;

        // 4. Drop pipes that have left the screen. `retain` keeps the items that
        // pass the test. The closure `|p| ...` is an inline function that takes
        // one pipe and returns a bool.
        self.pipes.retain(|p| p.x > -PIPE_W - 12.0);

        // 5. Add a new pipe when the last one has moved far enough left.
        //
        // `.last()` gives Option<&Pipe>. `.map(|p| p.x)` turns that into
        // Option<f64>. `.unwrap_or(W)` uses W when the list is empty.
        let last_x = self.pipes.last().map(|p| p.x).unwrap_or(W);
        if self.pipes.is_empty() || last_x <= W - PIPE_SPACING {
            self.spawn(last_x + PIPE_SPACING);
        }

        // 6. Collisions. Checked after the move and after scoring, so a bird
        // that reaches a pipe and a bird that dies on it are judged on the same
        // positions.
        let mut done = false;

        // The ceiling. The top of the bird is y minus half its height.
        if self.y - BIRD_HH <= 0.0 {
            done = true;
        } else if self.y + BIRD_HH >= GROUND_Y {
            // The ground.
            done = true;
        } else {
            // The pipe bodies. `&self.pipes` borrows the list read only.
            for p in &self.pipes {
                if self.hits(p) {
                    done = true;
                    break;
                }
            }
        }

        // 7. The step cap. The run ends, but this is not a crash.
        if self.t >= MAX_STEPS {
            done = true;
            self.timed_out = true;
        }

        // 8. The death cost. Without it, a fresh table learns that dying pays:
        // every step pays, including the last one, so crashing beats trying.
        if done {
            reward += REWARD_DEATH;
        }

        // The tuple is the return value.
        (self.obs(), reward, done)
    }

    /// Does the bird overlap this pipe body?
    fn hits(&self, p: &Pipe) -> bool {
        // No overlap in x. `||` is OR. Return early, the expensive test is not
        // needed.
        if BIRD_X + BIRD_HW < p.x || BIRD_X - BIRD_HW > p.x + PIPE_W {
            return false;
        }

        // The gap runs from gap minus half the hole to gap plus half the hole.
        let top = p.gap - PIPE_GAP / 2.0;
        let bot = p.gap + PIPE_GAP / 2.0;

        // Touch the upper pipe or the lower pipe. A bird fully inside the hole
        // touches neither.
        self.y - BIRD_HH < top || self.y + BIRD_HH > bot
    }

    /// The four bucket numbers for the current state.
    pub fn obs(&self) -> Obs {
        // The next pipe is the first one whose right edge is at or past the
        // bird's left edge.
        //
        // Option<&Pipe> is either Some(pipe) or None. There is no null in Rust.
        let mut target: Option<&Pipe> = None;
        for p in &self.pipes {
            if p.x + PIPE_W >= BIRD_X - BIRD_HW {
                target = Some(p);
                break;
            }
        }

        // `match` unpacks the Option. Both cases must be handled, so a missing
        // pipe cannot be forgotten.
        let (dx, dy) = match target {
            // Distance to the pipe, and height difference from the gap centre.
            Some(p) => (p.x - (BIRD_X + BIRD_HW), self.y - p.gap),
            // No pipe in front. These numbers are outside every edge, so the
            // bucket is the last one, meaning "far away, level with the gap".
            None => (60.0, 0.0),
        };

        // Four bucket lookups, in the order distance, offset, speed, height.
        // The order must match the table shape in agent.rs.
        [
            bucket(dx, &DX_EDGES),
            bucket(dy, &DY_EDGES),
            bucket(self.vy, &VY_EDGES),
            bucket(self.y, &Y_EDGES),
        ]
    }
}
