//! The agent. A flat `Vec<f64>` of 6,000 numbers and one rule for improving it.
//!
//! Read the file in this order:
//!   1. `base`    four bucket numbers turned into one row number
//!   2. `act`     the action during training, which sometimes guesses
//!   3. `greedy`  the action during testing, which never guesses
//!   4. `update`  the one line that does the learning
//!
//! Python indexes a five dimensional numpy array with a tuple. Rust has no cheap
//! tuple index, so every state folds into one flat index over a single
//! allocation. That change is also most of the speed difference.

// `use` imports names from another module. `crate` means "this program". The
// module `env` is the file env.rs, declared in main.rs with `mod env;`.
use crate::env::{N_ACTIONS, N_DY, N_STATES, N_VY, N_Y, Obs, Rng};

/// The table and the settings for learning.
pub struct Agent {
    // The whole table. 6000 floats, flat. Row r starts at index r * 2 and holds
    // the coast score, then the flap score.
    pub q: Vec<f64>,

    // How far a score moves on one update. 0.20 means "move a fifth of the way".
    pub alpha: f64,

    // How much the future counts. 0.95 means a reward ten steps away is worth
    // 0.95^10, about 0.60, of a reward today.
    pub gamma: f64,

    // The generator for the guessing and the tie breaks. Separate from the
    // game's generator, so the two never consume each other's numbers.
    rng: Rng,
}

impl Agent {
    /// Build an empty table, all zeros.
    pub fn new(alpha: f64, gamma: f64, seed: u64) -> Self {
        Agent {
            // `vec![0.0; 3000 * 2]` makes a Vec with 6000 copies of 0.0. This is
            // the only allocation in the whole program's hot path. No hash map,
            // no per step allocation.
            q: vec![0.0; N_STATES * N_ACTIONS],
            alpha,
            gamma,

            // `^` is XOR. Flipping some seed bits gives the agent a different
            // stream from the game, even though both start from the same seed
            // value. Without this, the two streams would be identical, which is
            // confusing to reason about but not fatal.
            rng: Rng::new(seed ^ 0x5DEE_CE66_D1CE_B00F),
        }
    }

    /// Fold four bucket numbers into the first index of one table row.
    ///
    /// This is the line that replaces numpy's multi dimensional indexing, so it
    /// is worth reading slowly.
    ///
    /// Worked example with state [5, 4, 0, 3], which means distance bucket 5,
    /// offset bucket 4, speed bucket 0, height bucket 3:
    ///
    ///   5 * 10 + 4  = 54        distance and offset, as one number
    ///   54 * 5 + 0  = 270       add speed
    ///   270 * 10 + 3 = 2703     add height, this is the row
    ///
    /// Row 2703 of 3000. `act` and `update` then multiply by 2 to reach the
    /// coast and flap scores of that row.
    #[inline]
    fn base(state: Obs) -> usize {
        // Cayley-Dickson style packing: multiply the accumulator by the size of
        // the next dimension, then add the next bucket number.
        //
        // `Self` means the type being implemented, here Agent. The function has
        // no `self` argument, so it is called as Agent::base(state).
        ((state[0] * N_DY + state[1]) * N_VY + state[2]) * N_Y + state[3]
    }

    /// The action to take during training. Guesses with probability epsilon.
    ///
    /// `&mut self` is needed because the RNG changes on every call.
    pub fn act(&mut self, state: Obs, epsilon: f64) -> usize {
        // The exploration coin flip. At epsilon 1.0 this is always true, so the
        // first lives are pure random play. By the end epsilon is 0.02, so one
        // call in fifty guesses.
        if self.rng.unit() < epsilon {
            // `below(2)` gives 0 or 1 with equal chance.
            return self.rng.below(N_ACTIONS);
        }

        // Row 2703 starts at index 2703 * 2 = 5406. Coast sits there, flap at
        // 5407. One multiply, then two reads.
        let row = Self::base(state) * N_ACTIONS;
        let coast = self.q[row];
        let flap = self.q[row + 1];

        // Take the bigger score. On a fresh table both are zero, so the last
        // branch runs and the choice is random.
        if coast > flap {
            0
        } else if flap > coast {
            1
        } else {
            self.rng.below(N_ACTIONS)
        }
    }

    /// The action to take when testing. Never guesses.
    ///
    /// `&self` and not `&mut self`: this method only reads, so it can be called
    /// on a shared reference and cannot accidentally change the table.
    pub fn greedy(&self, state: Obs) -> usize {
        let row = Self::base(state) * N_ACTIONS;

        // `>=` breaks a tie toward flapping. An untrained table then looks like
        // a bird that panics rather than a bird that does nothing, which is
        // honest about how little it knows.
        if self.q[row + 1] >= self.q[row] { 1 } else { 0 }
    }

    /// One Q-learning step. This is the only place the table changes.
    ///
    /// In words: the score for this state and this action should have been the
    /// reward just received, plus a discounted guess at the future. Move the
    /// stored score a fraction (alpha) of the way toward that value.
    ///
    /// A terminal step has no future, so it must not add a guess. Getting that
    /// wrong teaches the agent that dying still pays.
    #[inline]
    pub fn update(&mut self, state: Obs, action: usize, reward: f64, next: Obs, done: bool) {
        // The exact index of the number being changed: the row, plus 0 for
        // coast or 1 for flap.
        let row = Self::base(state) * N_ACTIONS + action;

        // `if` as an expression, assigned to target.
        let target = if done {
            // The episode ended. The future is worth zero, by definition.
            reward
        } else {
            // The next state's row.
            let n = Self::base(next) * N_ACTIONS;

            // `self.q[n].max(self.q[n + 1])` is the better of the two choices in
            // the next state. Multiply by gamma to discount it, then add the
            // reward just received.
            reward + self.gamma * self.q[n].max(self.q[n + 1])
        };

        // The update itself. `+=` means add to the value already there.
        //
        // target - current is the error. alpha * error is the step size. Move
        // that far. A score can never reach the target in one step unless alpha
        // is 1.0, so the table creeps toward the truth instead of jumping.
        self.q[row] += self.alpha * (target - self.q[row]);
    }
}
