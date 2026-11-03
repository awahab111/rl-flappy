//! The agent. A flat `Vec<f64>` of 6,000 numbers and one rule for improving it.
//!
//! Python indexes a five dimensional numpy array with a tuple. Rust has no
//! cheap tuple index, so every state is folded into one flat index and the
//! table is a single allocation. That is the whole difference. It also happens
//! to be the reason the Rust version is fast.

use crate::env::{N_ACTIONS, N_DY, N_STATES, N_VY, N_Y, Obs, Rng};

pub struct Agent {
    pub q: Vec<f64>,
    pub alpha: f64,
    pub gamma: f64,
    rng: Rng,
}

impl Agent {
    pub fn new(alpha: f64, gamma: f64, seed: u64) -> Self {
        Agent {
            // 6,000 zeros. No dictionary, no hash, no allocation during play.
            q: vec![0.0; N_STATES * N_ACTIONS],
            alpha,
            gamma,
            rng: Rng::new(seed ^ 0x5DEE_CE66_D1CE_B00F),
        }
    }

    /// Fold four bucket indexes into one offset. Same row of the table, in a
    /// single addition chain instead of four pointer hops.
    #[inline]
    fn base(state: Obs) -> usize {
        ((state[0] * N_DY + state[1]) * N_VY + state[2]) * N_Y + state[3]
    }

    /// Training action. Explores with probability epsilon, otherwise takes the
    /// better score and breaks ties at random.
    pub fn act(&mut self, state: Obs, epsilon: f64) -> usize {
        if self.rng.unit() < epsilon {
            return self.rng.below(N_ACTIONS);
        }
        let row = Self::base(state) * N_ACTIONS;
        let coast = self.q[row];
        let flap = self.q[row + 1];
        if coast > flap {
            0
        } else if flap > coast {
            1
        } else {
            self.rng.below(N_ACTIONS)
        }
    }

    /// Evaluation action. Never guesses, and breaks a tie toward flapping, so
    /// an untrained table looks like a bird that panics instead of a bird that
    /// does nothing.
    pub fn greedy(&self, state: Obs) -> usize {
        let row = Self::base(state) * N_ACTIONS;
        if self.q[row + 1] >= self.q[row] { 1 } else { 0 }
    }

    /// One Q-learning step.
    ///
    /// `target` is what the score should have been: the reward just received,
    /// plus a discounted guess at the future. A terminal step has no future,
    /// so it must not bootstrap. Getting that wrong teaches the agent that
    /// dying still pays.
    #[inline]
    pub fn update(&mut self, state: Obs, action: usize, reward: f64, next: Obs, done: bool) {
        let row = Self::base(state) * N_ACTIONS + action;
        let target = if done {
            reward
        } else {
            let n = Self::base(next) * N_ACTIONS;
            reward + self.gamma * self.q[n].max(self.q[n + 1])
        };
        self.q[row] += self.alpha * (target - self.q[row]);
    }
}
