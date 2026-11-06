"""The agent: a table of scores plus the rule that improves it.

There is no model. `q` is a flat list of 6,000 floats. The four state numbers are
folded into one row number, and the action picks one of the two values in that
row.

This table used to be a five dimensional numpy array. Both forms give the same
numbers, bit for bit, and a full training run with each ends on the same table.
The list is about three times faster, because numpy spends more time building
views and checking shapes than it spends comparing two floats.

`act` and `greedy` look the same but they are not, and mixing them up is the
easiest way to publish a number that is not true. `act` explores during training.
`greedy` never guesses and is the only thing that should be evaluated or filmed.
"""
from __future__ import annotations

import json
import random

from env import N_ACTIONS, N_DX, N_DY, N_VY, N_Y

# 3,000 rows, two scores each: coast at the even index, flap at the odd one.
N_ROWS = N_DX * N_DY * N_VY * N_Y
Q_SIZE = N_ROWS * N_ACTIONS


class QAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.95, seed: int = 0):
        # One allocation at startup. Nothing is allocated during play.
        self.q = [0.0] * Q_SIZE
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ indexing
    @staticmethod
    def row(state) -> int:
        """Fold four bucket numbers into one row number, 0 to 2,999.

        Worked example for state (5, 4, 0, 3):

            5 * 10 + 4   = 54      distance and offset as one number
            54 * 5 + 0   = 270     add speed
            270 * 10 + 3 = 2703    add height, this is the row

        Row 2703 holds two scores, at index 5406 and 5407 of the flat list.
        """
        return ((state[0] * N_DY + state[1]) * N_VY + state[2]) * N_Y + state[3]

    def scores(self, state):
        """The coast and flap scores for one state, in that order."""
        i = self.row(state) * N_ACTIONS
        return self.q[i], self.q[i + 1]

    # -------------------------------------------------------------------- acting
    def act(self, state, epsilon: float) -> int:
        # The exploration coin flip. At epsilon 1.0 every action is random. By
        # the end of training one call in fifty guesses.
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)

        i = self.row(state) * N_ACTIONS
        coast, flap = self.q[i], self.q[i + 1]
        if coast > flap:
            return 0
        if flap > coast:
            return 1
        # Equal scores, so there is no opinion to follow. Guess.
        return self.rng.choice([0, 1])

    # Same lookup, no randomness. Everything measured and rendered uses this.
    def greedy(self, state) -> int:
        i = self.row(state) * N_ACTIONS
        # Ties go to flap on purpose. A fresh table then looks like a bird that
        # panics, not a bird that does nothing, which is honest about how little
        # it knows.
        return 1 if self.q[i + 1] >= self.q[i] else 0

    # ------------------------------------------------------------------ learning
    # One step of Q-learning. `target` is what the score should have been: the
    # reward just received, plus a discounted guess at the future (0 if the life
    # just ended). The score then moves a fraction (alpha) of the way toward it.
    #
    # `done` is not a detail. Bootstrapping past a terminal state teaches the
    # agent that dying still has a future, which is how a learner talks itself
    # into crashing.
    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        i = self.row(state) * N_ACTIONS + action

        if done:
            # No future. The target is just what this step paid.
            target = reward
        else:
            j = self.row(next_state) * N_ACTIONS
            # The better of the two choices available in the next state.
            target = reward + self.gamma * max(self.q[j], self.q[j + 1])

        # Move the stored score toward the target. A score can never reach the
        # target in one step unless alpha is 1.0, so the table creeps instead of
        # jumping.
        self.q[i] += self.alpha * (target - self.q[i])

    # --------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        # Line 1 is metadata, line 2 is the table. The table is written as nested
        # lists so numpy, the renderers and older checkpoints all keep working.
        with open(path, "w") as fh:
            json.dump({"alpha": self.alpha, "gamma": self.gamma}, fh)
            fh.write("\n")
            json.dump(self._nested(), fh)

    @classmethod
    def load(cls, path: str) -> "QAgent":
        with open(path) as fh:
            meta = json.loads(fh.readline())
            nested = json.loads(fh.read())
        agent = cls(alpha=meta.get("alpha", 0.15), gamma=meta.get("gamma", 0.95))
        agent.q = cls._flatten(nested)
        return agent

    # -------------------------------------------------------------- file shape
    def _nested(self):
        """The flat table rebuilt as five nested lists, the shape on disk.

        This runs once per save, so it can be plain and slow. The loop order must
        match `row`: distance, offset, speed, height, then the two scores.
        """
        i = 0
        outer = []
        for _ in range(N_DX):
            level_dx = []
            for _ in range(N_DY):
                level_dy = []
                for _ in range(N_VY):
                    level_vy = []
                    for _ in range(N_Y):
                        level_vy.append([self.q[i], self.q[i + 1]])
                        i += 2
                    level_dy.append(level_vy)
                level_dx.append(level_dy)
            outer.append(level_dx)
        return outer

    @staticmethod
    def _flatten(nested):
        """Nested lists from a file back into the flat list the agent uses."""
        flat = []
        for outer in nested:
            for level_dx in outer:
                for level_dy in level_dx:
                    for pair in level_dy:
                        # Each pair is [coast, flap]. float() rejects junk early.
                        flat.append(float(pair[0]))
                        flat.append(float(pair[1]))

        # A wrong shape means a corrupt or old file. Fail here, not in the game.
        if len(flat) != Q_SIZE:
            raise ValueError(f"table holds {len(flat)} values, expected {Q_SIZE}")
        return flat
