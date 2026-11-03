"""Tabular Q-learning. The whole agent is ~40 lines because the state is tiny."""
from __future__ import annotations

import json
import random

import numpy as np

from flappy import N_ACTIONS, N_DX, N_DY, N_VY, N_Y

Q_SHAPE = (N_DX, N_DY, N_VY, N_Y, N_ACTIONS)


class QAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.95, seed: int = 0):
        self.Q = np.zeros(Q_SHAPE, dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ acting
    def act(self, state, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(N_ACTIONS)
        q = self.Q[state]
        best = q.max()
        ties = np.flatnonzero(q == best)
        return int(ties[0]) if len(ties) == 1 else int(self.rng.choice(ties.tolist()))

    def greedy(self, state) -> int:
        q = self.Q[state]
        best = q.max()
        ties = np.flatnonzero(q == best)
        # deterministic tie-break at eval time: a fresh table has no opinion, so
        # prefer flapping and let it show the "panic flap" behaviour honestly
        return int(ties[-1])

    # ---------------------------------------------------------------- learning
    def update(self, state, action: int, reward: float, next_state, done: bool) -> None:
        target = reward if done else reward + self.gamma * self.Q[next_state].max()
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"alpha": self.alpha, "gamma": self.gamma}, fh)
            fh.write("\n")
            json.dump(self.Q.tolist(), fh)

    @classmethod
    def load(cls, path: str) -> "QAgent":
        with open(path) as fh:
            meta = json.loads(fh.readline())
            q = json.loads(fh.read())
        agent = cls(alpha=meta.get("alpha", 0.15), gamma=meta.get("gamma", 0.95))
        agent.Q = np.array(q, dtype=np.float64)
        return agent
