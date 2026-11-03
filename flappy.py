"""Flappy-style gym: hand-rolled physics, zero dependencies.

The bird sits at a fixed x and the world scrolls. A tabular agent sees a
discretised slice of reality and chooses flap-or-not. That is the whole game.

State is four small integers:
    dx   distance to the next pipe
    dy   height offset from the middle of its gap
    vy   vertical speed (rising / level / falling)
    y    absolute height  (without this the agent cannot see the ceiling)
"""
from __future__ import annotations

import random

# --- world constants (world units are "pixels" of a 120x90 field) ---
W, H = 120.0, 90.0
GROUND_Y = H - 9.0
BIRD_X = 26.0
BIRD_HW = 3.2          # half-width of the hitbox
BIRD_HH = 2.6          # half-height of the hitbox
GRAVITY = 0.40
FLAP_V = -3.2
MAX_FALL = 5.5
PIPE_W = 11.0
PIPE_SPEED = 1.7
PIPE_GAP = 30.0
PIPE_SPACING = 56.0
MAX_STEPS = 2400

# --- state discretisation ---
DX_EDGES = (8.0, 16.0, 26.0, 38.0, 52.0)                        # gap to next pipe
DY_EDGES = (-18.0, -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0, 18.0)  # offset from gap centre
VY_EDGES = (-2.4, -0.8, 0.8, 2.4)                               # fast rise .. fast fall
Y_EDGES = (10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 66.0, 74.0)  # absolute height
N_DX, N_DY, N_VY, N_Y = (len(DX_EDGES) + 1, len(DY_EDGES) + 1,
                         len(VY_EDGES) + 1, len(Y_EDGES) + 1)
N_ACTIONS = 2                                                  # 0 = coast, 1 = flap

REWARD_ALIVE = 0.1     # tiny: survive, but do not dawdle
REWARD_PIPE = 5.0      # the actual objective, worth ~50 alive-steps
REWARD_DEATH = -10.0   # without this, a fresh table learns "dying pays" and suicides

# potential-based reward shaping (Ng et al. 1999). Off by default: with the
# reward scale above it made things slightly worse, which is worth knowing.
SHAPE_K = 0.0
SHAPE_CAP = 40.0
SHAPE_GAMMA = 0.95


def bucket(value: float, edges: tuple[float, ...]) -> int:
    """Index of the interval that contains value."""
    i = 0
    for e in edges:
        if value >= e:
            i += 1
        else:
            break
    return i


def sample_gap(rng: random.Random) -> float:
    """Vertical centre of one pipe gap. Shared by the gym and the swarm sim."""
    lo = 14.0 + PIPE_GAP / 2
    hi = GROUND_Y - 14.0 - PIPE_GAP / 2
    return rng.uniform(lo, hi)


def observe(pipes, y: float, vy: float):
    """Discretised state for any bird anywhere in the level."""
    target = None
    for p in pipes:
        if p[0] + PIPE_W >= BIRD_X - BIRD_HW:
            target = p
            break
    if target is None:
        dx, dy = 60.0, 0.0
    else:
        dx = target[0] - (BIRD_X + BIRD_HW)
        dy = y - target[1]
    return (bucket(dx, DX_EDGES), bucket(dy, DY_EDGES),
            bucket(vy, VY_EDGES), bucket(y, Y_EDGES))


class Flappy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # No episode yet: reset() draws the level, and constructing an env must
        # not silently consume a random draw (that made callers that also call
        # reset() start on the second level).
        self.y = GROUND_Y * 0.45
        self.vy = 0.0
        self.t = 0
        self.pipes_passed = 0
        self.alive = True
        self.death_reason = ""
        self.pipes: list[list] = []
        self.phi = 0.0

    # ---------------------------------------------------------------- episode
    def reset(self):
        self.y = GROUND_Y * 0.45
        self.vy = 0.0
        self.t = 0
        self.pipes_passed = 0
        self.alive = True
        self.death_reason = ""
        self.pipes: list[list] = []          # [x, gap_centre, counted]
        self._spawn(W + 6.0)
        self.phi = self._phi()
        return self.obs()

    def _spawn(self, x: float) -> None:
        self.pipes.append([x, sample_gap(self.rng), False])

    # ------------------------------------------------------------------ physics
    def step(self, action: int):
        if action == 1:
            self.vy = FLAP_V
        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        self.y += self.vy
        self.t += 1

        reward = REWARD_ALIVE
        for p in self.pipes:
            p[0] -= PIPE_SPEED
            if not p[2] and p[0] + PIPE_W < BIRD_X - BIRD_HW:
                p[2] = True
                self.pipes_passed += 1
                reward += REWARD_PIPE
        self.pipes = [p for p in self.pipes if p[0] > -PIPE_W - 12.0]
        if not self.pipes or self.pipes[-1][0] <= W - PIPE_SPACING:
            self._spawn((self.pipes[-1][0] if self.pipes else W) + PIPE_SPACING)

        done = False
        if self.y - BIRD_HH <= 0.0:
            done, self.death_reason = True, "ceiling"
        elif self.y + BIRD_HH >= GROUND_Y:
            done, self.death_reason = True, "ground"
        else:
            for p in self.pipes:
                if self._hits(p):
                    done, self.death_reason = True, "pipe"
                    break
        if self.t >= MAX_STEPS:
            done, self.death_reason = True, "time limit"
        if done:
            self.alive = False
            reward += REWARD_DEATH
        # potential-based shaping: gamma*phi(s') - phi(s)
        new_phi = self._phi()
        reward += SHAPE_GAMMA * new_phi - self.phi
        self.phi = new_phi
        return self.obs(), reward, done

    def _phi(self) -> float:
        p = self.target_pipe()
        if p is None:
            return 0.0
        dy = abs(self.y - p[1])
        return -SHAPE_K * min(dy, SHAPE_CAP) / SHAPE_CAP

    def _hits(self, p: list) -> bool:
        if BIRD_X + BIRD_HW < p[0] or BIRD_X - BIRD_HW > p[0] + PIPE_W:
            return False
        top = p[1] - PIPE_GAP / 2
        bot = p[1] + PIPE_GAP / 2
        return self.y - BIRD_HH < top or self.y + BIRD_HH > bot

    # ------------------------------------------------------------- observation
    def target_pipe(self):
        for p in self.pipes:
            if p[0] + PIPE_W >= BIRD_X - BIRD_HW:
                return p
        return None

    def obs(self):
        return observe(self.pipes, self.y, self.vy)


def rollout(env: Flappy, policy, max_steps: int = MAX_STEPS, record: bool = False):
    """Run one greedy episode. Returns (pipes_passed, steps, frames|None)."""
    obs, frames = env.reset(), []
    done, steps = False, 0
    while not done and steps < max_steps:
        if record:
            frames.append({
                "y": env.y,
                "pipes": [(p[0], p[1]) for p in env.pipes],
                "passed": env.pipes_passed,
                "score_t": env.t,
            })
        obs, _, done = env.step(policy(obs))
        steps += 1
    if record:
        frames.append({
            "y": env.y,
            "pipes": [(p[0], p[1]) for p in env.pipes],
            "passed": env.pipes_passed,
            "score_t": env.t,
        })
    return env.pipes_passed, steps, (frames if record else None)
