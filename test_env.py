"""Checks on the game rules. Run: python3 test_env.py

The order matters. Every test before the last one proves the game CAN punish the
bird: ground, ceiling, pipe body, a terminal reward that is strongly negative,
and a state that never leaves the table. Only after all of that does one test
confirm that a trained checkpoint clears pipes.

A game that cannot punish a bird will still pass a happy path test, which is why
that test comes last.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

import flappy as F
from agent import QAgent

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def drop_until_dead(env: F.Flappy, action: int = 0) -> tuple[int, str]:
    obs = env.obs()
    done, n = False, 0
    while not done and n < F.MAX_STEPS:
        obs, _, done = env.step(action)
        n += 1
    return n, env.death_reason


def main() -> None:
    # --- ways to die -------------------------------------------------------
    env = F.Flappy(seed=1)
    env.reset()
    _, reason = drop_until_dead(env, action=0)
    check("never flapping ends on the ground", reason == "ground", f"got {reason!r}")

    env = F.Flappy(seed=2)
    env.reset()
    _, reason = drop_until_dead(env, action=1)
    check("always flapping ends on the ceiling", reason == "ceiling", f"got {reason!r}")

    # flying straight into a pipe body must kill
    env = F.Flappy(seed=3)
    env.reset()
    pipe = env.target_pipe()
    env.y = pipe[1] - F.PIPE_GAP / 2 - F.BIRD_HH - 1.0     # just above the gap
    pipe[0] = F.BIRD_X - 1.0
    _, _, done = env.step(0)
    check("pipe body is lethal", done and env.death_reason == "pipe", f"got {env.death_reason!r}")

    # ...but the same height inside the gap must not be lethal
    env = F.Flappy(seed=4)
    env.reset()
    pipe = env.target_pipe()
    env.y = pipe[1]
    env.vy = 0.0
    pipe[0] = F.BIRD_X - 1.0
    _, _, done = env.step(0)
    check("the gap is not lethal", not done, f"died by {env.death_reason!r}")

    # --- reward signs ------------------------------------------------------
    env = F.Flappy(seed=5)
    env.reset()
    _, r_alive, done = env.step(0)
    check("surviving pays a small positive reward",
          (not done) and 0.0 < r_alive < F.REWARD_PIPE, f"r={r_alive}")

    env = F.Flappy(seed=5)
    env.reset()
    env.y = F.GROUND_Y - F.BIRD_HH - 0.1
    _, r_dead, done = env.step(0)
    check("dying is strongly negative", done and r_dead < -5.0, f"r={r_dead}")

    # --- observation hygiene ----------------------------------------------
    env = F.Flappy(seed=6)
    obs = env.reset()
    ok = True
    for _ in range(500):
        obs, _, done = env.step(1 if env.vy > 0 else 0)
        if not (0 <= obs[0] < F.N_DX and 0 <= obs[1] < F.N_DY
                and 0 <= obs[2] < F.N_VY and 0 <= obs[3] < F.N_Y):
            ok = False
            break
        if done:
            env.reset()
    check("observations stay inside the table bounds", ok, f"last obs {obs}")

    # --- agent -------------------------------------------------------------
    agent = QAgent(alpha=0.5, gamma=0.9)
    s, s2 = (0, 4, 1, 3), (1, 4, 1, 3)
    agent.update(s, 1, 1.0, s2, done=True)
    check("terminal update ignores the future", abs(agent.Q[s][1] - 0.5) < 1e-9, f"{agent.Q[s][1]}")
    agent.update(s, 1, 1.0, s2, done=False)
    expected = 0.5 + 0.5 * (1.0 + 0.9 * 0.0 - 0.5)
    check("bootstrap at gamma 0.9 for an empty next state",
          abs(agent.Q[s][1] - expected) < 1e-9, f"{agent.Q[s][1]}")

    # --- persistence round trip -------------------------------------------
    tmp = os.path.join(tempfile.mkdtemp(), "q.json")
    agent.save(tmp)
    back = QAgent.load(tmp)
    check("save/load is lossless", np.array_equal(agent.Q, back.Q))

    # --- happy path, once --------------------------------------------------
    trained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")
    if os.path.exists(trained_path):
        trained = QAgent.load(trained_path)
        env, scores = F.Flappy(seed=999), []
        for _ in range(3):
            obs, done = env.reset(), False
            while not done:
                obs, _, done = env.step(trained.greedy(obs))
            scores.append(env.pipes_passed)
        check("trained agent clears pipes on an unseen world", min(scores) >= 5, f"{scores}")
    else:
        print("  skip trained-agent check (run train.py first)")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
