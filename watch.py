"""Fly the trained bird in a terminal, using ASCII instead of sprites.

Same physics, same table, no ffmpeg. It is the fastest way to check that a
checkpoint actually flies before spending three minutes rendering it in 4K.

    python3 watch.py            # animated, 30 fps
    python3 watch.py --steps 3  # just show a few frames
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import flappy as F
from agent import QAgent

COLS, ROWS = 60, 30
HERE = os.path.dirname(os.path.abspath(__file__))


def frame(env: F.Flappy) -> list[str]:
    grid = [[" "] * COLS for _ in range(ROWS)]

    def put(x: int, y: int, ch: str) -> None:
        if 0 <= x < COLS and 0 <= y < ROWS:
            grid[y][x] = ch

    ground_row = int(F.GROUND_Y / F.H * ROWS)
    for x in range(COLS):
        put(x, ground_row, "_")
    for px, gap, _counted in env.pipes:
        left, right = int(px / F.W * COLS), int((px + F.PIPE_W) / F.W * COLS)
        top = int((gap - F.PIPE_GAP / 2) / F.H * ROWS)
        bot = int((gap + F.PIPE_GAP / 2) / F.H * ROWS)
        for x in range(left, right + 1):
            for y in range(0, top):
                put(x, y, "|")
            for y in range(bot, ground_row):
                put(x, y, "|")
        for x in range(max(0, left - 1), min(COLS, right + 2)):
            put(x, top - 1, "=")
            put(x, bot, "=")
    by = int(env.y / F.H * ROWS)
    put(int(F.BIRD_X / F.W * COLS), by, "@")
    put(int(F.BIRD_X / F.W * COLS) - 1, by - 1, ">")
    return ["".join(row) for row in grid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=os.path.join(HERE, "q_table.json"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=0, help="0 = fly until it dies")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    agent = QAgent.load(args.q)
    env = F.Flappy(seed=args.seed)
    obs, done, n = env.reset(), False, 0
    sys.stdout.write("\033[?25l")
    try:
        while not done and (args.steps == 0 or n < args.steps):
            obs, _, done = env.step(agent.greedy(obs))
            n += 1
            out = ["\033[H" + line + "\033[K" for line in frame(env)]
            out.append(f"\033[K  pipes cleared: {env.pipes_passed}   step {env.t}\033[K")
            sys.stdout.write("\n".join(out) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")
    print(f"\n  done: {env.pipes_passed} pipes, {env.t} steps, ended by {env.death_reason or 'your step limit'}")


if __name__ == "__main__":
    main()
