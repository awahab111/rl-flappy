"""Print the frames that `flappy-q --trace N` prints, for a diff.

The Rust game is only useful if it is the same game. This script swaps the
random gap sampler for the same scripted sequence the Rust trace mode uses, so
both sides fly an identical level, and prints the same four columns.

    cargo run --release -- --trace 400 > rust_trace.txt
    python3 rust/python/parity_trace.py 400 > py_trace.txt
    diff rust_trace.txt py_trace.txt

Any difference means the port drifted from env.py.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import env as F  # noqa: E402

spawned = [0]


def scripted_gap(_rng) -> float:
    """The same 29.0 to 52.0 ladder the Rust side builds."""
    i = spawned[0]
    spawned[0] += 1
    return 29.0 + 23.0 * ((i * 37) % 100) / 100.0


# Flappy._spawn looks this name up at call time, so replacing it is enough.
F.sample_gap = scripted_gap


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    env = F.Flappy(seed=1)
    env.reset()
    for t in range(steps):
        env.step(1 if t % 7 == 0 else 0)
        print(f"{env.t} {env.y:.12f} {env.vy:.12f} {env.pipes_passed}")


if __name__ == "__main__":
    main()
