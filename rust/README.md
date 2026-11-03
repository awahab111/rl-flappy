# The same agent, in Rust

Same game, same table, same update rule as the Python version in `../python/`. It has
no dependencies and no game engine, just three files.

Nothing here argues that Rust is the right language for Q-learning. It is faster, and
the useful part was finding out what that speed buys on a game this small.

## Build and run

```bash
cd rust
cargo build --release
./target/release/flappy-q --episodes 20000 --seed 19 --out q_table_rust.json
```

```
trained 20000 lives in 0.19s  (17.2 M steps/s, 3192079 updates)
cells touched 2024 of 3000   first 100 lives 0.00 pipes   last 100 7.20   best 57
greedy on 20 unseen levels: 71.0 pipes (20 hit the step cap)
wrote q_table_rust.json (59 KB)
```

Flags: `--episodes`, `--seed`, `--out`, `--eval-levels`, `--trace`.

The table it writes uses the same two line format `python/agent.py` reads, so the
Python renderers and `watch.py` can fly a Rust trained bird without changes.

## What is different

The physics is a line by line port. The learner is the same equation. The data
layout is the only real design change.

Python indexes a five dimensional numpy array with a tuple of four numbers. Rust
folds those four into one integer and keeps the table in a single `Vec<f64>`:

```
offset = ((dx * N_DY + dy) * N_VY + vy) * N_Y + y
```

One multiply and add chain, one pointer, and the whole table sits in a contiguous
48 KB block that stays in cache. Same 6,000 numbers, far fewer instructions to reach
one.

The random number generator is twelve lines of xorshift64* instead of the `rand`
crate. A crate would work fine, but a demo that claims no dependencies should have
none.

## How the two were checked

The first check is frame by frame. `--trace 400` flies a level with no randomness and
a fixed action pattern, printing height and speed every step, and
`python/parity_trace.py` prints the same four hundred lines from the Python game. The
two files are identical to twelve decimal places.

```bash
./target/release/flappy-q --trace 400 > rust_trace.txt
python3 ../python/parity_trace.py 400 > py_trace.txt
diff rust_trace.txt py_trace.txt
```

The second check is end to end. The two versions use different RNGs, so the same seed
gives different levels and the scores move around. To take that out of the
comparison, I ported the Rust xorshift into Python, flew the same table, and got the
same 20 level evaluation, level for level.

Without those two, the honest claim is "similar score". With them, the claim is that
the game is the same and any difference in score comes from the levels.

## Where the speed comes from

Measured on this machine, one core, release build:

| | Python | Rust |
|---|---|---|
| 20,000 lives | 42.0 s | 0.16 s |
| updates per second | 66,000 | 19,300,000 |
| table | numpy array | `Vec<f64>` |
| dependencies | numpy, Pillow | none |

Most of the gap is the interpreter. Every `env.step()` in Python is a method call, a
tuple build, and an array lookup, while Rust compiles the same step into a few dozen
instructions. Some of the rest is numpy, since `self.Q[state].max()` builds a view and
dispatches through numpy to compare two numbers, which costs more than the arithmetic
it performs. The last piece is layout, because a flat index into one contiguous
vector stays in cache.

None of that is a claim about language speed in general. When a step is twenty
floating point operations, almost all of the cost is the overhead around them.

## Seeds, and what the speed is for

Scores swing a lot between seeds in both versions. Across 100 seeds at 20,000 lives
each:

| statistic | pipes on 20 unseen levels |
|---|---|
| best seed | 71.0 (8 seeds reach it) |
| median | 60.5 |
| mean | 55.4 |
| worst | 4.8 |

The whole sweep takes 19 seconds in Rust and about 70 minutes in Python. That is the
difference that matters. One run tells you whether a seed was lucky. A hundred tell
you what the trainer usually does, and which table to keep.

`q_table_rust.json` is seed 19, the best of that sweep.

## If this were a real project

The table is 6,000 numbers, so none of this is needed here. For something bigger:

- `rurel` 0.6.0, last release June 2024, about 20,000 downloads. It is tabular Q
  learning in a crate. For a table this small, the crate is bigger than the problem.
- `candle` 0.11.1 and `burn` 0.23 for tensors, or `tch` 0.26 if you want libtorch.
  `dfdx` has shipped nothing since 2023. You need one of these the moment the table
  becomes a neural network, and not before.
- `pyo3` 0.30 plus `maturin` 1.16 to call Rust from Python. Crossing the boundary
  costs about 100 ns, against roughly 500 ns for a numpy dispatch, which is why
  projects like `rlox` move whole loops across instead of calling Rust per step.
- `rayon` 1.12 for parallel rollouts. Q-learning updates depend on each other, so
  this means Hogwild style atomics or a merge step, and both change what
  reproducible means.
- `criterion` 0.8 if you want to measure rather than trust a table in a README.

## Limits

One core. The eight cores on this box sat idle during training, because a single
episode is sequential and every episode writes to the same table.

The exploration schedule, the reward sizes and the bucket edges are copied from the
Python version rather than retuned. A port with its own tuning would probably reach
71 more often than 8 seeds in 100.

The score depends on the seed more than on the language, so any comparison between
the two has to control for the levels. That is what the two checks above are for.
