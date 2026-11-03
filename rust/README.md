# The same agent, in Rust

Same game, same table, same update rule as the Python version in `../python/`,
ported without a game engine, a tensor library, or a single crate. Three files, no
dependencies.

The point is not that Rust is a better language for Q-learning. The point is
what the speed buys, and what porting it teaches about where the time goes.

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

The table it writes is the same two line format `python/agent.py` reads, so the
Python renderers and `watch.py` can fly a Rust trained bird without changes.

## What is different from the Python version

The physics is a line by line port. The learner is the same equation. The data
layout is the only real design change.

Python indexes a five dimensional numpy array with a tuple of four numbers.
Rust folds those four numbers into one integer and keeps the table in a single
`Vec<f64>`:

```
offset = ((dx * N_DY + dy) * N_VY + vy) * N_Y + y
```

One multiply and add chain, one pointer, no bounds checked tuple indexing, and
the table sits in one contiguous 48 KB block that stays in cache. Same 6,000
numbers. Different number of instructions to reach one.

The random number generator is twelve lines of xorshift64* instead of `rand`.
Not because a crate is bad, but because a demo that claims no dependencies
should have none.

## What is the same, and how that was checked

Two tests, both runnable:

1. **Physics, frame by frame.** `--trace 400` flies a level with no randomness
   and a fixed action pattern, printing height and speed every step.
   `python/parity_trace.py` prints the same four hundred lines from the Python
   game. The files are identical to twelve decimal places.

   ```bash
   ./target/release/flappy-q --trace 400 > rust_trace.txt
   python3 ../python/parity_trace.py 400 > py_trace.txt
   diff rust_trace.txt py_trace.txt
   ```

2. **End to end, same level.** The two versions use different RNGs, so the same
   seed gives different levels and the scores move. To remove that from the
   comparison, port the Rust xorshift into Python, fly the same table, and the
   20 level evaluation matches Rust exactly, level for level.

Without test 2 the honest claim is "similar score". With it the claim is "the
same game, and any difference in score comes from the levels, not the code".

## Where the speed comes from

Measured on this machine, one core, release build:

| | Python | Rust |
|---|---|---|
| 20,000 lives | 42.0 s | 0.16 s |
| updates per second | 66,000 | 19,300,000 |
| table | numpy array | `Vec<f64>` |
| dependencies | numpy, Pillow | none |

Three reasons, in order of size:

1. **The interpreter.** Every `env.step()` in Python is a method call, a tuple
   build, and a dict or array lookup. Rust compiles the whole step into a few
   dozen instructions.
2. **numpy per call overhead.** `self.Q[state].max()` allocates a view and
   dispatches through numpy for two numbers. That costs more than the arithmetic
   it performs.
3. **Layout.** A flat index into a contiguous vector is friendlier to the cache
   than a multi dimensional array lookup.

The 250 times gap is not a language benchmark. It is what happens when the work
per step is 20 floating point operations and the per step overhead is a
function call in a slow language.

## Seeds, and what the speed is actually for

Scores vary a lot between seeds, in both versions. Across 100 seeds at 20,000
lives each:

| statistic | pipes on 20 unseen levels |
|---|---|
| best seed | 71.0 (8 seeds reach it) |
| median | 60.5 |
| mean | 55.4 |
| worst | 4.8 |

The whole sweep takes 19 seconds in Rust. In Python it would take about 70
minutes. That is the practical difference. With one run you tune a seed. With a
hundred runs you measure a distribution and ship the best one.

`q_table_rust.json` is seed 19, the best of that sweep.

## If this were a real project

The table is 6,000 numbers, so none of this is needed here. For a bigger
problem:

- **Tabular, still.** `rurel` (0.6.0, last release June 2024, about 20,000
  downloads) gives you the same Q learning inside a crate. Useful for the traits,
  not for the speed. The table here is smaller than the crate.
- **Neural network.** `candle` (0.11, about 11 million downloads) and `burn`
  (0.23) are the live options for tensors. `tch` binds libtorch, which is not
  native Rust but is the shortest path from PyTorch. `dfdx` has not shipped since
  2023.
- **Parallel rollouts.** `rayon` over episodes. Q-learning updates depend on
  each other, so this means either Hogwild style atomic updates or a merge step,
  and both change what "reproducible" means.
- **Calling it from Python.** `pyo3` plus `maturin`. The boundary costs about
  100 ns per call, which is why projects like `rlox` move whole loops into Rust
  instead of calling Rust once per step.
- **Benchmarks.** `criterion`, if you want to measure rather than trust a table
  in a README.

## Limits

- One core. The 8 core box here is idle during training, because a single
  episode is inherently sequential and the table is shared.
- The exploration schedule, the reward sizes and the bucket edges are copied
  from the Python version, not retuned. A Rust port with its own tuning would
  probably reach 71 more often than 8 seeds in 100.
- The score depends on the seed more than on the language. Any comparison
  between the two implementations has to control for the levels, which is what
  the two tests above are for.
