# The same agent, in Rust

`rust/` holds the same game, the same table, and the same update rule as `python/`. It
has no dependencies and no game engine. It has three files.

This port does not claim that Rust is better for Q-learning. Rust is faster here, and
the useful part was to measure what the speed buys on a small game.

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

The table uses the same two line format that `python/agent.py` reads. The Python
renderers and `watch.py` can play a Rust table without changes.

## What is different

The physics is a line by line port. The learner uses the same equation. The data
layout is the only design change.

Python indexes a five dimensional numpy array with a tuple of four numbers. Rust folds
the four numbers into one integer. The table is a single `Vec<f64>`.

```
offset = ((dx * N_DY + dy) * N_VY + vy) * N_Y + y
```

The path to one number uses one multiply and add chain and one pointer. The table is a
48 KB block that stays in cache. The numbers are the same 6,000 numbers.

The random number generator is twelve lines of xorshift64*. It replaces the `rand`
crate. A crate would also work. But a demo that claims no dependencies should have
none.

## How the two were checked

The first check compares frames. `--trace 400` flies a level with no random numbers
and a fixed action pattern. It prints height and speed at each step.
`python/parity_trace.py` prints the same 400 lines from the Python game. The two files
match to twelve decimal places.

```bash
./target/release/flappy-q --trace 400 > rust_trace.txt
python3 ../python/parity_trace.py 400 > py_trace.txt
diff rust_trace.txt py_trace.txt
```

The second check compares a full run. The two versions use different random
generators. The same seed gives different levels, and the scores move. I removed that
difference. I ported the Rust generator to Python and flew the same table. The 20 level
evaluation then matched Rust level for level.

Without the second check, the honest claim is "similar score". With it, the claim is
that the game is the same. A difference in score then comes from the levels.

## Where the speed comes from

Measured on this machine, one core, release build:

| | Python | Rust |
|---|---|---|
| 20,000 lives | 42.0 s | 0.16 s |
| updates per second | 66,000 | 19,300,000 |
| table | numpy array | `Vec<f64>` |
| dependencies | numpy, Pillow | none |

Most of the gap comes from the interpreter. One `env.step()` in Python is a method
call, a tuple build, and an array lookup. Rust compiles the same step into a few dozen
instructions. Part of the rest comes from numpy. `self.Q[state].max()` builds a view
and dispatches through numpy to compare two numbers. That costs more than the
arithmetic. The last part is layout. A flat index into one contiguous vector stays in
cache.

This is not a general claim about language speed. One step holds twenty floating point
operations. Almost all of the cost sits in the overhead around them.

## Seeds, and what the speed is for

Scores vary a lot between seeds in both versions. Here are 100 seeds at 20,000 lives
each.

| statistic | pipes on 20 unseen levels |
|---|---|
| best seed | 71.0 (8 seeds reach it) |
| median | 60.5 |
| mean | 55.4 |
| worst | 4.8 |

The whole sweep takes 19 seconds in Rust and about 70 minutes in Python. That
difference matters. One run tells you if a seed was lucky. A hundred runs tell you what
the trainer usually does. `q_table_rust.json` is seed 19, the best of that sweep.

## If this were a real project

The table holds 6,000 numbers, so none of this is needed here. A larger problem needs
other tools.

- `rurel` 0.6.0 (last release June 2024, about 20,000 downloads) puts tabular Q
  learning in a crate. For a table this small, the crate is bigger than the problem.
- `candle` 0.11.1 and `burn` 0.23 handle tensors. `tch` 0.26 binds libtorch. `dfdx`
  shipped nothing since 2023. You need one of these when the table becomes a neural
  network.
- `pyo3` 0.30 and `maturin` 1.16 call Rust from Python. One call across the boundary
  costs about 100 ns. A numpy dispatch costs about 500 ns. Projects like `rlox` move
  whole loops into Rust for that reason.
- `rayon` 1.12 runs episodes in parallel. Q-learning updates depend on each other.
  Parallel work then needs Hogwild atomics or a merge step. Both change what
  reproducible means.
- `criterion` 0.8 measures code. Use it when you want numbers, not a claim in a README.

## Limits

One core. The eight cores on this box sat idle during training. One episode is
sequential, and every episode writes to the same table.

The exploration schedule, the reward sizes, and the bucket edges come from the Python
version. A port with its own tuning would probably reach 71 on more than 8 seeds in
100.

The seed matters more than the language. A comparison between the two versions must
control for the levels. The two checks above do that.
