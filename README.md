# rl-flappy

A bird flies through pipes. A table of 6,000 numbers decides when the bird flaps.
There is no neural network. Training runs on one core in 12 seconds.

The project exists in two versions. `python/` is the readable one, and the videos come
from it. `rust/` holds the same game and the same learner, with no dependencies. It
runs about 70 times faster.

[![The flock in the final round](videos/preview.gif)](videos/rl_flappy_rounds_1080.mp4)

The preview plays on the page. Click it to open the full clip at 1080p. The 4K master
is `videos/rl_flappy_rounds_4k.mp4` (3840x2160, 7 MB). GitHub and some browsers refuse
that file. Download it and open it locally if the player stays black.

## Start here

You need Python 3.10 or newer, numpy, and Pillow. You need ffmpeg only for video.

```bash
cd python
pip install numpy pillow
```

Run these three commands in this order.

```bash
python3 test_env.py     # 2 seconds. Checks the game and the rules.
python3 train.py        # 12 seconds. Builds the table from zero.
python3 watch.py        # Shows the trained bird in your terminal.
```

Every Python command in this README runs inside `python/`. Every Rust command runs
inside `rust/`.

What you see:

1. `test_env.py` prints 11 lines that start with `ok`. The last line says
   `all checks passed`.
2. `train.py` prints a progress line every 2,000 lives. Then it prints a summary.
3. `watch.py` clears the screen and draws the game in ASCII at 30 frames per second.

If `watch.py` says a file is missing, run `train.py` first. It writes `q_table.json`
in the same folder.

## What each file does

| file | what it does | open it when |
|---|---|---|
| `python/env.py` | The game. Gravity, pipes, collisions, and the four numbers the bird sees. | You want to change the difficulty. |
| `python/agent.py` | The table of scores and the rule that improves it. | You want to change how fast it learns. |
| `python/train.py` | Runs 20,000 lives, saves checkpoints, measures the result, draws the curve. | You want more training or a new set of levels. |
| `python/watch.py` | Plays a saved table in the terminal, in ASCII. | You want to check a checkpoint in two seconds. |
| `python/render.py` | Four checkpoints side by side, as video. | You want the short comparison clip. |
| `python/swarm.py` | 100 checkpoints on one level, with a stats panel. | You want the wide overview clip. |
| `python/rounds.py` | One checkpoint per round, ending at pipe 50. | You want the 4K clip. |
| `python/draw.py` | Drawing helpers and font loading. | You want different colours or type. |
| `python/test_env.py` | 11 checks on the rules of the game. | After you edit `env.py`. |
| `python/parity_trace.py` | Prints the same 400 frames as the Rust game. | After you edit the physics on either side. |
| `rust/src/env.rs` | The same game, with the same constants. | Same reasons as `env.py`. |
| `rust/src/agent.rs` | The same table, in one flat `Vec<f64>`. | You want to see the index arithmetic. |
| `rust/src/main.rs` | Training loop, evaluation, and the JSON writer. | You want the speed comparison. |

The data moves one way:

```
env.py  ->  train.py  ->  q_table.json  ->  watch.py
               |                        ->  render.py, swarm.py, rounds.py
               ->  stats.json, learning_curve.png, snapshots/
```

## How training works

`python3 train.py` does six steps.

1. It builds the game and an empty table of 6,000 zeros.
2. It runs 20,000 lives. A life ends when the bird dies or reaches the step cap.
3. It starts each life on a new level from the seed.
4. It learns from every step. The next section shows the rule.
5. It saves a checkpoint every N lives if you ask for one.
6. It stops exploring after 8,000 lives. The other 12,000 lives only refine scores.

The trainer calls one life an episode. The output uses both words.

### What the state is

The bird sees four numbers. Each number falls into a bucket.

| number | buckets | what it measures |
|---|---|---|
| distance | 6 | space left before the next pipe |
| offset | 10 | how far the bird sits above or below the middle of the gap |
| speed | 5 | rising, level, or falling, and how fast |
| height | 10 | how high the bird is on the screen |

Six times ten times five times ten is 3,000 states. Each state has two scores, one for
each action. The bird chooses the action with the higher score. That is the policy.

The height looks redundant next to the offset. It is not. Without height, a bird that
is safe in a low gap and a bird one pixel from the ceiling share a bucket. The bird
then flies into a wall that it cannot see.

### What one step does to the table

Each step runs five moves in `update`, inside `agent.py`.

1. Read the state. Four numbers give one row of the table.
2. Choose an action. Read the score for that action in the row.
3. Read the reward. 0.1 for survival, 5 for a pipe, minus 10 for death.
4. Build the target. Add the reward to 0.95 times the best score in the next state.
5. Move the score a fraction toward the target. The fraction is 0.20.

```
target = reward + 0.95 * best_score_of_next_state
score  = score + 0.20 * (target - score)
```

Here is a real update on a fresh table. The bird is in state `(5, 1, 2, 4)`. It flaps
and stays alive. The next state still holds zero.

```
target = 0.10 + 0.95 * 0.00 = 0.10
score  = 0.00 + 0.20 * (0.10 - 0.00) = 0.02
```

The training run printed the same row after four visits:

```
0.000  ->  0.020  ->  0.036  ->  0.052
```

Each visit adds less than the visit before. The target moves as the score catches up.
The table improves one row at a time.

The death step teaches caution. It pays 0.1 minus 10. There is no next state.

```
target = -9.90
score  = 0.00 + 0.20 * (-9.90 - 0.00) = -1.98
```

The row now holds a large negative number. The bird stops choosing it. The worst score
in the shipped table is -9.85. That row sits one step from the ground.

### What the table looks like after training

These rows come from `q_table.json`. The four numbers are distance, offset, speed, and
height.

| row | coast | flap | bird presses |
|---|---|---|---|
| `(5, 4, 0, 3)` | +2.31 | +2.39 | flap |
| `(5, 4, 1, 4)` | +2.19 | +0.71 | coast |
| `(2, 5, 3, 5)` | 0.00 | +3.50 | flap |
| `(0, 4, 2, 4)` | +5.41 | +2.68 | coast |

A score is the sum of future reward, reduced by distance. Survival alone reaches about
2.0. One step pays 0.1. Each step of delay multiplies by 0.95. A score above 2.0 came
from pipes. A negative score means the bird expects to die soon.

This run touched 1,061 of the 3,000 rows. 740 rows tell the bird to coast. 319 rows
tell it to flap. 1,941 rows stay at zero because the bird never visited them. A zero
row gives both actions the same score. The tie break in `greedy` chooses flap.

### Why the curve is flat, then steep

The table can learn about a state only after the bird reaches it. For the first 6,000
lives the bird cannot clear the first gap. Every row after that gap stays at zero.
Nothing downstream can improve.

After 6,000 lives the bird clears the first gap most of the time. The rows after it
start to fill. The bird can then clear the second gap. The same effect repeats. Almost
all of the improvement lands between 7,000 and 10,000 lives. The change is sudden, not
gradual.

### How much work that is

The run makes 2.78 million updates over 20,000 lives. A life averages 139 steps. One
core takes 12 seconds. About a thousand of the 6,000 numbers do the work.

Exploration gets the bird to the first gap. For the first 8,000 lives it chooses a
random action often. That is how it finds that flying up is possible. After that the
rate drops to 2 percent, one random action in fifty. The same 2 percent keeps the
training average low. One random flap ends a long life.

### Why those reward sizes

Survival pays 0.1. A pipe pays 5, which equals 50 steps of hovering. Death costs 10,
which equals 100 steps. The sizes matter. A fresh table starts at zero. If death paid
anything at all, crashing would look better than an untried action. It did, for 40,000
lives, until the death penalty arrived.

### What training prints

```
  ep   2000/20000  eps=0.755  mean pipes (last 2000) =   0.00  best =   0    0.1s
  ep   4000/20000  eps=0.510  mean pipes (last 2000) =   0.00  best =   0    0.2s
  ep   6000/20000  eps=0.265  mean pipes (last 2000) =   0.00  best =   0    0.4s
  ep   8000/20000  eps=0.020  mean pipes (last 2000) =   0.50  best =  16    0.8s
  ...
  ep  20000/20000  eps=0.020  mean pipes (last 2000) =   5.65  best =  67   11.1s

first 100 episodes : 0.00 pipes/run
last  100 episodes : 6.80 pipes/run  (with 2% random actions)
best single run    : 67 pipes
greedy eval        : 71.00 pipes/run over 20 unseen worlds (min 71, max 71)
long horizon       : [241, 241, 241, 241, 241] pipes (cap 8000 steps)
state space        : 3000 states, 6000 Q values, 1061 states ever visited
train time         : 11.4s
```

The progress line shows four things. `eps` is the share of random actions that
remains. `mean pipes` is the average, including random actions. `best` is the best
single life. The last number is elapsed time. Do not worry when `mean pipes` stays at
0.00 for a few thousand lives. That is normal.

The summary matters more. `last 100 episodes` stays low because of the 2 percent. The
`greedy eval` line is the real score. It uses no random action, on 20 levels that the
bird never saw.

### The knobs

| flag | default | what it does |
|---|---|---|
| `--episodes` | 20000 | How many lives to run. 20,000 is enough. 5,000 is not. |
| `--seed` | 23 | Changes the level sequence. The same seed gives the same result. |
| `--alpha` | 0.20 | How far each score moves per step. A higher value learns faster and jitters more. |
| `--gamma` | 0.95 | How much the future counts. A lower value makes the bird short sighted. |
| `--snap-every` | 0 | Save a checkpoint every N lives. The video scripts need this. |
| `--snap-marks` | 0.40,0.70 | Extra checkpoints at set lives, or as a fraction. |
| `--out` | script folder | Where to write the four output files. |

Training uses one core. More cores do not help, because one bird flies at a time.

## What training writes

| file | size | what it is |
|---|---|---|
| `q_table.json` | 65 KB | The table. 6,000 numbers as nested lists. This file is the agent. |
| `snapshots/ep_XXXXX.json` | 65 KB each | The same table, saved part way through the run. Only written with `--snap-every`. |
| `stats.json` | 230 KB | Every number from the run, plus the history of each life. |
| `learning_curve.png` | 25 KB | The picture. A thin line per life, a thick average line. |

The table is plain JSON on purpose. You can open it, compare two files, or copy a
checkpoint. There is no model file and no build step.

## Using a trained table

`watch.py` plays a table in the terminal. By default it reads `q_table.json` from the
folder of the script.

```bash
python3 watch.py                                # the trained table, until it dies
python3 watch.py --seed 99                      # a level it has never seen
python3 watch.py --steps 200                    # stop after 200 frames
python3 watch.py --fps 60                       # faster playback
python3 watch.py --q snapshots/ep_08000.json    # a checkpoint, see below
```

The last command is the useful one. It needs checkpoints on disk. Run `train.py
--snap-every 100` once (12 seconds). Then point `--q` at `ep_00000`, `ep_08000`,
`ep_12000`, and `ep_20000`. You then see the bird learn in about ten seconds, with no
video render.

To use a table in your own script:

```python
from agent import QAgent
from env import Flappy

agent = QAgent.load("q_table.json")
env = Flappy(seed=7)
state = env.reset()
while True:
    state, reward, done = env.step(agent.greedy(state))
    if done:
        break
print(env.pipes_passed, env.death_reason)
```

`agent.greedy(state)` never guesses. `agent.act(state, epsilon)` guesses. Use it in
training only.

## Making the videos

The three video scripts replay checkpoints. Train with checkpoints first.

```bash
python3 train.py --snap-every 100        # 201 checkpoints, one every 100 lives
```

A render takes 15 seconds to 4 minutes. Every script has `--dry-run`. It prints the
result of each run and the frame count, and it does not encode. Use it first.

| command | output | time |
|---|---|---|
| `python3 render.py --episodes 0,7500,8000,20000` | four panels, 720x540, 23s | 15s |
| `python3 swarm.py --birds 100 --pick even` | 100 birds, one level, 29s | 30s |
| `python3 rounds.py` | round robin, 4K at 60fps, 37s | 3m40s |
| `python3 rounds.py --height 1080` | the same clip at 1080p, for drafts | 90s |

`rounds.py` takes useful flags. `--target 50` sets the pipe that ends the video.
`--group 12` sets the birds per round. `--schedule 0,2000,8000,10000` sets the
checkpoints. `--seed` sets the level.

The 4K file is heavy, and some players refuse it. `videos/rl_flappy_rounds_1080.mp4`
is the same run at 1080p, and a browser can play that one. A fresh render with
`--height 1080` gives the small version in about 90 seconds.

`swarm.py` needs at least as many checkpoints as birds. `rounds.py` needs the
checkpoints in its schedule. The default schedule is every 2,000 lives up to 12,000.

## The same agent in Rust

`rust/` holds the same game and the same learner. It has no dependencies. It trains
the same 20,000 lives in 0.16 seconds, not 12. It writes a table that the Python tools
load without changes.

| | Python | Rust |
|---|---|---|
| 20,000 lives | 11.7s | 0.16s |
| updates per second | 238,000 | 19,300,000 |
| peak memory | 46 MB | 1.9 MB |
| table | flat list | `Vec<f64>` |
| dependencies | numpy, Pillow | none |

Both rows come from `/usr/bin/time` on the whole program. The Python memory includes
the interpreter and its libraries. The Rust memory is the process and nothing else.

Part of that gap used to belong to numpy. The Python agent kept its table in a five
dimensional array, and numpy spends more time building views than comparing two floats.
In one harness, with the same seed and the same 20,000 lives, numpy took 39.1 seconds
and a flat list took 12.1, for the same table bit for bit and the same 32 MB. The
Python version now uses the list. What is left, 11.7 seconds against 0.16, is the
language.

The Rust game is the same game. `--trace 400` prints 400 frames.
`python/parity_trace.py` prints the same 400 frames from the Python side. The two
files match to twelve decimal places.

The scores still differ, and the levels cause that. Each version has its own random
generator. The same seed builds different pipes. Fly the same table through the same
levels, and the results match level for level.

That is the argument for the port. Across 100 seeds, the Rust trainer reaches 71 pipes
on 8 seeds. The median is 60.5. The sweep takes 19 seconds. In Python it would take 20
minutes. One run tells you if a seed was lucky. A hundred runs tell you what the
trainer usually does. `rust/README.md` has the design notes and the crate choices for
a larger problem.

## Changing the game

The difficulty lives in `python/env.py`, as constants near the top.

| constant | now | what happens if you change it |
|---|---|---|
| `PIPE_GAP` | 30 | Wider gaps make the game easier. |
| `PIPE_SPACING` | 56 | More space means fewer pipes per second. |
| `PIPE_SPEED` | 1.7 | Faster pipes give less time to correct. |
| `GRAVITY` | 0.40 | Heavier gravity makes every flap matter more. |
| `FLAP_V` | -3.2 | A stronger flap overshoots narrow gaps. |

Retrain after you change a constant. An old table still loads, but it expects the old
physics.

The bucket edges in the same file set the table size. A change to the edges changes
the table shape. Old checkpoints then fail to load. The reward sizes live there too.
They set what the bird cares about.

### More buckets, or more training?

Both, but buckets cost more than most people expect. These numbers are measured, not
estimated. The seed and the 20 test levels stay the same. Only the bucket counts
change.

| buckets (distance, offset, speed, height) | cells | updates per visited cell | pipes per run | train time |
|---|---|---|---|---|
| 6, 10, 5, 10 | 3,000 | 927 | 71.0, every run at the cap | 12s |
| 6, 19, 5, 19 | 10,830 | 200 | 32.7 | 11s |
| 6, 37, 5, 37 | 41,070 | 37 | 7.1 | 9s |

At a fixed budget, more buckets make the bird much worse. The middle column explains
why. A cell that sees 37 updates never settles on a number. The bird then acts on
noise.

More training can pay for the buckets, up to a point. The 10,830 cell table trained
for 100,000 lives reaches 71.0 again. It takes 86s, not 12s. It catches up but does
not overtake, because the small table already flies to the step cap.

Here is the same comparison past the cap, on eight new levels with a 60,000 step
limit.

| table | cells | train time | mean pipes | runs that hit the cap |
|---|---|---|---|---|
| small | 3,000 | 12s | 1,707.6 | 7 of 8 |
| fine | 10,830 | 86s | 1,722.8 | 7 of 8 |

That gap is noise. It cost seven times the compute.

A table does not generalise. Every cell learns alone. Bucket 4 and bucket 5 are
strangers, although they are one pixel apart. More buckets do not add knowledge. They
split the same experience into more drawers.

Resolution pays when the task is hard enough that the small table fails. Narrow the
gaps or speed up the pipes. The fine table then has room to be better. The other fix
is to share learning between cells. Tile coding and function approximation do that.
One update then improves several nearby cells.

## Files in the repo

```
python/
  env.py            the game
  agent.py          the table and the learning rule
  train.py          training, evaluation, and the files it writes
  watch.py          terminal playback
  render.py         four panel video
  swarm.py          one hundred bird video
  rounds.py         round robin 4K video
  draw.py           drawing helpers and fonts
  test_env.py       11 checks on the game rules
  parity_trace.py   cross-check against the Rust game
  q_table.json      the trained table
  stats.json        every number in this README
  learning_curve.png  the picture
  assets/fonts/     Space Grotesk, used by the renderers
rust/
  src/env.rs        the same game, same constants
  src/agent.rs      the same table, flat Vec<f64>
  src/main.rs       training loop, evaluation, JSON writer
  Cargo.toml        no dependencies
  q_table_rust.json seed 19, best of a 100 seed sweep
videos/             three clips, a 1080p copy, and a preview GIF
```

`python/snapshots/` is not in the repo. It holds 201 checkpoints, which is 12 MB of
generated JSON. `python3 train.py --snap-every 100` rebuilds it exactly. Training uses
a fixed seed, so it produces the same table every time.
