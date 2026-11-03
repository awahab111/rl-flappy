# rl-flappy

A flappy-bird-shaped toy where a **table of 6,000 numbers** learns to fly. No neural
nets, no gym, no PyTorch. numpy does the maths, Pillow draws the frames, ffmpeg
encodes them.

The point is to make the learning process watchable. Three videos in `videos/` show
the same agent at different stages: four checkpoints side by side, a hundred
checkpoints on one level, and a round-robin where each checkpoint gets a fresh group
of birds until one of them clears 50 pipes.

Measured on one core: it trains in **45 seconds**, then clears **71 pipes per run on
20 worlds it never saw** (every one hitting the step cap), and **241 pipes** on a
long horizon.

![learning curve](learning_curve.png)

The curve is flat for about 6,000 episodes, then falls off a cliff between 7,000 and
10,000: the checkpoint at 7,500 clears one pipe, the one at 8,000 clears seven, and
by 10,000 it never dies. Everything interesting in the videos happens inside that
window, which is why the checkpoint schedule is dense there.

## Results

Seed 23, 20,000 episodes, 45.1 seconds of one CPU core:

| | pipes per run |
|---|---|
| episodes 1 to 100 | 0.00 |
| last 100 episodes (with 2% random actions still on) | 6.80 |
| greedy, 20 worlds it never saw | **71.00** (every run hit the step cap) |
| greedy, long horizon, 8,000 step cap | 241 on all 5 worlds |
| best single training run | 67 |
| state-action values ever touched | 1,061 of 6,000 |

## Quickstart

```bash
python3 train.py                      # 20k episodes, writes q_table.json + stats + curve
python3 watch.py                      # fly the trained agent in your terminal
python3 watch.py --seed 99 --fps 60   # try a different world
python3 test_env.py                   # 11 failure-first checks
python3 rounds.py --height 1080       # render a video (1080p draft, ~90s)
```

## What the bird sees

Four small integers per frame. That is the whole observation.

| dimension | bins | bucket edges | meaning |
|---|---|---|---|
| `dx` | 6 | 8, 16, 26, 38, 52 | distance to the next pipe |
| `dy` | 10 | -18, -12, -7, -3, 0, 3, 7, 12, 18 | height offset from that pipe's gap centre |
| `vy` | 5 | -2.4, -0.8, 0.8, 2.4 | vertical speed |
| `y` | 10 | 10, 18, 26, 34, 42, 50, 58, 66, 74 | absolute height |

6 x 10 x 5 x 10 = 3,000 states, two actions (coast or flap), so 6,000 Q values. The
agent indexes a numpy array with that tuple and picks the larger of two numbers.

## The videos

Three cuts ship in `videos/`:

- `rl_flappy_rounds_4k.mp4`: one checkpoint per round, 3840x2160 at 60 fps, ends the
  moment a bird clears the 50th pipe (37s)
- `rl_flappy_swarm.mp4`: 100 checkpoints on one level, with the analytics panel
  (1720x900, 29s)
- `rl_flappy_learning.mp4`: four checkpoints side by side, each panel freezing when
  its bird dies (720x540, 23s)

`render.py` puts four checkpoints on one level, side by side, and freezes each panel
when its bird dies. `swarm.py` drops **100 checkpoints on a single level**, one bird
per checkpoint, every 200 episodes from 0 to 19,800, coloured red (untrained)
through yellow to green (trained).

The swarm panel is 1720x900 and carries five live readouts: the alive and crashed
counts, a top-of-the-flock leaderboard (episode count and pipes for the five birds
furthest through), one bar per bird showing pipes cleared so far (bars freeze grey
where a bird died, so the chart is also a death record), a histogram of when the
crashes happened, and a step timeline under the ground. In the field itself every
crash leaves a small cross where the bird went down, so the wall of X marks tracks
the learning gap.

On that swarm level, 750 steps long:

| | birds |
|---|---|
| never cleared a pipe | 40 |
| cleared 1 to 5 | 11 |
| cleared 6 to 20 | 15 |
| still flying when the tape stops | 34 |

All 34 survivors had trained for at least 7,600 episodes. One bird trained for
18,600 still crashed early on this level, because a 3,000 state table cannot see
every gap position exactly. That is worth knowing before believing the 71 pipe
average: the policy is very good on average and not perfect per level.

### Round robin (`rounds.py`)

One checkpoint per round. A group of 12 birds from that checkpoint starts, the next
round begins only when the last one crashes, and the run advances to the next
checkpoint. The video stops the moment a single bird clears the 50th pipe. The play
field fills the frame height with the labels living in the side margins, so nothing
sits on top of the game. Long runs switch to 6x playback after 3 pipes and say so.

On level 31337 that plays out as:

| round | checkpoint | result |
|---|---|---|
| 1 to 4 | 0 to 6,000 episodes | wiped out before the first pipe |
| 5 | 8,000 | 31 pipes |
| 6 | 9,000 | 36 pipes |
| 7 | 10,000 | **one bird cleared 50**, video ends |

```bash
python3 rounds.py                       # 4K 60fps master, 37s (about 3m40s to render)
python3 rounds.py --height 1080         # same layout at 1080p, for quick drafts
python3 rounds.py --dry-run             # round outcomes and frame count, no encoding
python3 rounds.py --target 80 --schedule 0,5000,8000,9000,10000,12000
```

Output is 3840x2160 at 60 fps, about 7 MB at crf 23. The layout is designed in a
900-pixel-tall space and multiplied by height, so 1080p and 4K are the same
composition: 4:3 play field at full height, centred, with 16:9 dark margins holding
the checkpoint on the left and the flock counter on the right. Type is Space Grotesk
(`assets/fonts/`).

```bash
python3 train.py --snap-every 100        # 201 checkpoints, one every 100 episodes
python3 render.py --episodes 0,7500,8000,20000 --steps 750
python3 swarm.py --birds 100 --seed 4242 --steps 750 --pick even
```

`--pick even` spaces the birds evenly across the training run. `--pick score` picks
one bird per distinct survival length instead, which front-loads fewer instant
deaths. `--dry-run` scores and picks without rendering.

## How it was built

### Build order

1. **World first, with probes.** Before any learning I scripted three rollouts: never
   flap, always flap, random. Each had to die for a different, explained reason
   (ground, ceiling, pipe). Random play never reached the first pipe on any seed,
   which told me the discovery problem was real and the reward had to carry the
   learning.
2. **Agent second.** About forty lines: `act(state, epsilon)`, `greedy(state)`,
   `update(...)`, `save`/`load`. Keep `greedy` separate from `act`, because what you
   evaluate and what you put in a video must not include exploration.
3. **Training with instrumentation.** Score and step count per episode, a progress
   line every 10%, a checkpoint every N episodes, and a `stats.json` written at the
   end. That file is where every number in this README comes from.
4. **Honest evaluation.** Greedy, on world seeds the agent never trained on,
   reported separately from the training mean.
5. **Videos last**, built from the saved checkpoints rather than the live agent, so
   the same level can be replayed for every checkpoint.

### The world

A 120 by 90 pixel field, no rendering library involved in the physics.

| constant | value |
|---|---|
| gravity | 0.40 px per step squared |
| flap impulse | -3.2 px per step |
| terminal fall speed | 5.5 px per step |
| pipe speed | 1.7 px per step |
| gap height | 30 px (bird hitbox is 5.2 px tall) |
| pipe spacing | 56 px, about 33 steps apart |
| bird x | fixed at 26, the world scrolls |
| step cap | 2,400 (71 pipes at 1.7 px per step) |

Levels come from one generator, `sample_gap(rng)`, drawing the gap centre uniformly
between 29 and 52 px. Gap generation happens on a fixed step schedule that no bird
can influence, which is what makes a level identical for every agent in the videos.

### The agent

Tabular Q-learning. The update is two lines:

```python
target = reward if done else reward + gamma * Q[next_state].max()
Q[state][action] += alpha * (target - Q[state][action])
```

| hyperparameter | value |
|---|---|
| alpha | 0.20 |
| gamma | 0.95 |
| epsilon | 1.00 to 0.02, linear over the first 40% of training (8,000 episodes) |
| episodes | 20,000 |
| rewards | `+0.1` per surviving step, `+5` per pipe, `-10` on death |

The table is 6,000 floats saved as JSON (65 KB). There is no model to ship, which is
why a checkpoint is just a file and a video can mix 201 of them in one scene.

### The three bugs that decided the outcome

Everything else was tuning. These three changed the result.

### 1. The reward taught the agent to die

The first version paid `+1` per surviving step, `+2` per pipe, and nothing for dying.
It sat at 0 to 1 pipes for 40,000 episodes with best runs of zero. The trap is that
the dying step also pays the per-step reward, so a fresh all-zero table learns within
a few hundred episodes that crashing is worth `+1` while an untried action is worth
`0`. It optimises for the thing you were trying to penalise.

The fix was scale, not episodes: `+0.1` per step, `+5` per pipe (a pipe is worth 50
steps of hovering), `-10` on death. Same 20,000 episodes, from zero to a solved game.

### 2. The state could not see the ceiling

The first observation was `(dx, dy, vy)` with `dy` measured from the gap centre.
Because the gap centre moves, `dy = 0` can mean "safe in a low gap" or "one pixel
from the ceiling", and the agent died against a wall it literally could not perceive.
Finer buckets alone got best runs into double digits; adding the absolute height
bucket `y` is what made the policy consistent.

Rule of thumb that came out of this: any quantity that can kill the agent needs an
absolute bucket, not only a position relative to a moving target.

### 3. Constructing an environment ate a random draw

`Flappy(seed)` used to start an episode in its constructor. Every caller that then
called `reset()` silently got the **second** level from that seed. I found it when
the one-bird gym and the new multi-bird sim disagreed about the same checkpoint on
the same seed: 848 steps in one, 189 in the other.

The constructor now only sets up the RNG and `reset()` draws the level. After the fix
both sims report identical death step and identical score on every checkpoint I
tested, and retraining produced a better table (71.0 pipes versus 59.6).

### Dead ends

Before the reward fix I spent several sweeps turning knobs. All of these were
evaluated the same way, greedy over 20 unseen seeds after 20,000 episodes:

| change | greedy pipes |
|---|---|
| baseline | 0.20 |
| slower pipes 1.2, gap 32 | 0.25 |
| gentler physics (gravity 0.34, flap 2.4, gap 32) | 2.40 |
| alpha 0.05 | 0.60 |
| alpha 0.30 | 0.80 |
| gamma 0.90 | 0.75 |
| gamma 0.99 with alpha 0.05 | 0.85 |

Seven knobs, no fix. After the reward change the same pipeline scored 71. The lesson
is the one at the top of the list above: when a demo is stuck at zero, suspect the
reward scale before you suspect the hyperparameters.

Potential-based reward shaping (Ng et al. 1999) was also tried, after the reward fix,
because it is the textbook answer to slow learning. It came out slightly worse here
(64.9 pipes versus 71 at the cap): the potential jumps when the target switches from
one pipe to the next, which is exactly the moment of success. It stays in the code as
`SHAPE_K`, off by default, so you can reproduce the result.

### How the videos are made

No game engine, no matplotlib, no editing software. Three stages:

**1. Simulate everything first.** The multi-bird sim holds one pipe list and one RNG,
and every bird reads the same gaps. It records a snapshot per step: pipe positions,
live bird positions, score, deaths. Rendering is then a pure function of those
snapshots, which means a video can be re-run with a different font or resolution
without touching the physics.

**2. Pick which checkpoints fly.** Each checkpoint is scored on the exact level the
video will use, the survival lengths are printed, and then 100 of the 201 are chosen
(`--pick even` for even spacing over episodes, `--pick score` for one per distinct
survival length). A learning curve this steep has a behavioural cliff that takes up
about 10% of the run, and uniform sampling spends most of its birds on identical
outcomes, so the choice matters more than it looks.

**3. Draw frames and pipe them into ffmpeg.** Sprites are pre-rendered once per
colour and wing state, the sky gradient is one numpy array reused for every frame,
fonts are cached per size, and each finished frame goes to x264 as raw RGB24 through
stdin. Nothing is written to disk in between.

Two details that made the 4K cut feel like video rather than a slideshow:

- **60 fps by resampling, not re-simulating.** The sim runs at 40 steps per second.
  The renderer walks output time with `s += rate / fps` and interpolates bird height
  and pipe position between simulation steps, so the scroll is smooth. Fast forward
  raises `rate` to 240 instead of dropping frames, so even the fast sections keep
  their motion. Wing beats key off the frame index, otherwise they alias at 6x.
- **A layout that survives a resolution change.** Everything is positioned in a
  900-pixel-tall design space and multiplied by the output height, so 1080p and 4K
  are the same composition. Both big numbers auto-fit their column: `10,000` at the
  original size is 262 px wide, which clips a 240 px margin at 1080p and a 480 px one
  at 4K.

Cost of the 4K master: 2,249 frames, 3 minutes 43 seconds, 7.0 MB at crf 23. The
1080p draft of the same video takes about 90 seconds.

### What this does not do

- **3,000 states is coarse.** A checkpoint trained for 18,600 episodes still crashed
  early on one level because a bucket boundary put it on the wrong side of a gap. The
  71 pipe average is real and so is that failure.
- **Round results are not monotone.** More training does not mean a better run on a
  fixed level, and the videos show that rather than hiding it.
- **A group is a best-of-N sample.** The 12 birds in a round share one policy and
  differ only by start position. That is enough to change the outcome: at jitter 0.9
  the 8,000 checkpoint reaches 50 pipes, at 1.3 it reaches 31.
- **One fixed level per video.** Fair comparisons across checkpoints, no claim about
  the distribution of levels beyond the 25 seeds used for evaluation.

## Files

```
flappy.py        environment: physics, collision, state discretisation
agent.py         tabular Q-learning (about 40 lines)
train.py         training loop, greedy evaluation, learning curve
render.py        4 checkpoints side by side -> ffmpeg
swarm.py         100 checkpoints on one shared level -> ffmpeg
rounds.py        one checkpoint per round, ends at the 50th pipe -> ffmpeg (4K)
draw.py          shared pixel drawing helpers
assets/fonts/    Space Grotesk + Inter (OFL), used by every renderer
watch.py         ASCII terminal playback
test_env.py      failure-first checks
q_table.json     the trained table
snapshots/       201 checkpoints from the training run
stats.json       every number quoted above
learning_curve.png
videos/          the three rendered cuts
```
