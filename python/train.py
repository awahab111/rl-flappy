"""Train the table, then write every artifact the README quotes.

    python3 train.py                     # 20k lives, write everything
    python3 train.py --episodes 40000    # more practice

Writes q_table.json, snapshots/ep_*.json, learning_curve.png and stats.json.
stats.json is the source of truth for the numbers in the README. If a number
appears in the docs and not in that file, it is a guess.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

import draw as D
import env as F
from agent import QAgent

CURVE_BG = (250, 250, 247)
CURVE_GRID = (226, 227, 221)
CURVE_FAINT = (186, 204, 198)
CURVE_LINE = (32, 132, 116)
CURVE_INK = (40, 44, 46)
CURVE_ACCENT = (206, 122, 40)


def train(episodes: int, seed: int, out_dir: str, snap_at: tuple[int, ...], alpha: float,
          gamma: float, eps_start: float = 1.0, eps_end: float = 0.02,
          decay_frac: float = 0.4, snap_every: int = 0) -> dict:
    agent = QAgent(alpha=alpha, gamma=gamma, seed=seed)
    env = F.Flappy(seed=seed)
    pipes_hist, steps_hist = [], []
    snap_at = tuple(sorted(set(snap_at)))
    snaps_done: list[int] = []
    decay_eps = max(1, int(episodes * decay_frac))

    os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
    _save_snapshot(agent, out_dir, 0)
    snaps_done.append(0)
    t0 = time.time()

    for ep in range(1, episodes + 1):
        # Exploration falls in a straight line from 1.0 to 0.02, then holds.
        # It does not stop at zero: a small amount of guessing keeps the table
        # from freezing on the first policy that happens to work.
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (ep / decay_eps))
        obs = env.reset()
        done, steps = False, 0
        while not done:
            a = agent.act(obs, eps)
            nxt, r, done = env.step(a)
            agent.update(obs, a, r, nxt, done)
            obs = nxt
            steps += 1
        pipes_hist.append(env.pipes_passed)
        steps_hist.append(steps)

        if ep in snap_at and ep not in snaps_done:
            _save_snapshot(agent, out_dir, ep)
            snaps_done.append(ep)
        if snap_every and ep % snap_every == 0 and ep not in snaps_done:
            _save_snapshot(agent, out_dir, ep)
            snaps_done.append(ep)
        if episodes >= 10 and ep % max(1, episodes // 10) == 0:
            tail = pipes_hist[-max(1, episodes // 10):]
            print(f"  ep {ep:6d}/{episodes}  eps={eps:.3f}  "
                  f"mean pipes (last {len(tail)}) = {np.mean(tail):6.2f}  "
                  f"best = {max(pipes_hist):3d}  {time.time() - t0:5.1f}s", flush=True)

    agent.save(os.path.join(out_dir, "q_table.json"))
    if episodes not in snaps_done:
        _save_snapshot(agent, out_dir, episodes)
        snaps_done.append(episodes)

    eval_res = evaluate(agent, n=20, seed_offset=90000)
    long_res = evaluate_long(agent, n=5, cap=8000, seed_offset=77000)
    deciles = []
    k = max(1, len(pipes_hist) // 10)
    for i in range(0, len(pipes_hist), k):
        chunk = pipes_hist[i:i + k]
        if len(chunk) >= max(1, k // 2):
            deciles.append({"episodes": [i + 1, i + len(chunk)],
                            "mean_pipes": round(float(np.mean(chunk)), 2),
                            "max_pipes": int(max(chunk))})

    stats = {
        "episodes": episodes,
        "seed": seed,
        "hyperparams": {"alpha": alpha, "gamma": gamma, "eps_start": eps_start,
                        "eps_end": eps_end, "decay_frac": decay_frac},
        "rewards": {"alive": F.REWARD_ALIVE, "pipe": F.REWARD_PIPE, "death": F.REWARD_DEATH},
        "pipes_per_episode": pipes_hist,
        "steps_per_episode": steps_hist,
        "deciles": deciles,
        "first_100_mean": round(float(np.mean(pipes_hist[:100])), 2),
        "last_100_mean": round(float(np.mean(pipes_hist[-100:])), 2),
        "best_episode_pipes": int(max(pipes_hist)),
        "eval_greedy": eval_res,
        "eval_long": long_res,
        "snapshots": snaps_done,
        "train_seconds": round(time.time() - t0, 1),
        "state_space": int(F.N_DX * F.N_DY * F.N_VY * F.N_Y),
        "q_states_learned": int(np.count_nonzero(np.abs(agent.Q).sum(axis=-1) > 0)),
        "q_values_total": int(agent.Q.size),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    plot_curve(stats, os.path.join(out_dir, "learning_curve.png"))
    return stats


def _save_snapshot(agent: QAgent, out_dir: str, ep: int) -> None:
    agent.save(os.path.join(out_dir, "snapshots", f"ep_{ep:05d}.json"))


def evaluate(agent: QAgent, n: int = 20, seed_offset: int = 90000, cap: int = None) -> dict:
    """Score the policy with no randomness, on levels it has never seen.

    This is the number worth reporting. The training loop runs with epsilon at
    2 percent, and one random flap ends a long run, so the training average sits
    far below what the policy can actually do. Both numbers are written to
    stats.json and the README explains the gap.
    """
    old_cap = F.MAX_STEPS
    if cap:
        F.MAX_STEPS = cap
    scores, reasons = [], {}
    try:
        for i in range(n):
            env = F.Flappy(seed=seed_offset + i)
            obs, done = env.reset(), False
            while not done:
                obs, _, done = env.step(agent.greedy(obs))
            scores.append(env.pipes_passed)
            reasons[env.death_reason] = reasons.get(env.death_reason, 0) + 1
    finally:
        F.MAX_STEPS = old_cap
    return {
        "n": n, "cap_steps": cap or old_cap,
        "mean_pipes": round(float(np.mean(scores)), 2),
        "median_pipes": float(np.median(scores)),
        "min_pipes": int(min(scores)), "max_pipes": int(max(scores)),
        "scores": scores, "death_causes": reasons,
    }


def evaluate_long(agent: QAgent, n: int = 5, cap: int = 8000, seed_offset: int = 77000) -> dict:
    return evaluate(agent, n=n, seed_offset=seed_offset, cap=cap)


def plot_curve(stats: dict, path: str, window: int = 100) -> None:
    Wd, Hd = 1000, 470
    pad_l, pad_r, pad_t, pad_b = 70, 26, 58, 78
    img = Image.new("RGB", (Wd, Hd), CURVE_BG)
    d = ImageDraw.Draw(img)
    series = stats["pipes_per_episode"]
    n = len(series)
    peak = max(series) if series else 1
    top = max(4, int(np.ceil(peak / 4.0)) * 4)
    x0, x1 = pad_l, Wd - pad_r
    y0, y1 = pad_t, Hd - pad_b

    def xmap(i):
        return x0 + (x1 - x0) * (i / max(1, n - 1))

    def ymap(v):
        return y1 - (y1 - y0) * (v / max(1e-9, top))

    for k in range(5):
        v = top * k / 4
        yy = ymap(v)
        d.line([(x0, yy), (x1, yy)], fill=CURVE_GRID, width=1)
        d.text((x0 - 42, yy - 9), str(int(v)), font=D.font(14), fill=CURVE_INK)
    for i in range(6):
        xx = x0 + (x1 - x0) * i / 5
        d.line([(xx, y0), (xx, y1)], fill=CURVE_GRID, width=1)
        d.text((xx - 20, y1 + 10), str(int(n * i / 5)), font=D.font(14), fill=CURVE_INK)

    if n > 1:
        d.line([(xmap(i), ymap(v)) for i, v in enumerate(series)], fill=CURVE_FAINT, width=1)
    if n >= window:
        smooth = np.convolve(np.array(series, dtype=float), np.ones(window) / window, mode="valid")
        d.line([(xmap(i + window // 2), ymap(v)) for i, v in enumerate(smooth)],
               fill=CURVE_LINE, width=4)

    d.text((pad_l, 12), "pipes cleared per training run", font=D.font(22), fill=CURVE_INK)
    d.text((x1 - 330, 10), f"greedy: {stats['eval_greedy']['mean_pipes']} pipes/run over "
                           f"{stats['eval_greedy']['n']} unseen worlds",
           font=D.font(13), fill=CURVE_ACCENT)
    lg = stats.get("eval_long", {})
    if lg:
        d.text((x1 - 330, 28), f"long horizon: up to {lg['max_pipes']} pipes",
               font=D.font(13), fill=CURVE_ACCENT)
    d.text((pad_l, y1 + 50),
           "thin line = one run (2% random actions), thick line = 100-run average, "
           f"epsilon 1.00 -> 0.02 over the first {int(stats['hyperparams']['decay_frac'] * 100)}%",
           font=D.font(12), fill=(126, 130, 126))
    d.text((x1 - 58, y1 + 28), "episode", font=D.font(15), fill=CURVE_INK)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--snap-marks", default="0.40,0.70",
                    help="comma separated episode counts or fractions, e.g. 0.4,12000")
    ap.add_argument("--snap-every", type=int, default=0,
                    help="also save a checkpoint every N episodes (for the swarm video)")
    args = ap.parse_args()
    marks = set()
    for token in args.snap_marks.split(","):
        token = token.strip()
        if not token:
            continue
        marks.add(int(args.episodes * float(token)) if float(token) < 1.0 else int(token))
    marks.add(args.episodes)
    stats = train(args.episodes, args.seed, args.out, tuple(marks), args.alpha, args.gamma,
                  snap_every=args.snap_every)
    print("\nfirst 100 episodes : %.2f pipes/run" % stats["first_100_mean"])
    print("last  100 episodes : %.2f pipes/run  (with 2%% random actions)" % stats["last_100_mean"])
    print("best single run    : %d pipes" % stats["best_episode_pipes"])
    ev, lg = stats["eval_greedy"], stats["eval_long"]
    print("greedy eval        : %.2f pipes/run over %d unseen worlds (min %d, max %d)"
          % (ev["mean_pipes"], ev["n"], ev["min_pipes"], ev["max_pipes"]))
    print("long horizon       : %s pipes (cap %d steps)"
          % (lg["scores"], lg["cap_steps"]))
    print("state space        : %d states, %d Q values, %d states ever visited"
          % (stats["state_space"], stats["q_values_total"], stats["q_states_learned"]))
    print("train time         : %.1fs" % stats["train_seconds"])


if __name__ == "__main__":
    main()
