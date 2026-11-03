"""100 birds, one level, one hundred different amounts of training.

Needs a run that kept many checkpoints:

    python3 train.py --snap-every 100
    python3 swarm.py --birds 100 --seed 4242

Every bird is a real checkpoint from the same training run, so the only thing
that differs is how much the bird has learned. The pipe layout is generated once
and shared, so nobody gets an easier level.

The panel on the right is not decoration. It answers the three questions the
video raises: how many are left, how far the leader got, and which training
amounts died first. Bars freeze grey where a bird crashed, so the chart doubles
as a record of the deaths.
"""
from __future__ import annotations

import argparse
import colorsys
import glob
import os
import random
import subprocess

from PIL import Image, ImageDraw

import draw as D
import env as F
from agent import QAgent

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 10.0
GAME_W, GAME_H = int(F.W * SCALE), int(F.H * SCALE)
PANEL_W = 520
CANVAS_W, CANVAS_H = GAME_W + PANEL_W, GAME_H
FPS = 40
INK = (22, 27, 31)
PANEL_BG = (20, 25, 29)
MUTED = (148, 158, 162)


# ---------------------------------------------------------------- shared level
def simulate(agents, seed: int, max_steps: int, spread: float = 0.16,
             y_offsets=None, vy_offsets=None, stop_at: int | None = None):
    """All birds in one world. One pipe list, one RNG, identical gaps.

    The gap sequence is drawn from a fixed schedule that no bird can influence,
    so every bird meets the same pipes in the same order. `spread` shifts each
    bird a fraction of a pixel apart; without it, identical policies stack into
    one sprite and look like a single bird.

    `stop_at` ends the run the moment any bird clears that many pipes, which is
    what makes the round robin video stop on a win instead of a timeout.
    """
    rng = random.Random(seed)
    pipes = [[F.W + 6.0, F.sample_gap(rng), 0]]
    next_id = 1
    n = len(agents)
    xs = [F.BIRD_X + (i - (n - 1) / 2.0) * spread for i in range(n)]
    ys = [F.GROUND_Y * 0.45 + (y_offsets[i] if y_offsets else 0.0) for i in range(n)]
    vys = [vy_offsets[i] if vy_offsets else 0.0 for i in range(n)]
    alive = [True] * n
    passed = [set() for _ in range(n)]
    died_at = [-1] * n
    cause = [""] * n
    death_pos = []
    frames = []

    for step in range(max_steps + 1):
        frames.append({
            "step": step,
            "pipes": [(p[0], p[1]) for p in pipes],
            "birds": [(i, xs[i], ys[i]) for i in range(n) if alive[i]],
            "alive": sum(alive),
            "best": max(len(s) for s in passed),
            "died_at": list(died_at),
            "pipes_now": [len(s) for s in passed],
        })
        if stop_at is not None and max(len(s) for s in passed) >= stop_at:
            break
        if not any(alive):
            break

        for i in range(n):
            if not alive[i]:
                continue
            action = agents[i].greedy(F.observe(pipes, ys[i], vys[i]))
            vy = F.FLAP_V if action == 1 else vys[i]
            vys[i] = min(vy + F.GRAVITY, F.MAX_FALL)
            ys[i] += vys[i]

        for p in pipes:
            p[0] -= F.PIPE_SPEED
        pipes = [p for p in pipes if p[0] > -F.PIPE_W - 12.0]
        if not pipes or pipes[-1][0] <= F.W - F.PIPE_SPACING:
            anchor = pipes[-1][0] if pipes else F.W
            pipes.append([anchor + F.PIPE_SPACING, F.sample_gap(rng), next_id])
            next_id += 1

        for i in range(n):
            if not alive[i]:
                continue
            for p in pipes:
                if p[2] not in passed[i] and p[0] + F.PIPE_W < xs[i] - F.BIRD_HW:
                    passed[i].add(p[2])
            reason = ""
            if ys[i] - F.BIRD_HH <= 0.0:
                reason = "ceiling"
            elif ys[i] + F.BIRD_HH >= F.GROUND_Y:
                reason = "ground"
            else:
                for p in pipes:
                    if _hits(xs[i], ys[i], p):
                        reason = "pipe"
                        break
            if reason:
                alive[i] = False
                died_at[i] = step + 1
                cause[i] = reason
                death_pos.append((step + 1, xs[i], ys[i]))

    stats = {"passed": [len(s) for s in passed], "died_at": died_at, "cause": cause,
             "death_pos": death_pos}
    return frames, stats


def _hits(x: float, y: float, p) -> bool:
    if x + F.BIRD_HW < p[0] or x - F.BIRD_HW > p[0] + F.PIPE_W:
        return False
    top = p[1] - F.PIPE_GAP / 2
    bot = p[1] + F.PIPE_GAP / 2
    return y - F.BIRD_HH < top or y + F.BIRD_HH > bot


# ------------------------------------------------------------------ checkpoints
def load_checkpoints(folder: str):
    paths = sorted(glob.glob(os.path.join(folder, "snapshots", "ep_*.json")))
    return [(int(os.path.basename(p).split("_")[1].split(".")[0]), p) for p in paths]


def score_checkpoints(marks, seed: int, max_steps: int):
    """How long does each checkpoint survive the level we are about to show?"""
    out = []
    for ep, path in marks:
        agent = QAgent.load(path)
        env = F.Flappy(seed=seed)
        obs, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            obs, _, done = env.step(agent.greedy(obs))
            steps += 1
        out.append({"ep": ep, "path": path, "steps": steps, "pipes": env.pipes_passed})
    return out


def pick_even(scored, n: int):
    scored = sorted(scored, key=lambda r: r["ep"])
    step = len(scored) / n
    return [scored[min(len(scored) - 1, int(i * step))] for i in range(n)]


def pick_by_score(scored, n: int):
    """One bird per distinct behaviour, so the deaths spread across the whole video."""
    by_steps: dict[int, list[dict]] = {}
    for row in scored:
        by_steps.setdefault(row["steps"], []).append(row)
    picks = []
    for steps in sorted(by_steps):
        group = sorted(by_steps[steps], key=lambda r: r["ep"])
        picks.append(group[len(group) // 2])
    if len(picks) > n:
        k = len(picks) / n
        picks = [picks[min(len(picks) - 1, int(i * k))] for i in range(n)]
    elif len(picks) < n:
        # fill up from the biggest groups, newest first
        for steps in sorted(by_steps, key=lambda s: -len(by_steps[s])):
            group = sorted(by_steps[steps], key=lambda r: r["ep"])
            for row in group:
                if len(picks) >= n:
                    break
                if row not in picks:
                    picks.append(row)
            if len(picks) >= n:
                break
    return sorted(picks[:n], key=lambda r: r["ep"])


# ---------------------------------------------------------------------- drawing
def hue_color(t: float) -> tuple[int, int, int]:
    """t in 0..1, red (untrained) through yellow to green (trained)."""
    r, g, b = colorsys.hsv_to_rgb((0.02 + 0.31 * t), 0.82, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def _darken(color, k=0.62):
    return tuple(int(c * k) for c in color)


def bird_sprite(color, wing: int, s: float = SCALE * 0.75) -> Image.Image:
    w, h = int(9.0 * s), int(7.0 * s)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w / 2.0, h / 2.0
    rw, rh = 2.25 * s, 1.85 * s
    d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=color + (232,))
    wy = -0.35 * s if wing else 0.45 * s
    d.ellipse([cx - 1.7 * s, cy + wy - 0.6 * s, cx + 0.1 * s, cy + wy + 1.2 * s],
              fill=_darken(color) + (232,))
    d.ellipse([cx + 0.45 * s, cy - 1.05 * s, cx + 1.5 * s, cy + 0.05 * s], fill=(255, 255, 255, 240))
    d.ellipse([cx + 0.9 * s, cy - 0.8 * s, cx + 1.35 * s, cy - 0.3 * s], fill=(24, 28, 32, 240))
    d.polygon([(cx + 2.05 * s, cy - 0.25 * s), (cx + 3.55 * s, cy + 0.5 * s),
               (cx + 1.95 * s, cy + 1.1 * s)], fill=(248, 152, 60, 240))
    return img


def background() -> Image.Image:
    """Full canvas once: sky gradient plus ground on the left, panel fill on the right."""
    import numpy as np
    ys = np.linspace(0.0, 1.0, GAME_H)[:, None]
    top, bot = np.array(D.SKY_TOP, float), np.array(D.SKY_BOT, float)
    grad = top[None, :] + (bot - top)[None, :] * ys
    arr = np.repeat(grad[:, None, :], GAME_W, axis=1).astype("uint8")
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), PANEL_BG)
    img.paste(Image.fromarray(arr, "RGB"), (0, 0))
    d = ImageDraw.Draw(img)
    gy = int(F.GROUND_Y * SCALE)
    d.rectangle([0, gy, GAME_W, GAME_H], fill=D.GROUND)
    d.rectangle([0, gy, GAME_W, gy + int(2 * SCALE)], fill=D.GROUND_TOP)
    return img


def draw_pipe(d: ImageDraw.ImageDraw, px: float, gap: float, t: int) -> None:
    left, right = px * SCALE, (px + F.PIPE_W) * SCALE
    if right < 0 or left > GAME_W:
        return
    top_end = (gap - F.PIPE_GAP / 2) * SCALE
    bot_start = (gap + F.PIPE_GAP / 2) * SCALE
    gy = F.GROUND_Y * SCALE
    for ya, yb in ((0.0, top_end), (bot_start, gy)):
        d.rectangle([left, ya, right, yb], fill=D.PIPE)
        d.rectangle([left, ya, left + 2.2 * SCALE, yb], fill=D.PIPE_LIGHT)
        d.rectangle([right - 2.2 * SCALE, ya, right, yb], fill=D.PIPE_DARK)
    lip = 2.6 * SCALE
    for ya, yb in ((top_end - lip, top_end), (bot_start, bot_start + lip)):
        d.rectangle([left - 1.6 * SCALE, ya, right + 1.6 * SCALE, yb], fill=D.PIPE)
        d.rectangle([left - 1.6 * SCALE, ya, right + 1.6 * SCALE, ya + 1.8 * SCALE], fill=D.PIPE_LIGHT)
        d.rectangle([left - 1.6 * SCALE, yb - 1.8 * SCALE, right + 1.6 * SCALE, yb], fill=D.PIPE_DARK)


PIPE_AXIS = 25          # bar chart scale, in pipes cleared
HIST_BUCKETS = 25       # crash histogram columns


def death_hist(died_at, step: int, total: int, nb: int = HIST_BUCKETS) -> list[int]:
    counts = [0] * nb
    for d in died_at:
        if 0 <= d <= step:
            counts[min(nb - 1, int(d / max(1, total) * nb))] += 1
    return counts


def ghost(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    """A small cross where a bird went down."""
    cx, cy, r = x * SCALE, y * SCALE, 4.5
    d.line([cx - r, cy - r, cx + r, cy + r], fill=(126, 116, 102), width=2)
    d.line([cx - r, cy + r, cx + r, cy - r], fill=(126, 116, 102), width=2)


def timeline(d: ImageDraw.ImageDraw, step: int, total: int) -> None:
    y, x0, x1 = GAME_H - 22, 44, GAME_W - 44
    d.rectangle([x0, y, x1, y + 4], fill=(198, 188, 120))
    w = (x1 - x0) * min(1.0, step / max(1, total))
    d.rectangle([x0, y, x0 + w, y + 4], fill=(118, 198, 148))
    d.ellipse([x0 + w - 4, y - 3, x0 + w + 4, y + 7], fill=(250, 250, 245))


def panel(canvas: Image.Image, frame: dict, picked, colors, total_steps: int) -> None:
    d = ImageDraw.Draw(canvas)
    d.rectangle([GAME_W, 0, CANVAS_W, CANVAS_H], fill=PANEL_BG)
    pad = 26
    left, right = GAME_W + pad, CANVAS_W - pad
    n = len(picked)
    alive = frame["alive"]

    d.text((left, 14), f"{n} birds", font=D.font(25), fill=(250, 250, 245))
    sub = "same level, different amounts of training"
    d.text((left, 46), sub, font=D.fit_font(sub, right - left, start=12, min_size=9), fill=MUTED)
    d.line([left, 76, right, 76], fill=(44, 50, 55))

    d.text((left, 88), "still flying", font=D.font(12), fill=MUTED)
    big = f"{alive}"
    d.text((left, 102), big, font=D.font(46), fill=(120, 226, 170))
    d.text((left + D.text_width(big, 46) + 10, 126), f"of {n}", font=D.font(16), fill=MUTED)
    d.text((left, 154), f"crashed: {n - alive}", font=D.font(12), fill=(198, 138, 122))
    d.text((left, 176), f"furthest: {frame['best']} pipes", font=D.font(15), fill=(230, 233, 235))
    d.text((left, 198), f"step {frame['step']} of {total_steps}", font=D.font(12), fill=MUTED)
    d.line([left, 222, right, 222], fill=(44, 50, 55))

    # live leaderboard: the birds furthest through the level
    d.text((left, 230), "top of the flock", font=D.font(12), fill=MUTED)
    d.text((right - D.text_width("pipes", 11), 232), "pipes", font=D.font(11), fill=(96, 106, 110))
    alive_idx = [i for i in range(n) if frame["died_at"][i] < 0]
    top = sorted(alive_idx, key=lambda i: (-frame["pipes_now"][i], -picked[i]["ep"]))[:5]
    y = 248
    for rank, i in enumerate(top, 1):
        d.text((left, y), f"{rank}", font=D.font(12), fill=(104, 114, 118))
        d.text((left + 18, y), f"ep {picked[i]['ep']:,}", font=D.font(12), fill=colors[i])
        txt = f"{frame['pipes_now'][i]}"
        d.text((right - D.text_width(txt, 12), y), txt, font=D.font(12), fill=(232, 236, 238))
        y += 18
    while y < 248 + 5 * 18:
        d.text((left + 18, y), "crashed", font=D.font(11), fill=(74, 80, 84))
        y += 18
    d.line([left, 346, right, 346], fill=(44, 50, 55))

    # one bar per bird: how many pipes it has cleared, ordered by training
    d.text((left, 354), "pipes cleared by each bird", font=D.font(12), fill=MUTED)
    bx0, bw = left + 78, right - (left + 78)
    by0, row = 376, 3.2
    for i in range(n):
        yy = by0 + i * row
        d.rectangle([bx0, yy, bx0 + bw, yy + row - 0.8], fill=(30, 36, 41))
        dead = frame["died_at"][i] >= 0
        pipes = frame["pipes_now"][i]
        w = bw * min(pipes, PIPE_AXIS) / PIPE_AXIS
        if w > 0:
            d.rectangle([bx0, yy, bx0 + w, yy + row - 0.8],
                        fill=(96, 100, 104) if dead else colors[i])
        if i % 25 == 0 or i == n - 1:
            lab = f"{picked[i]['ep']:,}"
            d.text((left, yy - 3.5), lab, font=D.font(9), fill=(122, 132, 136))
    by1 = by0 + n * row
    for tick in range(0, PIPE_AXIS + 1, 5):
        tx = bx0 + bw * tick / PIPE_AXIS
        d.line([tx, by1 + 1, tx, by1 + 4], fill=(70, 76, 80))
        lab = str(tick)
        d.text((tx - D.text_width(lab, 9) / 2, by1 + 6), lab, font=D.font(9), fill=(122, 132, 136))

    # when the flock died
    hy0 = by1 + 34
    d.text((left, hy0 - 17), "crashes over time", font=D.font(12), fill=MUTED)
    counts = death_hist(frame["died_at"], frame["step"], total_steps)
    hmax = max(6, max(counts))
    hh, bwid = 74, (right - left) / HIST_BUCKETS
    for k, c in enumerate(counts):
        if not c:
            continue
        bh = hh * c / hmax
        d.rectangle([left + k * bwid, hy0 + hh - bh,
                     left + (k + 1) * bwid - 1.5, hy0 + hh], fill=(196, 138, 122))
    d.line([left, hy0 + hh + 1, right, hy0 + hh + 1], fill=(60, 66, 70))
    d.text((left, hy0 + hh + 5), "0", font=D.font(9), fill=(122, 132, 136))
    lab = f"{total_steps} steps"
    d.text((right - D.text_width(lab, 9), hy0 + hh + 5), lab, font=D.font(9), fill=(122, 132, 136))

    note = "one row per checkpoint from the same run, ordered by how long it trained"
    d.text((left, CANVAS_H - 22), note,
           font=D.fit_font(note, right - left, start=9, min_size=8), fill=(104, 114, 118))


def card(lines, subs, bg=INK) -> Image.Image:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    d = ImageDraw.Draw(img)
    inner = CANVAS_W - 160
    f1 = D.fit_font(lines, inner, start=54)
    d.text(((CANVAS_W - D.text_width(lines, f1.size)) / 2, CANVAS_H / 2 - 150), lines,
           font=f1, fill=(250, 250, 245))
    y = CANVAS_H / 2 - 40
    for line in subs:
        f = D.fit_font(line, inner, start=22, min_size=12)
        d.text(((CANVAS_W - D.text_width(line, f.size)) / 2, y), line, font=f, fill=(140, 208, 172))
        y += f.size * 1.9
    return img


def summary_overlay(canvas: Image.Image, stats, picked, total_steps: int) -> None:
    d = ImageDraw.Draw(canvas, "RGBA")
    box = [int(GAME_W * 0.47), 150, GAME_W - 60, GAME_H - 120]
    d.rounded_rectangle(box, radius=26, fill=(14, 18, 22, 226))
    passed = stats["passed"]
    n = len(picked)
    alive_end = sum(1 for x in stats["died_at"] if x < 0)
    rows = [
        (f"{sum(1 for p in passed if p == 0)}", "never cleared a pipe"),
        (f"{sum(1 for p in passed if 1 <= p <= 5)}", "cleared 1 to 5 pipes"),
        (f"{sum(1 for p in passed if 6 <= p <= 20)}", "cleared 6 to 20 pipes"),
        (f"{sum(1 for p in passed if p > 20)}", "cleared more than 20"),
    ]
    d.text((box[0] + 60, box[1] + 40), f"{n} checkpoints, one level",
           font=D.font(34), fill=(250, 250, 245))
    y = box[1] + 110
    for big, label in rows:
        d.text((box[0] + 62, y), big, font=D.font(36), fill=(120, 226, 170))
        d.text((box[0] + 62 + D.text_width(big, 36) + 22, y + 12), label, font=D.font(20),
               fill=(226, 230, 232))
        y += 58
    d.text((box[0] + 62, y + 12), f"episodes trained: {picked[0]['ep']:,} to {picked[-1]['ep']:,}",
           font=D.font(18), fill=MUTED)
    d.text((box[0] + 62, y + 40), f"still flying when the tape stops: {alive_end}",
           font=D.font(18), fill=MUTED)


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--birds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242, help="level seed, shared by every bird")
    ap.add_argument("--steps", type=int, default=1000, help="step cap for the video")
    ap.add_argument("--pick", choices=("even", "score"), default="score")
    ap.add_argument("--dry-run", action="store_true", help="score and pick only, no video")
    ap.add_argument("--out", default=os.path.expanduser("~/videos/rl_flappy_swarm.mp4"))
    args = ap.parse_args()

    marks = load_checkpoints(HERE)
    if len(marks) < args.birds:
        raise SystemExit(f"need at least {args.birds} checkpoints, found {len(marks)}. "
                         f"Retrain with: python3 train.py --snap-every 100")
    print(f"  {len(marks)} checkpoints found, scoring each on level {args.seed} ...", flush=True)
    scored = score_checkpoints(marks, args.seed, args.steps)
    dist: dict[int, int] = {}
    for row in scored:
        dist[row["steps"]] = dist.get(row["steps"], 0) + 1
    print(f"  distinct survival lengths: {len(dist)}  "
          f"(top of range: {max(dist)} steps)")

    picked = pick_even(scored, args.birds) if args.pick == "even" else pick_by_score(scored, args.birds)
    print("  picked episodes:", [r["ep"] for r in picked][:6], "...", [r["ep"] for r in picked][-4:])
    if args.dry_run:
        pops = {}
        for row in picked:
            pops[row["steps"]] = pops.get(row["steps"], 0) + 1
        print("  picked survival lengths:", sorted(pops.items(),
                                                   key=lambda kv: kv[0])[:4], "...",
              sorted(pops.items(), key=lambda kv: kv[0])[-4:])
        return
    agents = [QAgent.load(r["path"]) for r in picked]
    top = max(r["ep"] for r in picked) or 1
    colors = [hue_color(r["ep"] / top) for r in picked]

    print("  simulating ...", flush=True)
    frames, stats = simulate(agents, args.seed, args.steps)
    sprites = [(bird_sprite(c, 0), bird_sprite(c, 1)) for c in colors]

    base = background()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for _ in range(60):
        proc.stdin.write(card(f"{len(picked)} birds, one level",
                              ["each bird is the same algorithm after a different amount of training",
                               "identical gaps, identical physics, only the learned Q-table differs"],
                              ).tobytes())

    assert base.size == (CANVAS_W, CANVAS_H), f"canvas {base.size} vs {(CANVAS_W, CANVAS_H)}"
    for fi, fr in enumerate(frames):
        canvas = base.copy()
        d = ImageDraw.Draw(canvas)
        for px, gap in fr["pipes"]:
            draw_pipe(d, px, gap, fi)
        for dstep, gx, gy in stats["death_pos"]:
            if dstep <= fr["step"]:
                ghost(d, gx, gy)
        timeline(d, fr["step"], args.steps)
        wing = 1 if (fi // 4) % 2 else 0
        for idx, x, y in fr["birds"]:
            sp = sprites[idx][wing]
            canvas.paste(sp, (int(x * SCALE - sp.width / 2), int(y * SCALE - sp.height / 2)), sp)
        panel(canvas, fr, picked, colors, args.steps)
        proc.stdin.write(canvas.tobytes())
        if fi % 200 == 0:
            print(f"    frame {fi:4d}  alive={fr['alive']:3d}  best={fr['best']:3d}", flush=True)

    last = base.copy()
    dlast = ImageDraw.Draw(last)
    for px, gap in frames[-1]["pipes"]:
        draw_pipe(dlast, px, gap, 0)
    for dstep, gx, gy in stats["death_pos"]:
        ghost(dlast, gx, gy)
    timeline(dlast, frames[-1]["step"], args.steps)
    for idx, x, y in frames[-1]["birds"]:
        sp = sprites[idx][0]
        last.paste(sp, (int(x * SCALE - sp.width / 2), int(y * SCALE - sp.height / 2)), sp)
    panel(last, frames[-1], picked, colors, args.steps)
    summary_overlay(last, stats, picked, args.steps)
    for _ in range(110):
        proc.stdin.write(last.tobytes())

    proc.stdin.close()
    proc.wait()

    total = 60 + len(frames) + 110
    print(f"  wrote {args.out}  ({total} frames, {total / FPS:.1f}s)")
    print(f"  alive at the end: {frames[-1]['alive']}/{len(picked)}   "
          f"furthest: {frames[-1]['best']} pipes")


if __name__ == "__main__":
    main()
