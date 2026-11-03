"""The four panel video: one level, four checkpoints, side by side.

All four birds play the same pipe layout with the same seed, so the only
difference on screen is how much each one trained. When a bird dies its panel
freezes and dims, and the other panels keep running. That is what makes the gap
between checkpoints visible: the left panels stop moving while the right ones
carry on.

Usage:  python3 render.py --episodes 0,7500,8000,20000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import draw as D
import env as F
from agent import QAgent

PANEL_W, PANEL_H = 360, 270          # 120x90 world at scale 3
SCALE = 3.0
COLS, ROWS = 2, 2
CANVAS_W, CANVAS_H = PANEL_W * COLS, PANEL_H * ROWS
FPS = 40
MAX_VIDEO_STEPS = 750
HERE = os.path.dirname(os.path.abspath(__file__))


def panel_bg() -> Image.Image:
    """Fast vertical sky gradient, computed once with numpy."""
    ys = np.linspace(0.0, 1.0, PANEL_H)[:, None]
    top = np.array(D.SKY_TOP, dtype=float)
    bot = np.array(D.SKY_BOT, dtype=float)
    grad = top[None, :] + (bot - top)[None, :] * ys     # (PANEL_H, 3)
    arr = np.repeat(grad[:, None, :], PANEL_W, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def rollout_frames(qpath: str, seed: int, max_steps: int):
    agent = QAgent.load(qpath)
    env = F.Flappy(seed=seed)
    _, _, frames = F.rollout(env, agent.greedy, max_steps=max_steps, record=True)
    return frames, env


def title_card(text: str, subs: list[str], bg=(24, 30, 34)) -> Image.Image:
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    d = ImageDraw.Draw(img)
    inner = CANVAS_W - 80
    f1 = D.fit_font(text, inner, start=46)
    bb = d.textbbox((0, 0), text, font=f1)
    d.text(((CANVAS_W - (bb[2] - bb[0])) / 2, CANVAS_H / 2 - 96), text, font=f1, fill=(250, 250, 245))
    y = CANVAS_H / 2 - 96 + (bb[3] - bb[1]) + 40
    for line in subs:
        f = D.fit_font(line, inner, start=21, min_size=12)
        d.text(((CANVAS_W - D.text_width(line, f.size)) / 2, y), line, font=f, fill=(150, 214, 180))
        y += f.size * 1.9
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/videos/rl_flappy_learning.mp4"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=MAX_VIDEO_STEPS)
    ap.add_argument("--episodes", default="",
                    help="comma separated checkpoints to compare, e.g. 0,8000,9000,20000")
    ap.add_argument("--dry-run", action="store_true", help="roll out each panel and report, no encoding")
    args = ap.parse_args()

    snaps = sorted(glob.glob(os.path.join(HERE, "snapshots", "ep_*.json")))
    if not snaps:
        raise SystemExit("no snapshots yet, run train.py first")
    if args.episodes:
        wanted = []
        for token in args.episodes.split(","):
            path = os.path.join(HERE, "snapshots", f"ep_{int(token.strip()):05d}.json")
            if not os.path.exists(path):
                raise SystemExit(f"missing checkpoint {path}")
            wanted.append(path)
        snaps = wanted
    elif len(snaps) > 4:
        snaps = [snaps[0], snaps[1], snaps[-2], snaps[-1]]
    eps = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in snaps]
    stats_path = os.path.join(HERE, "stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}

    bases, rolls = [], []
    for path in snaps:
        frames, env = rollout_frames(path, args.seed, args.steps)
        bases.append(frames)
        rolls.append(env)
        print(f"  {os.path.basename(path):>18} -> {env.pipes_passed:3d} pipes, "
              f"{len(frames)} frames, died by {env.death_reason or 'time limit'}")

    total = min(max(len(f) for f in bases), args.steps + 1)
    if args.dry_run:
        print(f"  {len(snaps)} panels, {total} frames, {total / FPS:.1f}s at {FPS} fps, no encoding")
        return
    bg = panel_bg()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    intro = 70
    n_states = stats.get("state_space", 3000)
    for _ in range(intro):
        img = title_card(f"can a {n_states:,}-state table learn to fly?",
                         ["tabular Q-learning, no neural nets, no dependencies"])
        proc.stdin.write(img.tobytes())

    for i in range(total):
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
        for k in range(len(snaps)):
            px, py = (k % COLS) * PANEL_W, (k // COLS) * PANEL_H
            canvas.paste(bg, (px, py))
            frames = bases[k]
            fr = frames[i] if i < len(frames) else frames[-1]
            dead = i >= len(frames) - 1
            D.draw_scene(canvas, (px, py, px + PANEL_W, py + PANEL_H),
                         fr["pipes"], fr["y"], i, s=SCALE)
            if dead:
                veil = Image.new("RGB", (PANEL_W, PANEL_H), (8, 12, 16))
                canvas.paste(Image.blend(canvas.crop((px, py, px + PANEL_W, py + PANEL_H)),
                                         veil, 0.45), (px, py))
            D.hud(canvas, (px + 8, py + 8), f"episode {eps[k]}", size=14)
            txt = f"{fr['passed']} pipes"
            d = ImageDraw.Draw(canvas)
            w = d.textbbox((0, 0), txt, font=D.font(14))[2]
            D.hud(canvas, (px + PANEL_W - w - 30, py + 8), txt, size=14,
                  bg=(32, 132, 116) if not dead else (150, 60, 52))
            if dead:
                msg = f"died after {fr['passed']} pipes" if fr["passed"] else "never cleared a pipe"
                d = ImageDraw.Draw(canvas)
                f = D.font(17)
                bb = d.textbbox((0, 0), msg, font=f)
                d.text((px + (PANEL_W - bb[2]) / 2, py + PANEL_H / 2 - 10), msg,
                       font=f, fill=(255, 240, 235))
        proc.stdin.write(canvas.tobytes())

    ev = stats.get("eval_greedy", {})
    lg = stats.get("eval_long", {})
    lines = [f"{stats.get('episodes', '?')} episodes, "
             f"{stats.get('q_states_learned', '?')} of {stats.get('state_space', '?')} states ever visited",
             f"{stats.get('train_seconds', '?')} seconds of ordinary Python, no GPU"]
    if lg:
        lines.append(f"long horizon best: {lg.get('max_pipes')} pipes")
    outro = title_card(
        f"{ev.get('mean_pipes', '?')} pipes per run on {ev.get('n', '?')} unseen worlds",
        lines, bg=(20, 44, 40))
    for _ in range(90):
        proc.stdin.write(outro.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out}  ({total + intro + 90} frames, {(total + intro + 90) / FPS:.1f}s)")


if __name__ == "__main__":
    main()
