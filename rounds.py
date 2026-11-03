"""Round robin: one training checkpoint per round, restart when the group dies.

    python3 rounds.py                 # 4K, 60 fps, ~38s
    python3 rounds.py --height 1080   # same layout at 1080p

A group of birds from one checkpoint starts. When the last one crashes, the next
checkpoint (more training) takes over with a fresh group. The run stops the moment
a single bird clears the 50th pipe.

Layout is designed in a 900-pixel-tall space and scaled to the output height, so
1080p and 4K are the same composition. The play field fills the frame height with
the labels living in the side margins, which keeps the game itself unobstructed.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import draw as D
import flappy as F
from agent import QAgent
from swarm import bird_sprite, hue_color, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
INK = (16, 20, 24)
PANEL = (18, 23, 27)
MUTED = (146, 157, 162)
WHITE = (248, 249, 246)
GREEN = (122, 228, 172)
GREEN_HOT = (176, 255, 205)
DEFAULT_SCHEDULE = "0,2000,4000,6000,8000,9000,10000,11000,12000"
NORMAL_STEPS_PER_SEC = 40.0

# set by setup()
SCALE = 24.0
FIELD_W = FIELD_H = CANVAS_W = CANVAS_H = MARGIN = 0
PX = 2.4

CLOUDS = (
    dict(x=0.11, y=0.08, w=0.26, drift=3.4),
    dict(x=0.49, y=0.05, w=0.32, drift=2.1),
    dict(x=0.80, y=0.15, w=0.20, drift=4.8),
    dict(x=0.31, y=0.20, w=0.15, drift=2.8),
)


def setup(height: int) -> None:
    global SCALE, FIELD_W, FIELD_H, CANVAS_W, CANVAS_H, MARGIN, PX
    FIELD_H = height
    SCALE = FIELD_H / F.H
    FIELD_W = int(F.W * SCALE)
    CANVAS_H = FIELD_H
    CANVAS_W = int(round(CANVAS_H * 16 / 9))
    MARGIN = (CANVAS_W - FIELD_W) // 2
    PX = CANVAS_H / 900.0


def lp(a, b, t: float):
    t = min(1.0, max(0.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


# ------------------------------------------------------------------ background
def background() -> Image.Image:
    ys = np.linspace(0.0, 1.0, FIELD_H)[:, None]
    top, bot = np.array(D.SKY_TOP, float), np.array(D.SKY_BOT, float)
    grad = top[None, :] + (bot - top)[None, :] * ys
    field = Image.fromarray(np.repeat(grad[:, None, :], FIELD_W, axis=1).astype("uint8"), "RGB")
    d = ImageDraw.Draw(field)
    gy = int(F.GROUND_Y * SCALE)
    d.rectangle([0, gy, FIELD_W, FIELD_H], fill=D.GROUND)
    d.rectangle([0, gy, FIELD_W, gy + int(2 * SCALE)], fill=D.GROUND_TOP)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), PANEL)
    img.paste(field, (MARGIN, 0))
    return img


def draw_clouds(d: ImageDraw.ImageDraw, seconds: float) -> None:
    for c in CLOUDS:
        w = c["w"] * FIELD_W
        h = w * 0.32
        span = FIELD_W + w
        cx = MARGIN + ((c["x"] * FIELD_W - c["drift"] * PX * seconds) % span) - w * 0.5
        cy = c["y"] * FIELD_H + h * 0.6
        for dx, dy, rw, rh in ((0.0, 0.0, 0.62, 0.62), (0.26, -0.16, 0.46, 0.5),
                               (-0.28, 0.08, 0.42, 0.46)):
            x0 = cx + dx * w - rw * w / 2
            y0 = cy + dy * h - rh * h / 2
            d.ellipse([x0, y0, x0 + rw * w, y0 + rh * h], fill=(243, 249, 251))


def draw_pipe(d: ImageDraw.ImageDraw, px_: float, gap: float) -> None:
    left = MARGIN + px_ * SCALE
    right = left + F.PIPE_W * SCALE
    if right < MARGIN - 4 or left > MARGIN + FIELD_W + 4:
        return
    top_end = (gap - F.PIPE_GAP / 2) * SCALE
    bot_start = (gap + F.PIPE_GAP / 2) * SCALE
    gy = F.GROUND_Y * SCALE
    for ya, yb in ((0.0, top_end), (bot_start, gy)):
        d.rectangle([left, ya, right, yb], fill=D.PIPE)
        d.rectangle([left, ya, left + 2.1 * SCALE, yb], fill=D.PIPE_LIGHT)
        d.rectangle([right - 2.1 * SCALE, ya, right, yb], fill=D.PIPE_DARK)
    lip = 2.6 * SCALE
    for ya, yb in ((top_end - lip, top_end), (bot_start, bot_start + lip)):
        d.rectangle([left - 1.6 * SCALE, ya, right + 1.6 * SCALE, yb], fill=D.PIPE)
        d.rectangle([left - 1.6 * SCALE, ya, right + 1.6 * SCALE, ya + 1.7 * SCALE], fill=D.PIPE_LIGHT)
        d.rectangle([left - 1.6 * SCALE, yb - 1.7 * SCALE, right + 1.6 * SCALE, yb], fill=D.PIPE_DARK)


def draw_ghost(d: ImageDraw.ImageDraw, x: float, y: float, pop: float) -> None:
    r = 4.6 * SCALE * (0.45 + 0.55 * pop)
    cx, cy = MARGIN + x * SCALE, y * SCALE
    col = lp((214, 206, 186), (120, 110, 96), pop)
    d.line([cx - r, cy - r, cx + r, cy + r], fill=col, width=max(2, int(0.5 * SCALE)))
    d.line([cx - r, cy + r, cx + r, cy - r], fill=col, width=max(2, int(0.5 * SCALE)))


# ------------------------------------------------------------------------ boxes
def blit_box(canvas: Image.Image, box, radius: float, fill, rows, alpha: float,
             pad: float = 0.0) -> None:
    """Rounded panel with centred text rows, alpha-blended onto the canvas."""
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                        fill=tuple(fill) + (int(255 * min(1.0, alpha)),))
    cy = h * 0.5 - sum(r[1] * 1.35 for r in rows) / 2
    for text, size, colour, dy in rows:
        f = D.font(size)
        bbox = d.textbbox((0, 0), text, font=f)
        d.text((w / 2, cy + (bbox[3] - bbox[1]) / 2 + dy), text, font=f,
               fill=tuple(colour) + (int(255 * min(1.0, alpha)),), anchor="mm")
        cy += size * 1.35
    region = canvas.crop((x0, y0, x1, y1)).convert("RGBA")
    canvas.paste(Image.alpha_composite(region, layer).convert("RGB"), (x0, y0))


# ------------------------------------------------------------------------- hud
def draw_hud(canvas: Image.Image, ctx, state, frame_in_round: int, alive_pulse: float,
             fast: bool) -> None:
    d = ImageDraw.Draw(canvas)
    appear = ease_out(min(1.0, (frame_in_round + 1) / 14.0))
    rise = (1 - appear) * PX * 26
    lx = MARGIN * 0.5
    rx = MARGIN + FIELD_W + MARGIN * 0.5
    dim = lp(PANEL, MUTED, appear)
    bright = lp(PANEL, WHITE, appear)

    d.text((lx, FIELD_H * 0.30 + rise), "checkpoint", font=D.font(PX * 20, "medium"),
           fill=dim, anchor="mm")
    ep_text = f"{ctx['ep']:,}"
    f_ep = D.fit_font(ep_text, int(MARGIN * 0.84), start=int(PX * 66), min_size=int(PX * 18))
    d.text((lx, FIELD_H * 0.30 + PX * 54 + rise), ep_text, font=f_ep, fill=bright, anchor="mm")
    d.text((lx, FIELD_H * 0.30 + PX * 108 + rise), "episodes",
           font=D.font(PX * 20, "medium"), fill=dim, anchor="mm")
    d.line([lx - PX * 60, FIELD_H * 0.30 + PX * 146 + rise,
            lx + PX * 60, FIELD_H * 0.30 + PX * 146 + rise],
           fill=lp(PANEL, (54, 62, 66), appear), width=max(1, int(PX * 1.5)))
    d.text((lx, FIELD_H * 0.30 + PX * 176 + rise), f"round {ctx['idx'] + 1} of {ctx['total']}",
           font=D.font(PX * 22, "medium"), fill=dim, anchor="mm")

    alive_col = lp(lp(PANEL, GREEN, appear), GREEN_HOT, alive_pulse)
    alive_text = f"{state['alive']}"
    f_alive = D.fit_font(alive_text, int(MARGIN * 0.84), start=int(PX * 66), min_size=int(PX * 18))
    d.text((rx, FIELD_H * 0.30 + rise), alive_text, font=f_alive,
           fill=alive_col, anchor="mm")
    d.text((rx, FIELD_H * 0.30 + PX * 54 + rise), f"of {ctx['group']} flying",
           font=D.font(PX * 20, "medium"), fill=dim, anchor="mm")
    d.line([rx - PX * 60, FIELD_H * 0.30 + PX * 92 + rise,
            rx + PX * 60, FIELD_H * 0.30 + PX * 92 + rise],
           fill=lp(PANEL, (54, 62, 66), appear), width=max(1, int(PX * 1.5)))
    d.text((rx, FIELD_H * 0.30 + PX * 122 + rise), f"best {state['best']} pipes",
           font=D.font(PX * 24), fill=bright, anchor="mm")
    d.text((rx, FIELD_H * 0.30 + PX * 158 + rise), f"target {ctx['target']} pipes",
           font=D.font(PX * 19, "medium"), fill=dim, anchor="mm")
    if fast:
        d.text((rx, FIELD_H * 0.30 + PX * 214 + rise), "6x speed",
               font=D.font(PX * 19, "medium"), fill=lp(PANEL, (196, 150, 96), appear),
               anchor="mm")

    # frame the field
    for x in (MARGIN - 1, MARGIN + FIELD_W):
        d.line([x, 0, x, CANVAS_H], fill=(38, 45, 50), width=max(1, int(PX * 1.2)))


# ----------------------------------------------------------------------- frames
def sample_state(frames: list, s: float) -> dict:
    i = int(s)
    a = s - i
    if i >= len(frames) - 1:
        return frames[-1]
    A, B = frames[i], frames[i + 1]
    pipes = [(px_ - F.PIPE_SPEED * a, gap) for px_, gap in A["pipes"]]
    nxt = {bi: (bx, by) for bi, bx, by in B["birds"]}
    birds = []
    for bi, bx, by in A["birds"]:
        if bi in nxt:
            nb = nxt[bi]
            birds.append((bi, bx + (nb[0] - bx) * a, by + (nb[1] - by) * a))
        else:
            birds.append((bi, bx, by))
    return {"pipes": pipes, "birds": birds, "alive": A["alive"], "best": A["best"],
            "step": A["step"] + a}


def sample_times(frames: list, fast_from, ff: int, fps: int) -> list[float]:
    normal = NORMAL_STEPS_PER_SEC / fps
    fast = normal * ff
    out, s, n = [], 0.0, len(frames) - 1
    while s < n:
        out.append(s)
        s += fast if (fast_from is not None and s >= fast_from) else normal
    out.append(float(n))
    return out


def compose(base: Image.Image, state: dict, ctx, frame_in_round: int, wing: int,
            alive_pulse: float, fast: bool, caption=None, final=None) -> Image.Image:
    canvas = base.copy()
    d = ImageDraw.Draw(canvas)
    seconds = state["step"] / NORMAL_STEPS_PER_SEC
    draw_clouds(d, seconds)
    for px_, gap in state["pipes"]:
        draw_pipe(d, px_, gap)
    for dstep, gx, gy in ctx["deaths"]:
        age = state["step"] - dstep
        if age >= 0:
            draw_ghost(d, gx, gy, min(1.0, 0.35 + age / 5.0))
    for _i, x, y in state["birds"]:
        sp = ctx["sprite"][wing]
        canvas.paste(sp, (int(MARGIN + x * SCALE - sp.width / 2),
                          int(y * SCALE - sp.height / 2)), sp)

    # wipe anything that spilled into the margins, then draw the labels
    d.rectangle([0, 0, MARGIN - 2, CANVAS_H], fill=PANEL)
    d.rectangle([MARGIN + FIELD_W + 2, 0, CANVAS_W, CANVAS_H], fill=PANEL)
    draw_hud(canvas, ctx, state, frame_in_round, alive_pulse, fast)

    if caption:
        text, alpha, rise_t = caption
        f = D.font(PX * 22)
        w = D.text_width(text, PX * 22) + PX * 44
        cx = CANVAS_W / 2
        cy = FIELD_H * 0.60 - (1 - ease_out(rise_t)) * PX * 26
        blit_box(canvas, (cx - w / 2, cy - PX * 30, cx + w / 2, cy + PX * 30),
                 PX * 16, (20, 26, 31), [(text, PX * 22, WHITE, 0)], alpha)

    if final:
        alpha, scale = final["alpha"], final["scale"]
        bw, bh = PX * 640 * scale, PX * 250 * scale
        cx, cy = CANVAS_W / 2, FIELD_H * 0.46
        rows = [(f"{final['target']} pipes", PX * 78, GREEN, -PX * 26),
                (final["lines"][0], PX * 24, WHITE, PX * 16),
                (final["lines"][1], PX * 22, (196, 208, 210), PX * 22)]
        blit_box(canvas, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                 PX * 26, (17, 40, 34), rows, alpha)
    return canvas


def intro_frame(t: int, total: int, lines: list[str]) -> Image.Image:
    bg = lp((0, 0, 0), INK, t / 26.0)
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    d = ImageDraw.Draw(img)
    a1 = ease_out((t - 10) / 44.0)
    d.text((CANVAS_W / 2, CANVAS_H * 0.40 - (1 - a1) * PX * 34),
           "one checkpoint per round", font=D.font(PX * 62),
           fill=lp(bg, WHITE, a1), anchor="mm")
    y = CANVAS_H * 0.40 + PX * 70
    for i, line in enumerate(lines):
        a = ease_out((t - 34 - i * 16) / 40.0)
        d.text((CANVAS_W / 2, y - (1 - a) * PX * 18), line,
               font=D.font(PX * 24, "medium"), fill=lp(bg, (150, 214, 180), a), anchor="mm")
        y += PX * 44
    if t > total - 20:
        k = (t - (total - 20)) / 20.0
        img = Image.blend(img, Image.new("RGB", img.size, (10, 14, 17)), k * 0.9)
    return img


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=12, help="birds per round")
    ap.add_argument("--spread", type=float, default=0.25, help="horizontal spacing, world px")
    ap.add_argument("--jitter", type=float, default=1.3, help="vertical start jitter, world px")
    ap.add_argument("--vjitter", type=float, default=0.15, help="vertical speed jitter")
    ap.add_argument("--seed", type=int, default=31337, help="level seed, same every round")
    ap.add_argument("--target", type=int, default=50, help="stop when one bird clears this pipe")
    ap.add_argument("--steps", type=int, default=2600, help="hard step cap per round")
    ap.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="checkpoints, comma separated")
    ap.add_argument("--ff", type=int, default=6, help="fast forward factor on long runs")
    ap.add_argument("--fast-pipes", type=int, default=3, help="fast forward after N pipes")
    ap.add_argument("--height", type=int, default=2160, help="output height (2160 = 4K UHD)")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--out", default=os.path.expanduser("~/videos/rl_flappy_rounds_4k.mp4"))
    ap.add_argument("--dry-run", action="store_true", help="simulate and report only")
    args = ap.parse_args()
    setup(args.height)

    schedule = [int(x) for x in args.schedule.split(",") if x.strip()]
    n = args.group
    rounds = []
    for r, ep in enumerate(schedule):
        path = os.path.join(HERE, "snapshots", f"ep_{ep:05d}.json")
        if not os.path.exists(path):
            raise SystemExit(f"missing checkpoint {path}")
        agent = QAgent.load(path)
        jrng = random.Random(args.seed * 7919 + r)
        y_off = [jrng.uniform(-args.jitter, args.jitter) for _ in range(n)]
        vy_off = [jrng.uniform(-args.vjitter, args.vjitter) for _ in range(n)]
        frames, stats = simulate([agent] * n, args.seed, args.steps, spread=args.spread,
                                 y_offsets=y_off, vy_offsets=vy_off, stop_at=args.target)
        best = max(stats["passed"])
        reached = best >= args.target
        rounds.append({"ep": ep, "frames": frames, "best": best, "reached": reached,
                       "deaths": sorted(stats["death_pos"])})
        print(f"  round {r + 1:2d}  checkpoint {ep:6,}  steps {len(frames) - 1:5d}  "
              f"best {best:3d} pipes  {'REACHED' if reached else 'wiped out'}", flush=True)
        if reached:
            break
    else:
        print("  schedule ran out before anyone cleared the target")

    INTRO = 195
    plans = []
    for r, row in enumerate(rounds):
        fast_from = next((i for i, fr in enumerate(row["frames"])
                          if fr["best"] >= args.fast_pipes), None)
        times = sample_times(row["frames"], fast_from, args.ff, args.fps)
        plans.append({"times": times, "fast_from": fast_from,
                      "color": hue_color(row["ep"] / (max(x["ep"] for x in rounds) or 1))})

    if args.dry_run:
        total = INTRO
        for r, row in enumerate(rounds):
            total += len(plans[r]["times"])
            total += 110 if (r == len(rounds) - 1 and row["reached"]) else 30
        print(f"  {CANVAS_W}x{CANVAS_H} @ {args.fps}fps: {total} frames, {total / args.fps:.1f}s")
        return

    base = background()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{CANVAS_W}x{CANVAS_H}", "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-preset", "faster", "-tune", "animation",
           "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    written = 0

    def emit(img: Image.Image) -> None:
        nonlocal written
        assert img.size == (CANVAS_W, CANVAS_H), img.size
        proc.stdin.write(img.tobytes())
        written += 1

    intro_lines = [f"a group of {n} birds starts, the next checkpoint takes over when they all crash",
                   f"the run ends the moment one bird clears {args.target} pipes"]
    for t in range(INTRO):
        emit(intro_frame(t, INTRO, intro_lines))

    prev_last = None
    for r, row in enumerate(rounds):
        plan = plans[r]
        ctx = {"ep": row["ep"], "idx": r, "total": len(rounds), "group": n,
               "target": args.target, "deaths": row["deaths"],
               "sprite": (bird_sprite(plan["color"], 0, SCALE * 0.9),
                          bird_sprite(plan["color"], 1, SCALE * 0.9))}
        frames = row["frames"]
        last_state = sample_state(frames, float(len(frames) - 1))
        last_img = None
        pulse = 0.0
        prev_alive = n
        for k, s in enumerate(plan["times"]):
            state = sample_state(frames, s)
            if state["alive"] != prev_alive:
                pulse = 1.0
                prev_alive = state["alive"]
            pulse = max(0.0, pulse - 1.0 / 12.0)
            fast = plan["fast_from"] is not None and s >= plan["fast_from"]
            img = compose(base, state, ctx, k, (k // 6) % 2, pulse, fast)
            if prev_last is not None and r > 0 and k < 10:
                img = Image.blend(prev_last, img, (k + 1) / 10.0)
            emit(img)
            last_img = img

        is_last = r == len(rounds) - 1
        if is_last and row["reached"]:
            lines = [f"checkpoint {row['ep']:,} episodes",
                     f"{last_state['alive']} of {n} birds still flying"]
            for k in range(110):
                alpha = ease_out(k / 22.0)
                scale = 0.88 + 0.12 * ease_out(k / 26.0)
                emit(compose(base, last_state, ctx, len(plan["times"]), 0, 0.0, False,
                             final={"alpha": alpha, "scale": scale,
                                    "target": args.target, "lines": lines}))
        else:
            text = (f"all {n} crashed, best was {row['best']} pipes" if row["best"]
                    else f"all {n} crashed before the first pipe")
            for k in range(30):
                emit(compose(base, last_state, ctx, len(plan["times"]), 0, 0.0, False,
                             caption=(text, ease_out(k / 10.0), min(1.0, k / 10.0))))
            prev_last = compose(base, last_state, ctx, len(plan["times"]), 0, 0.0, False,
                                caption=(text, 1.0, 1.0))

    proc.stdin.close()
    proc.wait()
    print(f"  wrote {args.out}  ({CANVAS_W}x{CANVAS_H}, {written} frames, "
          f"{written / args.fps:.1f}s, {os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
