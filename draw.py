"""Drawing helpers shared by every renderer and by the learning curve.

Pillow is the only graphics dependency in the project. These helpers do the parts
that are easy to get subtly wrong: loading a font at many sizes, pasting a sprite
that has transparent edges, and fitting text into a column that is narrower than
the text wants to be.

The font cache is not a micro optimisation. A 4K video is about 2,200 frames, and
Pillow re-reads and re-parses a TTF file on every `truetype` call.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

import flappy as F

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Space Grotesk ships with the project; DejaVu is only a fallback
BRAND_FONTS = {
    "bold": (os.path.join(HERE, "assets", "fonts", "SpaceGrotesk-Bold.ttf"),
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    "medium": (os.path.join(HERE, "assets", "fonts", "SpaceGrotesk-Medium.ttf"),
               "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    "heavy": (os.path.join(HERE, "assets", "fonts", "Inter-ExtraBold.ttf"),
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
}

SKY_TOP = (124, 197, 214)
SKY_BOT = (176, 224, 230)
CLOUD = (238, 248, 250)
GROUND = (222, 216, 149)
GROUND_TOP = (196, 186, 118)
PIPE = (86, 178, 79)
PIPE_DARK = (44, 110, 46)
PIPE_LIGHT = (140, 214, 120)
BIRD = (255, 212, 60)
BIRD_DARK = (214, 158, 24)
BEAK = (240, 130, 40)
INK = (28, 34, 38)


_FONT_CACHE: dict = {}


def font(size: int, weight: str = "bold"):
    """Cached brand font. weight: bold (default), medium, heavy."""
    size = max(6, int(size))
    key = (weight, size)
    hit = _FONT_CACHE.get(key)
    if hit is not None:
        return hit
    f = None
    for path in BRAND_FONTS.get(weight, BRAND_FONTS["bold"]):
        if os.path.exists(path):
            f = ImageFont.truetype(path, size)
            break
    if f is None:
        for path in FONT_CANDIDATES:
            if os.path.exists(path):
                f = ImageFont.truetype(path, size)
                break
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:                  # very old Pillow
            f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def sky_gradient(w: int, h: int) -> Image.Image:
    img = Image.new("RGB", (w, h), SKY_BOT)
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
        g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
        b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


def _clouds(d: ImageDraw.ImageDraw, w: int, h: int, t: float, s: float) -> None:
    for i in range(3):
        cx = ((t * 0.25 + i * 47) % (w / s + 40)) * s
        cy = (12 + i * 13) * s
        for dx, dy, r in ((0, 0, 7), (7, -2, 5.5), (-7, 1, 4.5)):
            d.ellipse(
                [(cx + dx * s - r * s, cy + dy * s - r * s * 0.7),
                 (cx + dx * s + r * s, cy + dy * s + r * s * 0.9)],
                fill=CLOUD,
            )


def draw_scene(base: Image.Image, box: tuple[int, int, int, int], pipes, bird_y: float,
               t: int, s: float = 3.0) -> None:
    """Draw one game frame into `base` inside box=(x0,y0,x1,y1)."""
    x0, y0 = box[0], box[1]
    d = ImageDraw.Draw(base)
    _clouds(d, box[2] - box[0], box[3] - box[1], t, s)

    # ground
    gy = y0 + F.GROUND_Y * s
    d.rectangle([x0, gy, box[2], box[3]], fill=GROUND)
    d.rectangle([x0, gy, box[2], gy + 2 * s], fill=GROUND_TOP)

    # pipes
    for px, gap in pipes:
        left = x0 + px * s
        right = left + F.PIPE_W * s
        if right < x0 - 2 or left > box[2] + 2:
            continue
        top_end = y0 + (gap - F.PIPE_GAP / 2) * s
        bot_start = y0 + (gap + F.PIPE_GAP / 2) * s
        for (ya, yb) in ((y0, top_end), (bot_start, gy)):
            d.rectangle([left, ya, right, yb], fill=PIPE)
            d.rectangle([left, ya, left + 1.6 * s, yb], fill=PIPE_LIGHT)
            d.rectangle([right - 1.6 * s, ya, right, yb], fill=PIPE_DARK)
        # lips at the gap
        lip = 2.5 * s
        for (ya, yb) in ((top_end - lip, top_end), (bot_start, bot_start + lip)):
            d.rectangle([left - 1.5 * s, ya, right + 1.5 * s, yb], fill=PIPE)
            d.rectangle([left - 1.5 * s, ya, right + 1.5 * s, ya + 1.4 * s], fill=PIPE_LIGHT)
            d.rectangle([left - 1.5 * s, yb - 1.4 * s, right + 1.5 * s, yb], fill=PIPE_DARK)

    # bird
    bx = x0 + F.BIRD_X * s
    by = y0 + bird_y * s
    wing = -1.4 * s if (t // 5) % 2 else 0.6 * s
    d.ellipse([bx - 4.2 * s, by - 3.4 * s, bx + 4.0 * s, by + 3.2 * s], fill=BIRD)
    d.ellipse([bx - 3.6 * s, by + wing - 1.6 * s, bx - 0.2 * s, by + wing + 1.8 * s], fill=BIRD_DARK)
    d.ellipse([bx + 0.8 * s, by - 2.2 * s, bx + 3.4 * s, by + 0.4 * s], fill=(255, 255, 255))
    d.ellipse([bx + 1.9 * s, by - 1.5 * s, bx + 2.9 * s, by - 0.4 * s], fill=INK)
    d.polygon([(bx + 3.6 * s, by - 0.2 * s), (bx + 6.6 * s, by + 1.0 * s), (bx + 3.4 * s, by + 1.8 * s)], fill=BEAK)


def hud(base: Image.Image, xy: tuple[int, int], text: str, size: int = 15,
        fg=(255, 255, 255), bg=(24, 30, 34)) -> tuple[int, int]:
    """Draw a pill with text, return its right edge."""
    d = ImageDraw.Draw(base)
    f = font(size)
    x, y = xy
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = size * 0.45
    d.rounded_rectangle([x, y, x + tw + pad * 2, y + th + pad * 1.6], radius=size * 0.5, fill=bg)
    d.text((x + pad, y + pad * 0.5), text, font=f, fill=fg)
    return x + int(tw + pad * 2)


def fit_font(text: str, max_w: int, start: int = 20, min_size: int = 9, weight: str = "bold"):
    """Largest font size at which `text` fits inside max_w pixels."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in range(start, min_size - 1, -1):
        f = font(size, weight)
        if probe.textlength(text, font=f) <= max_w:
            return f
    return font(min_size, weight)


def text_width(text: str, size: int, weight: str = "bold") -> float:
    return ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=font(size, weight))
