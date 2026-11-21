#!/usr/bin/env python3
"""Generate the Quiet Galaxy theme artwork: hairline drawings on near-black.

    uv run --with numpy --with pillow generate.py [name ...]
    uv run --with numpy --with pillow generate.py stills
    uv run --with numpy --with pillow generate.py videos   # needs ffmpeg

Everything is procedural and original: no film stills, logos or third-party
images. Each wallpaper is the theme's darkest grey except for the few pixels
that draw the picture, and the share of drawn pixels is printed as it is
written. Colours are grey, crawl yellow and saber blue, so nothing relies on
seeing red; the one red blade keeps a white core and a red light enough to
survive protanopia.

Writes, next to this directory:
  extras/backgrounds/NN-name.png         stills
  extras/backgrounds/intros/NN-name.mp4  the same drawing assembling itself; Omarchy
                                  plays it when the theme or boot lands on
                                  that still
  extras/backgrounds/NN-*-live.mp4       looping video wallpapers
  unlock.png, preview-unlock.png  boot splash emblem and its preview

preview.png is a real screenshot; "protanopia-preview" re-renders it as a
protanope sees it.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("THEME_DIR", Path(__file__).resolve().parent.parent / "omarchy-quiet-galaxy-theme"))
W, H = 3840, 2160  # drawing units; stills are rendered 1:1
VIDEO_SCALE = 2 / 3  # videos are 2560x1440
FPS = 30

BASE = "#0b0b0c"  # darker_background in colors.toml
WINDOW = "#161617"  # background in colors.toml
GREY = "#8a8a8a"
PALE = "#c8c8c8"
WHITE = "#f5f5f5"
YELLOW = "#ffe81f"
BLUE = "#80a2f8"
RED = "#ff5a4a"  # 4.7:1 on the base grey under simulated protanopia


def rgb(value):
    value = value.lstrip("#")
    return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


class Canvas:
    """A black float image plus coordinate grids in drawing units, so the
    same drawing code renders a 4K still or a smaller video frame."""

    def __init__(self, scale=1.0, w=W, h=H):
        self.s = scale
        self.w, self.h = int(round(w * scale)), int(round(h * scale))
        py, px = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        self.x, self.y = px / scale, py / scale
        self.image = np.zeros((self.h, self.w, 3), dtype=np.float32)

    def paint(self, coverage, colour, level=1.0):
        self.image = np.maximum(self.image, (coverage * level)[..., None] * rgb(colour))

    def stroke(self, distance, width):
        """Anti-aliased coverage of a line `width` units wide."""
        return np.clip((width / 2.0 - distance) * self.s + 0.5, 0.0, 1.0)

    def ring(self, cx, cy, radius, width):
        return self.stroke(np.abs(np.hypot(self.x - cx, self.y - cy) - radius), width)

    def disc(self, cx, cy, radius):
        return np.clip((radius - np.hypot(self.x - cx, self.y - cy)) * self.s + 0.5, 0.0, 1.0)

    def segment(self, ax, ay, bx, by, width, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        dx, dy = bx - ax, by - ay
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        return self.stroke(np.hypot(x - (ax + t * dx), y - (ay + t * dy)), width), t

    def ellipse(self, cx, cy, a, b, degrees, width):
        """Stroke of a rotated ellipse, plus its local v coordinate (negative
        on the far side) for hiding the half that passes behind something."""
        angle = np.radians(degrees)
        dx, dy = self.x - cx, self.y - cy
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        f = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        gradient = np.sqrt((u / a**2) ** 2 + (v / b**2) ** 2) / np.maximum(f, 1e-6)
        return self.stroke(np.abs(f - 1.0) / np.maximum(gradient, 1e-6), width), v


def ease(p):
    """Smooth 0..1 ramp that starts and ends at rest."""
    p = float(np.clip(p, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


def window(p, start, end):
    return ease((p - start) / (end - start))


def to_bytes(image):
    image = np.maximum(image, rgb(BASE))  # ink over the base grey
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_still(image, name):
    drawn = np.count_nonzero(image.any(axis=2)) / (image.shape[0] * image.shape[1])
    path = ROOT / "extras" / "backgrounds" / name
    Image.fromarray(to_bytes(image), mode="RGB").save(path, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}  drawn pixels: {drawn:.3%}")


def save_video(frames, path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-",
        "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", "-an", str(path),
    ]  # fmt: skip
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    count = 0
    for frame in frames:
        encoder.stdin.write(to_bytes(frame).tobytes())
        count += 1
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {path}")
    print(f"wrote {path.relative_to(ROOT)}  {count / FPS:.1f}s, {path.stat().st_size / 1e6:.1f} MB")


# -- the drawings. Each takes p in 0..1: 1 is the finished still, and lower
# -- values are the moments of its intro.


def twin_suns_art(c, p=1.0, cx=W * 0.5, cy=H * 0.47, unit=78.0, width=2.6, span=9.0):
    """Two suns over a horizon. `unit` is the larger sun's radius. The horizon
    draws outward from the middle, then the suns rise from behind it."""
    horizon = cy + unit * 1.9
    half = max(unit * span * window(p, 0.0, 0.45), 1.0)
    line, t = c.segment(cx - half, horizon, cx + half, horizon, width)
    line *= np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0.0, 1.0) ** 0.6  # fade out at both ends
    c.paint(line, GREY)
    above = np.clip((horizon - width - c.y) * c.s, 0.0, 1.0)
    for dx, dy, radius, colour, start in ((-0.95, -0.25, 1.0, YELLOW, 0.15), (1.25, 0.85, 0.72, PALE, 0.30)):
        rest = cy + unit * dy
        sunk = horizon + unit * radius + width  # wholly below the horizon
        y = sunk + (rest - sunk) * window(p, start, 1.0)
        c.paint(c.ring(cx + unit * dx, y, unit * radius, width) * above, colour)
    return c.image


LIGHTSPEED = None


def lightspeed_streaks():
    global LIGHTSPEED
    if LIGHTSPEED is None:
        rng = np.random.default_rng(1977)
        LIGHTSPEED = [
            dict(
                angle=rng.uniform(0, 2 * np.pi),
                start=rng.uniform(0.16, 1.0),
                length=rng.uniform(0.05, 0.22),
                colour=BLUE if rng.random() < 0.22 else GREY,
                level=rng.uniform(0.35, 1.0),
                width=float(rng.integers(2, 5)) / 2.0,
                phase=rng.random(),
                speed=int(rng.integers(1, 3)),
            )
            for _ in range(150)
        ]
    return LIGHTSPEED


def draw_streaks(c, lines):
    """lines: (angle, start radius, length, colour, level, width), radii in
    units. Drawn at twice the size and averaged down for clean edges."""
    over = 2
    layers = {colour: Image.new("L", (c.w * over, c.h * over)) for colour in (GREY, BLUE)}
    draws = {colour: ImageDraw.Draw(layer) for colour, layer in layers.items()}
    k = c.s * over
    cx, cy = W / 2 * k, H / 2 * k
    for angle, start, length, colour, level, width in lines:
        if level <= 0.004:
            continue
        ends = [
            (cx + np.cos(angle) * start * k, cy + np.sin(angle) * start * k),
            (cx + np.cos(angle) * (start + length) * k, cy + np.sin(angle) * (start + length) * k),
        ]
        draws[colour].line(ends, fill=int(255 * level), width=max(1, int(round(width * k))))
    for colour, layer in layers.items():
        c.paint(np.asarray(layer.reduce(over), dtype=np.float32) / 255.0, colour)
    return c.image


def lightspeed_art(c, p=1.0):
    """Stars stretch into streaks: the jump, frozen at the end."""
    reach = np.hypot(W / 2, H / 2)
    stretch = window(p, 0.15, 1.0)
    lines = []
    for s in lightspeed_streaks():
        start = reach * s["start"]
        length = 5.0 + (start * s["length"] - 5.0) * stretch
        lines.append((s["angle"], start, length, s["colour"], s["level"] * window(p, 0.0, 0.2), s["width"]))
    return draw_streaks(c, lines)


def lightspeed_live(c, t):
    """t in 0..1 over one loop. Each streak runs from near the vanishing
    point to past the corners, then starts again unseen."""
    reach = np.hypot(W / 2, H / 2)
    inner, outer = 0.10 * reach, 1.2 * reach
    lines = []
    for s in lightspeed_streaks()[:120]:
        u = (s["speed"] * t + s["phase"]) % 1.0
        start = inner * (outer / inner) ** u
        fade = ease(u / 0.18)
        lines.append((s["angle"], start, start * s["length"], s["colour"], s["level"] * fade, s["width"]))
    return draw_streaks(c, lines)


def star_points(count=340):
    rng = np.random.default_rng(1980)
    xs, ys = rng.integers(0, W - 2, count), rng.integers(0, H - 2, count)
    level = (0.25 + 0.75 * rng.random(count) ** 2.2).astype(np.float32)
    tint = np.stack([rgb(v) for v in (PALE, PALE, PALE, PALE, PALE, PALE, YELLOW, BLUE)])
    colour = tint[rng.integers(0, len(tint), count)]
    onset, phase, rate = rng.random(count) * 0.8, rng.random(count), rng.integers(1, 4, count)
    return xs, ys, level, colour, onset, phase, rate


def draw_stars(c, gain):
    xs, ys, level, colour, *_ = star_points()
    px = np.minimum((xs * c.s).astype(int), c.w - 2)
    py = np.minimum((ys * c.s).astype(int), c.h - 2)
    value = colour * (level * gain)[:, None]
    c.image[py, px] = value
    big = level > 0.8  # the brightest get a 2x2 dot
    for dx, dy in ((1, 0), (0, 1), (1, 1)):
        c.image[py[big] + dy, px[big] + dx] = value[big]
    return c.image


def starfield_art(c, p=1.0):
    """A few hundred single-pixel stars, coming out one by one."""
    onset = star_points()[4]
    return draw_stars(c, np.clip((p - onset) / 0.2, 0.0, 1.0).astype(np.float32))


def starfield_live(c, t):
    *_, phase, rate = star_points()
    twinkle = 0.62 + 0.38 * np.sin(2 * np.pi * (rate * t + phase))
    return draw_stars(c, twinkle.astype(np.float32))


def draw_saber(c, ox, oy, degrees, colour, p=1.0, blade=1069.0, shrouded=False):
    """A hairline hilt at (ox, oy) pointing along `degrees`, and a thin line
    of light. The hilt fades in, the blade runs out, and its glow settles.
    `shrouded` swaps the plain emitter for a slanted one over a ribbed grip."""
    angle = np.radians(degrees)
    dx, dy = c.x - ox, c.y - oy
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    hilt, half, width = 192.0, 13.0, 2.2

    reach = blade * window(p, 0.25, 0.65)
    if reach > 1.0:
        root, tip = hilt + 6.0, hilt + 6.0 + reach
        core, _ = c.segment(root, 0.0, tip, 0.0, 3.0, x=u, y=v)
        d = np.where((u >= root) & (u <= tip), np.abs(v), np.hypot(u - tip, v))
        d = np.where(u < root, 1e6, d)
        flash = float(np.exp(-(((p - 0.62) / 0.14) ** 2))) if p < 1.0 else 0.0
        halo = np.exp(-((d / (5.0 + 9.0 * flash)) ** 2)) * min(1.0, 0.75 + 0.25 * flash)
        halo[halo < 0.03] = 0.0  # cut off where it would only lift the base grey
        c.paint(halo, colour)
        c.paint(core, WHITE)

    shown = window(p, 0.0, 0.2)
    lines = [(0.0, -half, hilt, -half), (0.0, half, hilt, half), (0.0, -half, 0.0, half)]
    if shrouded:
        lines += [(hilt, -half, hilt + 30.0, half), (hilt, half, hilt + 30.0, half)]  # slanted emitter
        lines += [(rib, -half, rib, half) for rib in np.arange(22.0, 100.0, 11.0)]  # ribbed grip
        lines += [(112.0, -half - 9, 150.0, -half - 9), (112.0, -half - 9, 112.0, -half), (150.0, -half - 9, 150.0, -half)]
    else:
        lines += [(hilt, -half - 5, hilt, half + 5)]
        lines += [(rib, -half, rib, half) for rib in (30.0, 44.0, 58.0)]
    for ax, ay, bx, by in lines:
        c.paint(c.segment(ax, ay, bx, by, width, x=u, y=v)[0], GREY, shown)
    stud = c.stroke(np.abs(np.hypot(u - (hilt - 34.0), v) - 4.5), width)  # activation stud
    c.paint(stud, YELLOW, shown)


def saber_art(c, p=1.0):
    draw_saber(c, W * 0.335, H * 0.5, 0.0, BLUE, p)
    return c.image


def crossed_blades_art(c, p=1.0):
    """Two blades meeting, one blue and one yellow: a pair protanopia keeps."""
    cx, cy, meet = W * 0.5, H * 0.45, 700.0
    for degrees, colour, delay in ((-56.0, BLUE, 0.0), (-124.0, YELLOW, 0.12)):
        angle = np.radians(degrees)
        local = 1.0 if p >= 1.0 else np.clip((p - delay) / (1.0 - 0.12), 0.0, 0.999)
        draw_saber(c, cx - np.cos(angle) * meet, cy - np.sin(angle) * meet, degrees, colour, local, blade=980.0)
    clash = window(p, 0.6, 0.8)
    c.paint(c.disc(cx, cy, 5.0), WHITE, clash)
    return c.image


def dark_side_art(c, p=1.0):
    """One red blade, upright."""
    draw_saber(c, W * 0.5, H * 0.80, -90.0, RED, p, shrouded=True)
    return c.image


def no_moon_art(c, p=1.0):
    """A battle station the size of a small moon: a sphere, the trench around
    its middle, and the dish above it."""
    cx, cy, radius = W * 0.5, H * 0.47, 250.0
    d = np.hypot(c.x - cx, c.y - cy)
    inside = np.clip((radius - d) * c.s, 0.0, 1.0)
    c.paint(c.ring(cx, cy, radius, 2.6), PALE)
    for offset in (-7.0, 7.0):  # the trench, bowed as if seen from just above
        line, v = c.ellipse(cx, cy + offset, radius, radius * 0.07, 0.0, 2.0)
        c.paint(line * (v > 0) * inside, GREY)
    dish_x, dish_y = cx - radius * 0.40, cy - radius * 0.44
    c.paint(c.ellipse(dish_x, dish_y, 62.0, 52.0, -38.0, 2.2)[0], PALE)
    c.paint(c.ellipse(dish_x - 3.0, dish_y - 3.0, 22.0, 18.0, -38.0, 1.8)[0], GREY)
    c.paint(c.disc(dish_x - 5.0, dish_y - 5.0, 3.5), YELLOW)
    return reveal(c, p, cx, cy, radius + 140.0)


def reveal(c, p, cx, cy, radius):
    """Uncover a finished drawing outward from (cx, cy)."""
    if p >= 1.0:
        return c.image
    d = np.hypot(c.x - cx, c.y - cy)
    c.image *= np.clip((window(p, 0.0, 0.95) * radius - d) / 90.0 + 0.5, 0.0, 1.0)[..., None]
    return c.image


def dune_sea_art(c, p=1.0):
    """Three ridgelines, the near ones hiding the far, under two small suns."""
    edge_fade = np.clip(1.0 - np.abs(c.x / W - 0.5) * 2.0, 0.0, 1.0) ** 0.5
    ridges = [  # far to near: base, amplitude, wavelengths, phases, brightness
        (H * 0.585, 34.0, (2900.0, 1300.0), (0.4, 2.1), 0.45),
        (H * 0.630, 52.0, (3600.0, 1700.0), (2.6, 0.3), 0.70),
        (H * 0.700, 78.0, (4400.0, 2100.0), (5.1, 1.2), 1.00),
    ]
    curves = []
    for base, amp, (l1, l2), (f1, f2), _ in ridges:
        a1, a2 = 2 * np.pi / l1, 2 * np.pi / l2
        height = base + amp * (0.62 * np.sin(c.x * a1 + f1) + 0.38 * np.sin(c.x * a2 + f2))
        slope = amp * (0.62 * a1 * np.cos(c.x * a1 + f1) + 0.38 * a2 * np.cos(c.x * a2 + f2))
        curves.append((height, slope))
    for index, (_, _, _, _, level) in enumerate(ridges):
        height, slope = curves[index]
        line = c.stroke(np.abs(c.y - height) / np.sqrt(1.0 + slope**2), 2.4) * edge_fade
        for nearer, _ in curves[index + 1 :]:
            line *= np.clip((nearer - c.y) * c.s, 0.0, 1.0)  # hidden behind a nearer dune
        c.paint(line, GREY, level)
    c.paint(c.ring(W * 0.605, H * 0.462, 36.0, 2.4), YELLOW)
    c.paint(c.ring(W * 0.638, H * 0.500, 25.0, 2.4), PALE)
    return reveal(c, p, W * 0.5, H * 0.58, W * 0.56)


def ringed_planet_art(c, p=1.0):
    cx, cy, radius = W * 0.5, H * 0.47, 150.0
    c.paint(c.ring(cx, cy, radius, 2.6), PALE)
    inside = np.hypot(c.x - cx, c.y - cy) < radius
    for a, b, level in ((345.0, 80.0, 1.0), (292.0, 67.0, 0.55)):
        line, v = c.ellipse(cx, cy, a, b, -16.0, 2.2)
        line[inside & (v < 0)] = 0.0  # the far side passes behind the planet
        c.paint(line, GREY, level)
    c.paint(c.ring(cx + 560.0, cy - 215.0, 13.0, 2.2), YELLOW)  # a moon
    return reveal(c, p, cx, cy, 760.0)


def star_chart_art(c, p=1.0):
    """A navigation chart: orbits around a star, and a plotted route out."""
    cx, cy, tilt = W * 0.5, H * 0.48, -10.0
    c.paint(c.ring(cx, cy, 15.0, 2.4), YELLOW)
    c.paint(c.disc(cx, cy, 4.0), YELLOW)
    angle = np.radians(tilt)
    worlds = [  # orbit radius, position along it in degrees, dot radius, colour
        (150.0, 200.0, 4.5, PALE),
        (265.0, 320.0, 6.5, PALE),
        (395.0, 75.0, 5.5, BLUE),
        (545.0, 150.0, 9.0, PALE),
        (725.0, 15.0, 6.0, PALE),
    ]
    for a, where, dot, colour in worlds:
        b = a * 0.36
        c.paint(c.ellipse(cx, cy, a, b, tilt, 1.8)[0], GREY, 0.5)
        u, v = a * np.cos(np.radians(where)), b * np.sin(np.radians(where))
        px, py = cx + u * np.cos(angle) - v * np.sin(angle), cy + u * np.sin(angle) + v * np.cos(angle)
        c.paint(c.disc(px, py, dot), colour)
        if colour == BLUE:  # the route leaves from the blue world
            line, t = c.segment(px, py, px + 1150.0, py - 520.0, 2.0)
            dashes = (np.floor(t * 46.0) % 2 == 0) & (t > 0.03)
            c.paint(line * dashes * np.clip(1.0 - t, 0.0, 1.0) ** 0.5, BLUE)
            c.paint(c.ring(px, py, dot + 9.0, 1.8), BLUE)
    for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):  # corner brackets
        x0, y0 = cx + sx * 900.0, cy + sy * 400.0
        c.paint(c.segment(x0, y0, x0 - sx * 46.0, y0, 2.0)[0], GREY, 0.7)
        c.paint(c.segment(x0, y0, x0, y0 - sy * 46.0, 2.0)[0], GREY, 0.7)
    return reveal(c, p, cx, cy, 1250.0)


STILLS = {  # name: (file stem, drawing, intro seconds)
    "twin-suns": ("01-twin-suns", twin_suns_art, 3.2),
    "lightspeed": ("02-lightspeed", lightspeed_art, 2.6),
    "starfield": ("03-starfield", starfield_art, 3.0),
    "saber": ("04-saber", saber_art, 2.4),
    "crossed-blades": ("05-crossed-blades", crossed_blades_art, 2.8),
    "dark-side": ("06-dark-side", dark_side_art, 2.4),
    "no-moon": ("07-no-moon", no_moon_art, 2.4),
    "dune-sea": ("08-dune-sea", dune_sea_art, 2.6),
    "ringed-planet": ("09-ringed-planet", ringed_planet_art, 2.4),
    "star-chart": ("10-star-chart", star_chart_art, 2.8),
}

LIVE = {  # name: (file stem, drawing, loop seconds)
    "lightspeed-live": ("11-lightspeed-live", lightspeed_live, 8.0),
    "starfield-live": ("12-starfield-live", starfield_live, 12.0),
}


def still(name):
    stem, art, _ = STILLS[name]
    save_still(art(Canvas()), f"{stem}.png")


def intro(name):
    stem, art, seconds = STILLS[name]
    size = Canvas(VIDEO_SCALE)
    count = int(seconds * FPS)
    hold = FPS // 2  # rest on the finished drawing before Omarchy swaps in the still

    def frames():
        for index in range(count):
            yield art(Canvas(VIDEO_SCALE), index / count)
        final = art(Canvas(VIDEO_SCALE), 1.0)
        for _ in range(hold):
            yield final

    save_video(frames(), ROOT / "extras" / "backgrounds" / "intros" / f"{stem}.mp4", (size.w, size.h))


def live(name):
    stem, art, seconds = LIVE[name]
    size = Canvas(VIDEO_SCALE)
    count = int(seconds * FPS)
    frames = (art(Canvas(VIDEO_SCALE), index / count) for index in range(count))
    save_video(frames, ROOT / "extras" / "backgrounds" / f"{stem}.mp4", (size.w, size.h))


def emblem():
    """unlock.png: the twin suns on transparency, for the boot splash."""
    size, scale = 420, 4
    c = Canvas(w=size * scale, h=size * scale)
    centre = size * scale / 2
    image = twin_suns_art(c, cx=centre, cy=centre - 98, unit=300.0, width=3.0 * scale, span=2.7)
    alpha = image.max(axis=2)
    colour = np.where(alpha[..., None] > 0, image / np.maximum(alpha[..., None], 1e-6), 0.0)
    rgba = (np.dstack([colour, alpha]) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").resize((size, size), Image.LANCZOS).save(ROOT / "unlock.png", optimize=True)
    print("wrote unlock.png")


def unlock_preview():
    """preview-unlock.png: the boot splash as the unlock picker shows it."""
    scale = 2
    w, h = 1920 * scale, 1080 * scale
    page = Image.new("RGB", (w, h), WINDOW)
    logo = Image.open(ROOT / "unlock.png").convert("RGBA")
    logo = logo.resize((logo.width * scale, logo.height * scale), Image.LANCZOS)
    page.paste(logo, ((w - logo.width) // 2, (h - logo.height) // 2), logo)
    draw = ImageDraw.Draw(page)
    text = tuple(int(v * 255) for v in rgb(PALE))
    # Password field with four typed dots, and a padlock to its left.
    draw.rectangle([818 * scale, 770 * scale, 1102 * scale, 816 * scale], outline=text, width=2 * scale)
    for i in range(4):
        cx, cy = (840 + i * 12) * scale, 793 * scale
        draw.ellipse([cx - 3 * scale, cy - 3 * scale, cx + 3 * scale, cy + 3 * scale], fill=text)
    draw.rounded_rectangle([769 * scale, 789 * scale, 801 * scale, 813 * scale], radius=4 * scale, fill=text)
    draw.arc([775 * scale, 775 * scale, 795 * scale, 801 * scale], 180, 360, fill=text, width=4 * scale)
    draw.ellipse([782 * scale, 797 * scale, 788 * scale, 803 * scale], fill=WINDOW)
    page.resize((1920, 1080), Image.LANCZOS).save(ROOT / "preview-unlock.png", optimize=True)
    print("wrote preview-unlock.png")


def protanopia_preview():
    """docs/preview-protanopia.png: preview.png through the same simulation
    cvd.py uses, to show what survives without red."""
    from cvd import PROTAN

    data = np.asarray(Image.open(ROOT / "preview.png").convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(data <= 0.04045, data / 12.92, ((data + 0.055) / 1.055) ** 2.4)
    seen = np.clip(linear @ PROTAN.T, 0.0, 1.0)
    out = np.where(seen <= 0.0031308, seen * 12.92, 1.055 * seen ** (1 / 2.4) - 0.055)
    Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(
        ROOT / "docs" / "preview-protanopia.png", optimize=True
    )
    print("wrote docs/preview-protanopia.png")


def contact_sheet():
    """docs/wallpapers.png: every still side by side, for the README."""
    tile_w, tile_h, gap, columns = 640, 360, 16, 5
    names = list(STILLS)
    rows = -(-len(names) // columns)
    sheet = Image.new("RGB", (columns * tile_w + (columns + 1) * gap, rows * tile_h + (rows + 1) * gap), WINDOW)
    for index, name in enumerate(names):
        tile = Image.open(ROOT / "extras" / "backgrounds" / f"{STILLS[name][0]}.png")
        # Crop to the middle before shrinking, so hairlines survive at this size.
        tile = tile.crop((W // 4, H // 4, W * 3 // 4, H * 3 // 4)).resize((tile_w, tile_h), Image.LANCZOS)
        sheet.paste(tile, (gap + (index % columns) * (tile_w + gap), gap + (index // columns) * (tile_h + gap)))
    sheet.save(ROOT / "docs" / "wallpapers.png", optimize=True)
    print("wrote docs/wallpapers.png")


OTHER = {
    "emblem": emblem,
    "unlock-preview": unlock_preview,
    "protanopia-preview": protanopia_preview,
    "contact-sheet": contact_sheet,
}

if __name__ == "__main__":
    (ROOT / "extras" / "backgrounds").mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or ["stills"]:
        if name == "stills":
            for each in STILLS:
                still(each)
        elif name == "videos":
            for each in STILLS:
                intro(each)
            for each in LIVE:
                live(each)
        elif name in STILLS:
            still(name)
        elif name.removesuffix("-intro") in STILLS:
            intro(name.removesuffix("-intro"))
        elif name in LIVE:
            live(name)
        else:
            OTHER[name]()
