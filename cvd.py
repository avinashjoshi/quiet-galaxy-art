#!/usr/bin/env python3
"""Check the palette in colors.toml under simulated protanopia.

    uv run --with numpy .art/cvd.py            # report on colors.toml
    uv run --with numpy .art/cvd.py --search   # look for a better hue set

Protanopia is simulated with the Machado, Oliveira & Fernandes (2009) matrix
at full severity, applied in linear RGB. Two things are measured on the
simulated colours, because those are what a protanope has to work with:

  contrast  WCAG contrast against the background (red light is dim to a
            protanope, so a red that passes for others can fail here)
  distance  OKLab distance between every pair of hues that carry meaning
            (0.02 is about one just-noticeable step)
"""

import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("THEME_DIR", Path(__file__).resolve().parent.parent / "omarchy-quiet-galaxy-theme"))

PROTAN = np.array(
    [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]
)

M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def to_linear(hex_colour):
    h = hex_colour.lstrip("#")
    c = np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_hex(linear):
    c = np.clip(linear, 0.0, 1.0)
    c = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def protan(linear):
    return np.clip(PROTAN @ linear, 0.0, 1.0)


def oklab(linear):
    return M2 @ np.cbrt(M1 @ linear)


def from_oklch(lightness, chroma, hue_degrees):
    a = chroma * np.cos(np.radians(hue_degrees))
    b = chroma * np.sin(np.radians(hue_degrees))
    lms = np.linalg.inv(M2) @ np.array([lightness, a, b])
    return np.linalg.inv(M1) @ (lms**3)


def luminance(linear):
    return float(np.array([0.2126, 0.7152, 0.0722]) @ linear)


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def distance(a, b):
    return float(np.linalg.norm(oklab(a) - oklab(b)))


def read_colours(path):
    colours = {}
    for line in path.read_text().splitlines():
        match = re.match(r'\s*([a-z_]+)\s*=\s*"(#[0-9a-fA-F]{6})"', line)
        if match:
            colours[match.group(1)] = match.group(2)
    return colours


# Hues that tell the reader something by being different from one another.
SIGNAL = ["foreground", "muted", "red", "orange", "yellow", "green", "cyan", "blue", "magenta"]
BRIGHT = ["bright_red", "bright_yellow", "bright_green", "bright_cyan", "bright_blue", "bright_magenta"]


def report(colours):
    background = to_linear(colours["background"])
    print(f"{'role':<16}{'hex':<10}{'protan':<10}{'contrast':>9}{'protan contrast':>17}")
    for name in SIGNAL + BRIGHT:
        linear = to_linear(colours[name])
        seen = protan(linear)
        print(
            f"{name:<16}{colours[name]:<10}{to_hex(seen):<10}"
            f"{contrast(linear, background):>8.1f}:1{contrast(seen, protan(background)):>15.1f}:1"
        )
    for label, names in (("normal", SIGNAL), ("bright", BRIGHT)):
        pairs = sorted(
            (distance(protan(to_linear(colours[a])), protan(to_linear(colours[b]))), a, b)
            for a, b in itertools.combinations(names, 2)
        )
        print(f"\nclosest {label} pairs under protanopia (OKLab distance):")
        for d, a, b in pairs[:5]:
            print(f"  {d:.3f}  {a} / {b}")


def search(colours, rounds=40000, steps=40000, seed=7):
    """Look for the six free hues: a random search, then hill-climbing from
    the best find. Yellow and the greys stay put."""
    rng = np.random.default_rng(seed)
    background = protan(to_linear(colours["background"]))
    fixed = {name: protan(to_linear(colours[name])) for name in ("foreground", "muted", "yellow")}
    # name: (hue range in OKLCH degrees, lightness range)
    slots = {
        "red": ((18, 42), (0.60, 0.76)),
        "orange": ((48, 72), (0.68, 0.82)),
        "green": ((140, 178), (0.68, 0.88)),
        "cyan": ((195, 232), (0.76, 0.92)),
        "blue": ((250, 272), (0.56, 0.74)),
        "magenta": ((300, 352), (0.60, 0.86)),
    }
    low = np.array([[l0, 0.08, h0] for (h0, _), (l0, _) in slots.values()])
    high = np.array([[l1, 0.24, h1] for (_, h1), (_, l1) in slots.values()])

    def score(lch):
        seen = dict(fixed)
        for name, (lightness, chroma, hue) in zip(slots, lch):
            linear = from_oklch(lightness, chroma, hue)
            if linear.min() < 0.0 or linear.max() > 1.0:
                return -1.0
            seen[name] = protan(linear)
            if contrast(seen[name], background) < 5.0:
                return -1.0
        return min(distance(seen[a], seen[b]) for a, b in itertools.combinations(seen, 2))

    best, best_score = None, -1.0
    for _ in range(rounds):
        candidate = rng.uniform(low, high)
        value = score(candidate)
        if value > best_score:
            best, best_score = candidate, value
    spread = (high - low) * 0.06
    for _ in range(steps):
        candidate = np.clip(best + rng.normal(0.0, 1.0, best.shape) * spread, low, high)
        value = score(candidate)
        if value > best_score:
            best, best_score = candidate, value

    print(f"best minimum protan distance: {best_score:.3f}")
    for name, (lightness, chroma, hue) in zip(slots, best):
        print(f'{name} = "{to_hex(from_oklch(lightness, chroma, hue))}"')


if __name__ == "__main__":
    palette = read_colours(ROOT / "colors.toml")
    if "--search" in sys.argv:
        search(palette)
    else:
        report(palette)
