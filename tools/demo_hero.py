"""Animated SVG heroes for the README, from a demo timeline (colours only, no audio).

Usage: uv run python tools/demo_hero.py docs/missyou.json docs [--still SECONDS]

Writes hero-wide.svg and hero-narrow.svg: a tube whose colour follows the
default mood through the whole track in a 40 s loop, the colour of every two
seconds as a band, a playhead, and the sections music mode heard. CSS
animation only, so GitHub shows it in a README; with reduced motion it holds
still on a breakdown moment (--still).
"""
import colorsys, json, sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
STILL = float(sys.argv[sys.argv.index("--still") + 1]) if "--still" in sys.argv else 150.0
cols, frames, dur = data["colours"]["ethereal"], data["frames"], data["duration"]
LOOP = 40.0  # seconds for the whole track: about nine times real speed
STEP = 2.0   # seconds of music per keyframe
NAMES = {"breakdown": "Breakdown", "build": "Build", "climax": "Drop"}


def rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def light(c):
    return rgb(c[0], c[1] / 100, 0.18 + 0.82 * c[2] / 100)


def wall(c):
    return rgb(c[0], c[1] / 100 * 0.7, 0.035 + 0.13 * c[2] / 100)


def at(t):
    return cols[max(0, min(len(cols) - 1, round(t * 10) - 1))]


def keyframes(name, prop, fn):
    stops, t = [], 0.0
    while t <= dur:
        stops.append(f"{t / dur * 100:.2f}%{{{prop}:{fn(at(t))}}}")
        t += STEP
    return f"@keyframes {name}{{{''.join(stops)}}}"


def sections():
    spans, start, cur = [], 0.0, frames[0][1]
    for f in frames[1:]:
        if f[1] != cur:
            spans.append((start, f[0], cur)); start, cur = f[0], f[1]
    spans.append((start, frames[-1][0], cur))
    # Short breakdowns (the kick out for a few bars) would crowd out the build and drop.
    return [s for s in spans if s[2] in NAMES and s[1] - s[0] >= (14 if s[2] == "breakdown" else 4)]


def svg(width, height, tube, band, label_y, font):
    x0, y0, bw, bh = band
    seg = []
    n = int(dur // STEP)
    for i in range(n):
        seg.append(f'<rect x="{x0 + i * bw / n:.2f}" y="{y0}" width="{bw / n + 0.6:.2f}" height="{bh}" fill="{light(at(i * STEP + STEP / 2))}"/>')
    labels = []
    last_end = -1e9
    for a, b, name in sections():
        x = x0 + a / dur * bw
        if x < last_end + 6:
            continue
        labels.append(f'<rect x="{x:.1f}" y="{label_y}" width="{max(3, (b - a) / dur * bw):.1f}" height="5" rx="2.5" fill="#eef0f6" fill-opacity="0.55"/>'
                      f'<text x="{x:.1f}" y="{label_y + 22}">{NAMES[name]}</text>')
        last_end = x + len(NAMES[name]) * font * 0.58
    tx, ty, tw, th = tube
    css = (keyframes("tube", "fill", light) + keyframes("wall", "fill", wall)
           + f"@keyframes head{{from{{transform:translateX(0)}}to{{transform:translateX({bw}px)}}}}"
           f".t{{animation:tube {LOOP}s linear infinite}}.w{{animation:wall {LOOP}s linear infinite}}"
           f".h{{animation:head {LOOP}s linear infinite}}"
           f"text{{font:500 {font}px -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;fill:#c3c8d6}}"
           # Still, at the start of the breakdown, for anyone who asked for less motion.
           f"@media (prefers-reduced-motion:reduce){{.t,.w,.h{{animation:none}}"
           f".t{{fill:{light(at(STILL))}}}.w{{fill:{wall(at(STILL))}}}.h{{transform:translateX({STILL / dur * bw:.1f}px)}}}}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<style>{css}</style>'
            f'<rect class="w" width="{width}" height="{height}" rx="14" fill="{wall(at(0))}"/>'
            f'<rect class="t" x="{tx}" y="{ty}" width="{tw}" height="{th}" rx="{min(tw, th) / 2}" fill="{light(at(0))}"/>'
            f'<g>{"".join(seg)}</g>'
            f'<rect class="h" x="{x0 - 1}" y="{y0 - 6}" width="2" height="{bh + 12}" rx="1" fill="#eef0f6"/>'
            f'{"".join(labels)}'
            f'</svg>')


out.mkdir(parents=True, exist_ok=True)
(out / "hero-wide.svg").write_text(svg(880, 240, (64, 32, 22, 176), (150, 56, 690, 72), 150, 14))
(out / "hero-narrow.svg").write_text(svg(400, 230, (24, 24, 16, 182), (64, 40, 312, 72), 134, 13))
for f in ("hero-wide.svg", "hero-narrow.svg"):
    print(f, (out / f).stat().st_size // 1024, "KB")
