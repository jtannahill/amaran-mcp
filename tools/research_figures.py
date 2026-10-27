"""Figures for docs/research.md: static SVG, one light and one dark copy of each.

Usage: uv run python tools/research_figures.py --out docs/figures [--before-colours OLD_TIMELINE.json]

Draws from the numbers in research.md and the demo timelines in docs/:
  models-*.svg     held-out scores of the section models, one panel per measure
  tempo-*.svg      the tempo confidence threshold sweep on GiantSteps
  missyou-*.svg    Miss You: the light, what music mode heard, what the human expert marked
  colour-*.svg     the default mood before and after the colour change (needs --before-colours)
Colours: categorical slots 1 to 3 of the dataviz reference palette, validated on
GitHub's own light (#ffffff) and dark (#0d1117) surfaces.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#1f2328", "soft": "#59636e", "muted": "#818b98", "grid": "#e6e8eb",
              "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "rest": "#c4c9cf", "tick": "#1f2328"},
    "dark": {"surface": "#0d1117", "ink": "#f0f6fc", "soft": "#9198a1", "muted": "#7d8590", "grid": "#262c36",
             "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "rest": "#4a525c", "tick": "#f0f6fc"},
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

MODELS = ["Hand-written rules", "Trees", "Spectrogram CNN", "Blend (shipped)"]
SCORES = {  # held out, pooled over 402 SSME tracks (research.md, "Results")
    "Right section, balanced": [0.364, 0.482, 0.436, 0.486],
    "Breakdown starts (F)": [0.016, 0.045, 0.106, 0.090],
    "Build starts (F)": [0.000, 0.083, 0.066, 0.073],
    "Drops (F)": [0.095, 0.186, 0.151, 0.200],
}
SWEEP = {  # GiantSteps Accuracy 1, tracks 100 to 150 BPM, by confidence threshold
    "thresholds": [0.55, 0.40, 0.30, 0.25, 0.20],
    "Trance": [75.0, 95.6, 98.5, 97.1, 97.1],
    "House": [32.9, 57.5, 78.1, 86.3, 87.7],
    "All": [39.1, 60.5, 73.9, 78.6, 84.4],
}


def svg(width: int, height: int, body: str, theme: dict, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{title}"><title>{title}</title>'
            f'<style>text{{font-family:{FONT};fill:{theme["soft"]}}} .ink{{fill:{theme["ink"]}}} .muted{{fill:{theme["muted"]}}}</style>'
            f'<rect width="{width}" height="{height}" fill="{theme["surface"]}"/>{body}</svg>')


def models(theme: dict, narrow: bool) -> str:
    cols = 2 if narrow else 4
    panel_w, panel_h, label_w = (190, 150, 0) if narrow else (200, 150, 0)
    names_w = 0 if narrow else 132
    width = names_w + cols * panel_w + 24
    rows = (len(SCORES) + cols - 1) // cols
    height = rows * panel_h + 16
    out = []
    for p, (measure, values) in enumerate(SCORES.items()):
        cx = names_w + (p % cols) * panel_w + 12
        cy = (p // cols) * panel_h + 12
        top = max(values) * 1.25
        out.append(f'<text x="{cx}" y="{cy + 14}" font-size="13" font-weight="600" class="ink">{measure}</text>')
        for i, (model, value) in enumerate(zip(MODELS, values)):
            y = cy + 30 + i * 26
            bar = max(2.0, (panel_w - 64) * value / top)
            colour = theme["s1"] if model == "Blend (shipped)" else theme["rest"]
            out.append(f'<rect x="{cx}" y="{y}" width="{bar:.1f}" height="16" rx="4" fill="{colour}"/>')
            out.append(f'<text x="{cx + bar + 6:.1f}" y="{y + 12}" font-size="12" class="ink">{value:.2f}</text>')
            if narrow:
                pass
            elif p == 0:
                out.append(f'<text x="{names_w - 8}" y="{y + 12}" font-size="12" text-anchor="end" class="ink">{model}</text>')
            elif p % cols == 0:
                out.append(f'<text x="{names_w - 8}" y="{y + 12}" font-size="12" text-anchor="end" class="ink">{model}</text>')
    if narrow:  # a key instead of row names
        height += 40
        for i, model in enumerate(MODELS):
            colour = theme["s1"] if model == "Blend (shipped)" else theme["rest"]
            x = 12 + (i % 2) * 190
            y = height - 30 + (i // 2) * 16
            out.append(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{colour}"/>'
                       f'<text x="{x + 16}" y="{y}" font-size="12" class="ink">{i + 1}. {model}</text>')
    return svg(width, height, "".join(out), theme, "Held-out scores of the section models")


def tempo(theme: dict, narrow: bool) -> str:
    width, height = (400, 300) if narrow else (720, 320)
    left, right, top, bottom = 44, (86 if narrow else 120), 20, 44
    xs = SWEEP["thresholds"]
    def px(i: int) -> float:
        return left + i * (width - left - right) / (len(xs) - 1)
    def py(v: float) -> float:
        return top + (100 - v) * (height - top - bottom) / 70  # 30% to 100%
    out = []
    for v in (40, 60, 80, 100):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="{theme["grid"]}" stroke-width="1"/>'
                   f'<text x="{left - 8}" y="{py(v) + 4:.1f}" font-size="12" text-anchor="end" class="muted">{v}%</text>')
    for i, t in enumerate(xs):
        out.append(f'<text x="{px(i):.1f}" y="{height - bottom + 20}" font-size="12" text-anchor="middle" class="muted">{t:.2f}</text>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 6}" font-size="12" text-anchor="middle" class="muted">confidence needed to lock a tempo</text>')
    chosen = xs.index(0.25)
    out.append(f'<line x1="{px(chosen):.1f}" x2="{px(chosen):.1f}" y1="{top}" y2="{height - bottom}" stroke="{theme["muted"]}" stroke-width="1" stroke-dasharray="3 3"/>'
               f'<text x="{px(chosen) + 6:.1f}" y="{height - bottom - 8}" font-size="12" class="ink">chosen</text>')
    # End labels, nudged apart so none touch.
    ends = sorted(((py(SWEEP[n][-1]), n) for n in ("Trance", "House", "All")))
    placed = {}
    for y, n in ends:
        y = max(y, max(placed.values(), default=-99) + 15)
        placed[n] = y
    for name, key in (("Trance", "s1"), ("House", "s2"), ("All", "s3")):
        values = SWEEP[name]
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
        out.append(f'<polyline points="{points}" fill="none" stroke="{theme[key]}" stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(values):
            out.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="{theme[key]}" stroke="{theme["surface"]}" stroke-width="2"/>')
        out.append(f'<text x="{px(len(xs) - 1) + 10:.1f}" y="{placed[name] + 4:.1f}" font-size="12" class="ink">{name} {values[-1]:.0f}%</text>')
    return svg(width, height, "".join(out), theme, "Tempo accuracy by confidence threshold, GiantSteps")


def rgb(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def light(c: list) -> str:
    return rgb(c[0], c[1] / 100, 0.18 + 0.82 * c[2] / 100)


def band(x: float, y: float, w: float, h: float, colours: list, step: int = 10) -> str:
    n = len(colours)
    parts = []
    for i in range(0, n, step):
        parts.append(f'<rect x="{x + i * w / n:.2f}" y="{y}" width="{w * step / n + 0.6:.2f}" height="{h}" fill="{light(colours[min(n - 1, i + step // 2)])}"/>')
    return f'<g>{"".join(parts)}</g>'


def missyou(theme: dict, narrow: bool) -> str:
    data = json.loads((DOCS / "missyou.json").read_text())
    frames, colours, dur = data["frames"], data["colours"]["ethereal"], data["duration"]
    width = 400 if narrow else 880
    left, right = (12, 12) if narrow else (130, 16)
    w = width - left - right
    x = lambda t: left + t / dur * w  # noqa: E731
    out = [band(left, 12, w, 44, colours)]
    # What music mode heard: breakdowns, builds and drops, filled with the light's average colour.
    spans, start, cur = [], 0.0, frames[0][1]
    for f in frames[1:]:
        if f[1] != cur:
            spans.append((start, f[0], cur)); start, cur = f[0], f[1]
    spans.append((start, frames[-1][0], cur))
    names = {"breakdown": "Breakdown", "build": "Build", "climax": "Drop"}
    row1, row2 = 92, 150
    if not narrow:
        out.append(f'<text x="{left - 12}" y="{row1 + 10}" font-size="12" text-anchor="end" class="ink">Music mode heard</text>')
        out.append(f'<text x="{left - 12}" y="{row2 + 10}" font-size="12" text-anchor="end" class="ink">Human expert</text>')
    else:
        out.append(f'<text x="{left}" y="{row1 - 8}" font-size="12" class="ink">Music mode heard</text>')
        out.append(f'<text x="{left}" y="{row2 - 8}" font-size="12" class="ink">Human expert, best reading</text>')
    for a, b, name in spans:
        if name not in names:
            continue
        mid = colours[min(len(colours) - 1, int((a + b) / 2 * 10))]
        out.append(f'<rect x="{x(a):.1f}" y="{row1}" width="{max(2, x(b) - x(a) - 2):.1f}" height="12" rx="3" fill="{light(mid)}"/>')
        if x(b) - x(a) > 45:
            out.append(f'<text x="{x(a):.1f}" y="{row1 + 28}" font-size="12">{names[name]}</text>')
    blooms = [f[0] for f in frames if f[5]]
    for t in [b for i, b in enumerate(blooms) if i == 0 or b - blooms[i - 1] > 4]:
        out.append(f'<rect x="{x(t) - 1:.1f}" y="{row1 - 5}" width="2" height="22" rx="1" fill="{theme["tick"]}"/>')
    # The human expert's marks, read into sections (data["expert"]): neutral spans, named moments as ticks.
    expert = data["expert"]
    for a, b, label in expert["spans"]:
        out.append(f'<rect x="{x(a):.1f}" y="{row2}" width="{max(2, x(b) - x(a) - 2):.1f}" height="12" rx="3" fill="{theme["rest"]}"/>')
        if x(b) - x(a) > 45:
            out.append(f'<text x="{x(a):.1f}" y="{row2 + 28}" font-size="12">{label}</text>')
    for t, _ in expert["points"]:
        out.append(f'<rect x="{x(t) - 1:.1f}" y="{row2 - 5}" width="2" height="22" rx="1" fill="{theme["soft"]}"/>')
    for m in range(0, int(dur // 60) + 1):
        out.append(f'<text x="{x(m * 60):.1f}" y="{height_of(narrow) - 8}" font-size="11" class="muted">{m}:00</text>')
    return svg(width, height_of(narrow), "".join(out), theme, "Miss You: the light, music mode's sections and the human expert's marks")


def height_of(narrow: bool) -> int:
    return 220 if narrow else 212


def colour_strip(theme: dict, narrow: bool, before: list, after: list) -> str:
    width = 400 if narrow else 880
    left = 12 if narrow else 130
    w = width - left - 16
    out = []
    for i, (name, cols) in enumerate((("Before", before), ("After", after))):
        y = 12 + i * 70 + (18 if narrow else 0)
        if narrow:
            out.append(f'<text x="{left}" y="{y - 6}" font-size="12" class="ink">{name}</text>')
        else:
            out.append(f'<text x="{left - 12}" y="{y + 26}" font-size="13" text-anchor="end" class="ink">{name}</text>')
        out.append(band(left, y, w, 44, cols))
    return svg(width, 160 if narrow else 140, "".join(out), theme, "The default mood before and after the colour change")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DOCS / "figures")
    parser.add_argument("--before-colours", type=Path, help="a timeline computed before the colour change")
    parser.add_argument("--after-colours", type=Path, help="the same track after it")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for theme_name, theme in THEMES.items():
        for narrow in (False, True):
            suffix = f"{'narrow' if narrow else 'wide'}-{theme_name}"
            (args.out / f"models-{suffix}.svg").write_text(models(theme, narrow))
            (args.out / f"tempo-{suffix}.svg").write_text(tempo(theme, narrow))
            (args.out / f"missyou-{suffix}.svg").write_text(missyou(theme, narrow))
            if args.before_colours and args.after_colours:
                before = json.loads(args.before_colours.read_text())["colours"]["ethereal"]
                after = json.loads(args.after_colours.read_text())["colours"]["ethereal"]
                (args.out / f"colour-{suffix}.svg").write_text(colour_strip(theme, narrow, before, after))
    print("\n".join(sorted(p.name for p in args.out.glob("*.svg"))))


if __name__ == "__main__":
    main()
