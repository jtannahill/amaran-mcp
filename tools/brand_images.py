"""The plugin icon and the demo page's share image, drawn from a track's real light.

Usage: uv run --with playwright python tools/brand_images.py [--track docs/missyou.json]

Writes .claude-plugin/icon.png (1024 px square) and docs/og.png (1200 x 630).
Both show the tube in the colours music mode gave the track, on the dark wall
the demo page uses, so the icon, the share card and the page read as one thing.
Rendered by headless Chromium with the page's own self-hosted fonts.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).parent.parent
WALL = "#0b0d14"


def light_rgb(h: float, s: float, b: float) -> tuple[int, int, int]:
    """The demo page's lightRGB: the light at its colour and level."""

    r, g, bl = colorsys.hsv_to_rgb((h % 360) / 360, s / 100, 0.18 + 0.82 * b / 100)
    return round(r * 255), round(g * 255), round(bl * 255)


def css(rgb: tuple[int, int, int], alpha: float = 1.0) -> str:
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


def journey(colours: list, stops: int) -> list[tuple[int, int, int]]:
    """The track's colour at evenly spaced moments, each averaged over its stretch."""

    n, out = len(colours), []
    for i in range(stops):
        part = colours[i * n // stops:(i + 1) * n // stops]
        rgbs = [light_rgb(*c) for c in part]
        out.append(tuple(round(sum(x[k] for x in rgbs) / len(rgbs)) for k in range(3)))
    return out


def vivid(colours: list, stops: int) -> list[tuple[int, int, int]]:
    """The track's most vivid colour in each stretch: what the eye remembers of it."""

    n, out = len(colours), []
    for i in range(stops):
        part = colours[i * n // stops:(i + 1) * n // stops]
        out.append(light_rgb(*max(part, key=lambda c: c[1] * c[2])))
    return out


def gradient(rgbs: list, direction: str) -> str:
    return f"linear-gradient({direction}, " + ", ".join(
        f"{css(c)} {100 * i / (len(rgbs) - 1):.1f}%" for i, c in enumerate(rgbs)) + ")"


FONTS = f"""
@font-face {{ font-family: "Unbounded"; font-weight: 300 400; src: url({(ROOT / "docs/fonts/Unbounded.woff2").as_uri()}) format("woff2"); }}
@font-face {{ font-family: "Hanken Grotesk"; font-weight: 400 600; src: url({(ROOT / "docs/fonts/HankenGrotesk.woff2").as_uri()}) format("woff2"); }}
* {{ margin: 0; box-sizing: border-box; }}
"""


def icon_html(rgbs: list) -> str:
    glow = rgbs[len(rgbs) // 2]
    return f"""<!doctype html><style>{FONTS}
body {{ width: 1024px; height: 1024px; background: {WALL}; position: relative; overflow: hidden; }}
.spill {{ position: absolute; inset: 0;
  background: radial-gradient(ellipse 46% 60% at 50% 50%, {css(glow, 0.55)}, {css(glow, 0.12)} 55%, transparent 80%); }}
.tube {{ position: absolute; left: 50%; top: 112px; bottom: 112px; width: 236px; margin-left: -118px;
  border-radius: 118px; background: {gradient(rgbs, "to bottom")};
  box-shadow: 0 0 120px 24px {css(glow, 0.45)}; }}
.sheen {{ position: absolute; left: 50%; top: 112px; bottom: 112px; width: 236px; margin-left: -118px; border-radius: 118px;
  background: linear-gradient(to right, rgba(255,255,255,0) 4%, rgba(255,255,255,0.16) 26%, rgba(255,255,255,0.05) 42%, rgba(255,255,255,0) 62%); }}
</style><div class="spill"></div><div class="tube"></div><div class="sheen"></div>"""


def og_html(rgbs: list, band: list) -> str:
    glow = rgbs[len(rgbs) // 2]
    return f"""<!doctype html><style>{FONTS}
body {{ width: 1200px; height: 630px; background: {WALL}; color: #eef0f6; position: relative; overflow: hidden;
  font: 400 30px/1.4 "Hanken Grotesk", sans-serif; }}
.spill {{ position: absolute; inset: 0;
  background: radial-gradient(ellipse 40% 80% at 12% 45%, {css(glow, 0.5)}, {css(glow, 0.1)} 55%, transparent 80%); }}
.tube {{ position: absolute; left: 110px; top: 70px; height: 400px; width: 64px; border-radius: 32px;
  background: {gradient(rgbs, "to bottom")}; box-shadow: 0 0 80px 14px {css(glow, 0.4)}; }}
.words {{ position: absolute; left: 260px; right: 80px; top: 104px; }}
h1 {{ font: 300 92px/1.02 "Unbounded", sans-serif; letter-spacing: -0.03em; margin-bottom: 30px; }}
p {{ color: #c3c8d6; max-width: 19em; }}
.band {{ position: absolute; left: 110px; right: 80px; bottom: 74px; height: 44px; border-radius: 10px;
  background: {gradient(band, "to right")}; }}
.name {{ position: absolute; left: 260px; bottom: 136px; font: 600 24px/1 "Hanken Grotesk", sans-serif; color: #eef0f6; }}
</style><div class="spill"></div><div class="tube"></div>
<div class="words"><h1>Play a breakdown</h1>
<p>An amaran light that follows a trance track's sections, not its beat.</p></div>
<div class="name">amaran-mcp</div><div class="band"></div>"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", type=Path, default=ROOT / "docs/missyou.json")
    parser.add_argument("--mood", default="ethereal")
    args = parser.parse_args()
    colours = json.loads(args.track.read_text())["colours"][args.mood]
    tube, band = vivid(colours, 6), journey(colours, 60)
    jobs = [(ROOT / ".claude-plugin/icon.png", icon_html(tube), 1024, 1024),
            (ROOT / "docs/og.png", og_html(tube, band), 1200, 630)]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for out, html, w, h in jobs:
            page = browser.new_page(viewport={"width": w, "height": h})
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
                f.write(html)  # a file page, so the file:// fonts load
            page.goto(Path(f.name).as_uri())
            Path(f.name).unlink()
            page.evaluate("document.fonts.ready")
            page.wait_for_timeout(300)
            page.screenshot(path=str(out))
            page.close()
            print(out.relative_to(ROOT), f"{out.stat().st_size // 1024} KB")
        browser.close()


if __name__ == "__main__":
    main()
