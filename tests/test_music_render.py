"""Renderer: slew limits, mood bounds, breakdowns and drops."""

import random

from amaran_mcp.music.analysis import Features
from amaran_mcp.music.moods import MOODS
from amaran_mcp.music.render import Renderer


def features(energy=0.5, section="playing", drop=False):
    return Features(energy=energy, bass=energy, is_music=True, section=section,
                    drop=drop, digital_zero=False)


LOUD = features(1.0)
QUIET = features(0.1)
BREAKDOWN = features(0.4, section="breakdown")
DROP = features(1.0, drop=True)



def run(renderer, f, seconds, hz=8):
    return [renderer.step(f, 1 / hz) for _ in range(int(seconds * hz))]


def hue_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def test_slew_limits_hold_under_chaotic_input():
    r = Renderer(MOODS["ethereal"])
    rng = random.Random(1)
    outs = []
    for _ in range(8 * 120):
        f = Features(energy=rng.random(), bass=rng.random(), is_music=True,
                     section=rng.choice(["playing", "breakdown"]), drop=rng.random() < 0.02,
                     digital_zero=False)
        outs.append(r.step(f, 1 / 8))
    for a, b in zip(outs, outs[1:]):
        assert abs(b.brightness - a.brightness) <= 12 / 8 + 1e-9
        assert abs(b.saturation - a.saturation) <= 20 / 8 + 1e-9
        assert hue_gap(a.hue, b.hue) <= 30 / 8 + 1e-9


def test_bounds_never_leave_the_mood():
    for mood in MOODS.values():
        r = Renderer(mood)
        for f in (LOUD, QUIET, BREAKDOWN, DROP):
            for o in run(r, f, 20):
                assert mood.brightness[0] - 1e-9 <= o.brightness <= mood.brightness[1] + 1e-9
                assert o.saturation <= mood.max_saturation + 1e-9


def test_breakdown_goes_pale():
    r = Renderer(MOODS["ethereal"])
    run(r, LOUD, 20)
    assert run(r, BREAKDOWN, 8)[-1].saturation <= 30


def test_drop_reaches_next_anchor_within_three_seconds():
    r = Renderer(MOODS["ethereal"])
    run(r, LOUD, 10)
    target = r.current_target_hue_after_drop()
    outs = [r.step(DROP, 1 / 8)] + run(r, LOUD, 3.5 - 1 / 8)
    assert hue_gap(outs[-1].hue, target) < 5


def test_brightness_offset_shifts_range():
    r = Renderer(MOODS["ethereal"], brightness_offset=15)
    assert all(40 - 1e-9 <= o.brightness <= 85 + 1e-9 for o in run(r, LOUD, 30))
