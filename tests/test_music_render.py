"""Renderer: slew limits, mood bounds, sections and releases."""

import random

from amaran_mcp.music.moods import MOODS
from amaran_mcp.music import render
from amaran_mcp.music.render import Renderer
from music_fixtures import make_features


def features(energy=0.5, section="groove", release=False):
    return make_features(energy=energy, bass=energy, section=section, release=release)


LOUD = features(1.0)
QUIET = features(0.1)
BREAKDOWN = features(0.4, section="breakdown")
DROP = features(1.0, release=True)



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
        f = make_features(energy=rng.random(), bass=rng.random(),
                          section=rng.choice(["groove", "breakdown"]), release=rng.random() < 0.02)
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


def test_drop_reaches_next_anchor_within_three_seconds():
    r = Renderer(MOODS["ethereal"])
    run(r, LOUD, 10)
    target = r.current_target_hue_after_drop()
    outs = [r.step(DROP, 1 / 8)] + run(r, LOUD, 3.5 - 1 / 8)
    assert hue_gap(outs[-1].hue, target) < 5


def test_brightness_offset_shifts_range():
    r = Renderer(MOODS["ethereal"], brightness_offset=15)
    assert all(40 - 1e-9 <= o.brightness <= 85 + 1e-9 for o in run(r, LOUD, 30))


def test_drop_skips_pale_anchors():
    """A drop lands on a colour, never on a near-white stop."""
    from dataclasses import replace
    misty = replace(MOODS["ethereal"], anchors=((225, 85), (270, 78), (190, 70), (220, 12)))
    r = Renderer(misty, start_anchor=2)  # teal; the next stop is misty white
    run(r, LOUD, 1)
    assert r.current_target_hue_after_drop() == misty.anchors[0][0]
    outs = [r.step(DROP, 1 / 8)] + run(r, LOUD, 4)
    assert outs[-1].saturation >= 40


# Section helpers: features for one section, with the named readings.

def intro_f(energy=0.5, **kw):
    return make_features(energy=energy, section="intro", **kw)


def groove_f(energy=0.6, **kw):
    return make_features(energy=energy, section="groove", **kw)


def breakdown_f(seconds=8.0, chord_root=None, balance=0.0, **kw):
    return make_features(energy=0.5, section="breakdown", section_seconds=seconds,
                         chord_root=chord_root, balance=balance, **kw)


def build_f(tension=0.8, **kw):
    return make_features(energy=0.6, section="build", tension=tension, **kw)


def climax_f(**kw):
    return make_features(energy=0.9, section="climax", **kw)


def warm(hue):
    return hue >= 280 or hue <= 50


def test_breakdown_starts_pale_then_takes_chord_colour():
    r = Renderer(MOODS["ethereal"])
    run(r, groove_f(), 10)
    early = run(r, breakdown_f(seconds=1, chord_root=9, balance=-0.8), 3)[-1]
    assert early.saturation <= render.BREAKDOWN_SATURATION
    later = run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.8), 6)[-1]
    assert later.saturation > early.saturation + 15 and 210 <= later.hue <= 280      # cool family


def test_major_balance_moves_to_warm_family():
    r = Renderer(MOODS["ethereal"])
    run(r, breakdown_f(seconds=8, chord_root=0, balance=0.8), 10)
    assert warm(run(r, breakdown_f(seconds=12, chord_root=0, balance=0.8), 4)[-1].hue)


def test_bloom_swells_and_eases_back():
    r = Renderer(MOODS["ethereal"])
    base = run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.5), 8)[-1]
    peak = max(o.brightness for o in [r.step(breakdown_f(seconds=9, chord_root=9, balance=-0.5, bloom=True, vocal=0.9), 1 / 8)]
               + run(r, breakdown_f(seconds=10, chord_root=9, balance=-0.5, vocal=0.9), 4))
    assert peak >= base.brightness + 8
    after = run(r, breakdown_f(seconds=20, chord_root=9, balance=-0.5, vocal=0.0), 8)[-1]
    assert after.brightness <= base.brightness + 3


def test_bloom_warms_the_colour():
    r = Renderer(MOODS["ethereal"])
    run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.8), 8)
    r.step(breakdown_f(seconds=9, chord_root=9, balance=-0.8, bloom=True, vocal=0.9), 1 / 8)
    assert warm(run(r, breakdown_f(seconds=10, chord_root=9, balance=-0.8, vocal=0.9), 5)[-1].hue)


def test_build_gathers_drift_stops():
    r = Renderer(MOODS["ethereal"])
    hues = [o.hue for o in run(r, build_f(tension=0.95), 12)]
    assert max(hues[-16:]) - min(hues[-16:]) < 2


def test_chord_change_waits_for_bar():
    r = Renderer(MOODS["ethereal"])
    run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.8), 6)
    before = r.step(breakdown_f(seconds=9, chord_root=5, balance=-0.8, change=True, bpm=138,
                                confidence=0.9, seconds_to_bar=1.0), 1 / 8).hue
    held = run(r, breakdown_f(seconds=9.2, chord_root=5, balance=-0.8, bpm=138, confidence=0.9,
                              seconds_to_bar=0.7), 0.5)[-1].hue
    assert abs(held - before) < 1.5                                 # waits for the bar
    moved = run(r, breakdown_f(seconds=10, chord_root=5, balance=-0.8, bpm=138, confidence=0.9,
                               seconds_to_bar=1.5), 3)[-1].hue
    assert abs(moved - before) > 10                                 # then goes


def test_long_breakdown_without_changes_keeps_drifting():
    r = Renderer(MOODS["ethereal"])
    hues = [o.hue for o in run(r, breakdown_f(seconds=40, chord_root=9, balance=-0.8), 30)]
    assert max(hues) - min(hues) > 3


def test_intro_is_restrained():
    r = Renderer(MOODS["ethereal"])
    low, high = MOODS["ethereal"].brightness
    assert run(r, intro_f(energy=1.0), 20)[-1].brightness <= low + (high - low) / 3 + 1e-9


def test_release_opens_to_the_ceiling():
    r = Renderer(MOODS["ethereal"])
    run(r, build_f(tension=0.9), 8)
    outs = [r.step(climax_f(release=True), 1 / 8)] + run(r, climax_f(), 6)
    assert outs[-1].brightness >= MOODS["ethereal"].brightness[1] - 3


def test_openness_reads_luminous():
    closed = run(Renderer(MOODS["ethereal"]), groove_f(openness=0.0), 15)[-1]
    open_ = run(Renderer(MOODS["ethereal"]), groove_f(openness=1.0), 15)[-1]
    assert open_.saturation < closed.saturation and open_.brightness > closed.brightness


def test_trance_reads_as_colour_not_grey():
    # The owner: "the trance music should have more color". Through a whole
    # track (groove, a breakdown with its chords and a lift, build, drop) the
    # light holds real saturation; at 34 on average it read as grey.
    for name, mood in MOODS.items():
        r = Renderer(mood)
        outs = run(r, intro_f(), 8) + run(r, groove_f(0.4), 20) + run(r, groove_f(0.8), 20)
        for i, root in enumerate((10, 8, 6, 8)):
            outs += run(r, breakdown_f(seconds=8.0 + 6 * i, chord_root=root, balance=-0.8), 6)
        outs += run(r, breakdown_f(seconds=32.0, chord_root=8, balance=0.8, bloom=True, vocal=0.6), 1)
        outs += run(r, breakdown_f(seconds=33.0, chord_root=8, balance=0.8, vocal=0.6), 8)
        outs += run(r, build_f(), 10) + [r.step(climax_f(release=True), 1 / 8)] + run(r, climax_f(), 20)
        sats = [o.saturation for o in outs[8 * 8:]]  # after the intro
        assert sum(sats) / len(sats) >= 55, name
        assert sum(s < 30 for s in sats) / len(sats) <= 0.1, name
