"""Chord, balance and lift tracking on synthetic chords."""

from amaran_mcp.music.harmony import HarmonyTracker
from music_fixtures import feed_in_blocks, progression, riser

RATE = 24000


def test_identifies_a_minor_progression():
    seq = [("A", True), ("F", False), ("C", False), ("G", False)]
    hs = feed_in_blocks(HarmonyTracker(RATE), progression(RATE, seq, 4.0))
    seen = [h.chord for h in hs if h.change]
    assert seen[:4] == ["Am", "F", "C", "G"]


def test_lift_on_minor_to_major():
    hs = feed_in_blocks(HarmonyTracker(RATE), progression(RATE, [("A", True), ("C", False)], 5.0))
    lifts = [i for i, h in enumerate(hs) if h.lift]
    assert len(lifts) == 1 and lifts[0] * 1024 / RATE >= 5.0


def test_whole_step_key_change_onto_minor_is_not_a_lift():
    hs = feed_in_blocks(HarmonyTracker(RATE), progression(RATE, [("A", True), ("B", True)], 5.0))
    assert not any(h.lift for h in hs)


def test_noise_has_no_confident_chord():
    hs = feed_in_blocks(HarmonyTracker(RATE), riser(RATE, 8))
    assert all(h.chord is None or h.confidence < 0.5 for h in hs[-40:])


def test_balance_tracks_major_minor_mix():
    minor = feed_in_blocks(HarmonyTracker(RATE), progression(RATE, [("A", True), ("D", True)] * 3, 4.0))
    assert minor[-1].balance < -0.5


def test_major_to_major_is_not_a_lift():
    """Seen on a real episode: Bbm-Ab-Gb-Ab would otherwise lift on every chord."""
    hs = feed_in_blocks(HarmonyTracker(RATE), progression(RATE, [("G#", False), ("F#", False), ("G#", False)], 5.0))
    assert not any(h.lift for h in hs)
