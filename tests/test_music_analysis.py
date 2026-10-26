"""Music analysis on synthetic audio."""

from amaran_mcp.music.analysis import Analyzer
from music_fixtures import feed_in_blocks, kick_track, pads_only, silence, speech_like

RATE = 24000


def test_kick_track_is_music_with_steady_energy():
    fs = feed_in_blocks(Analyzer(RATE), kick_track(RATE, 30))
    tail = fs[len(fs) // 2:]
    assert all(f.is_music for f in tail)
    assert all(f.section == "playing" for f in tail)
    energies = [f.energy for f in tail]
    assert max(energies) - min(energies) < 0.15  # breathes, does not pulse per kick


def test_speech_is_not_music():
    fs = feed_in_blocks(Analyzer(RATE), speech_like(RATE, 20))
    assert not any(f.is_music for f in fs[len(fs) // 2:])


def test_silence_after_ten_seconds():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 20))
    fs = feed_in_blocks(a, silence(RATE, 12))
    seconds = [i * 1024 / RATE for i, f in enumerate(fs) if f.section == "silence"]
    assert seconds and 8.0 <= seconds[0] <= 11.0
    assert fs[-1].digital_zero


def test_breakdown_then_one_drop():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 40))
    breakdown = feed_in_blocks(a, pads_only(RATE, 16))
    assert breakdown[-1].section == "breakdown"
    after = feed_in_blocks(a, kick_track(RATE, 10))
    assert sum(f.drop for f in after) == 1
    assert sum(f.drop for f in breakdown) == 0
