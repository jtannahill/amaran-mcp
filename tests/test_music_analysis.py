"""Music analysis on synthetic audio."""

from amaran_mcp.music.analysis import Analyzer, merge_events
from music_fixtures import (arrangement, feed_in_blocks, kick_track, make_features, pads_only, silence,
                            speech_like, speech_over)

RATE = 24000


def test_kick_track_is_music_with_steady_energy():
    fs = feed_in_blocks(Analyzer(RATE), kick_track(RATE, 30))
    tail = fs[len(fs) // 2:]
    assert all(f.is_music for f in tail)
    assert all(f.section == "groove" for f in tail)
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


def at(fs, seconds):
    return fs[int(seconds * RATE / 1024)]


def test_arrangement_sections_in_order():
    samples, marks = arrangement(RATE)
    fs = feed_in_blocks(Analyzer(RATE), samples)
    assert at(fs, marks["intro"][1] - 2).section == "intro"
    assert at(fs, marks["groove"][1] - 2).section == "groove"
    assert at(fs, marks["breakdown"][0] + 15).section == "breakdown"
    assert at(fs, marks["build"][1] - 2).section == "build"
    assert at(fs, marks["climax"][0] + 10).section == "climax"
    assert sum(f.release for f in fs) == 1


def test_bloom_only_on_lift_inside_breakdown():
    samples, marks = arrangement(RATE)
    fs = feed_in_blocks(Analyzer(RATE), samples)
    blooms = [i * 1024 / RATE for i, f in enumerate(fs) if f.bloom]
    assert blooms and all(marks["breakdown"][0] <= t <= marks["build"][0] for t in blooms)


def test_host_speech_over_groove_changes_nothing():
    bed = kick_track(RATE, 30)
    plain = feed_in_blocks(Analyzer(RATE), bed)
    talked = feed_in_blocks(Analyzer(RATE), speech_over(RATE, 30, bed))
    assert [f.section for f in plain[-200:]] == [f.section for f in talked[-200:]]
    assert not any(f.bloom for f in talked)


def test_quiet_breakdown_is_not_silence():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 30))
    fs = feed_in_blocks(a, pads_only(RATE, 25) * 0.05)
    assert all(f.section != "silence" for f in fs)


def test_merge_events_keeps_one_tick_events():
    earlier = make_features(bloom=True, bar=True, lift=True)
    later = make_features()
    merged = merge_events(earlier, later)
    assert merged.bloom and merged.beat.bar and merged.harmony.lift
    assert not merged.release
