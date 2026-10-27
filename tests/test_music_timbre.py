"""Vocal, tension and openness measures on synthetic audio."""

import numpy as np

from amaran_mcp.music.timbre import TimbreTracker
from music_fixtures import feed_in_blocks, kick_track, pads_only, riser, speech_over, sung

RATE = 24000


def test_sung_vocal_over_pads_reads_high():
    mix = pads_only(RATE, 12) + sung(RATE, 12, notes=[440, 494, 523])
    assert np.mean([t.vocal for t in feed_in_blocks(TimbreTracker(RATE), mix)[-100:]]) > 0.6


def test_static_pads_read_low():
    assert np.mean([t.vocal for t in feed_in_blocks(TimbreTracker(RATE), pads_only(RATE, 12))[-100:]]) < 0.2


def test_riser_raises_tension_steadily():
    ts = feed_in_blocks(TimbreTracker(RATE), np.concatenate([kick_track(RATE, 30), riser(RATE, 16)]))
    build = [t.tension for t in ts[-int(16 * RATE / 1024):]]  # from the start of the riser
    assert build[-1] > 0.7 and build[-1] - build[0] > 0.4
    third = len(build) // 3
    assert build[0] < build[third] < build[2 * third] < build[-1]


def test_quiet_breakdown_level_is_reported():
    quiet = pads_only(RATE, 10) * 0.1   # about -45 dBFS
    assert -50 < feed_in_blocks(TimbreTracker(RATE), quiet)[-1].level_db < -40


def test_host_speech_over_groove_is_not_tension():
    ts = feed_in_blocks(TimbreTracker(RATE), speech_over(RATE, 20, kick_track(RATE, 20)))
    assert max(t.tension for t in ts[-200:]) < 0.5


def feed_with_classes(tracker, samples, classes, block=1024):
    return [tracker.feed(samples[i : i + block], classes) for i in range(0, len(samples) - block + 1, block)]


def test_classifier_singing_drives_vocal():
    ts = feed_with_classes(TimbreTracker(RATE), pads_only(RATE, 4), {"singing": 0.9, "speech": 0.1})
    assert ts[-1].vocal > 0.7 and ts[-1].singing == 0.9


def test_classifier_speech_holds_vocal_down():
    """A host talking over sung vocals: speech wins."""
    ts = feed_with_classes(TimbreTracker(RATE), pads_only(RATE, 4), {"singing": 0.9, "speech": 0.9})
    assert ts[-1].vocal < 0.2


def test_choir_counts_as_singing():
    ts = feed_with_classes(TimbreTracker(RATE), pads_only(RATE, 4), {"singing": 0.1, "choir_singing": 0.8, "speech": 0.0})
    assert ts[-1].vocal > 0.6
