"""Tempo, bar and phrase tracking on synthetic kicks."""

import numpy as np
import pytest

from amaran_mcp.music.tempo import CONFIDENT, TempoTracker
from music_fixtures import feed_in_blocks, kick_track, pads_only

RATE = 24000


@pytest.mark.parametrize("bpm", [125, 138])
def test_tempo_locks_within_one_bpm(bpm):
    beats = feed_in_blocks(TempoTracker(RATE), kick_track(RATE, 30, bpm=bpm))
    assert beats[-1].bpm == pytest.approx(bpm, abs=1)


def test_bars_every_four_beats_and_phrases_every_eight_bars():
    beats = feed_in_blocks(TempoTracker(RATE), kick_track(RATE, 80, bpm=138))
    bar_times = [i * 1024 / RATE for i, b in enumerate(beats) if b.bar]
    gaps = np.diff(bar_times[3:])
    block = 1024 / RATE  # boundaries are reported to the nearest block
    assert np.all(np.abs(gaps - 4 * 60 / 138) <= block + 0.02)
    assert abs(gaps.mean() - 4 * 60 / 138) < 0.01
    phrases = [b for b in beats if b.phrase]
    assert 1 <= len(phrases) <= 6
    phrase_times = [i * 1024 / RATE for i, b in enumerate(beats) if b.phrase]
    assert np.all(np.abs(np.diff(phrase_times) - 32 * 60 / 138) <= block + 0.02)


def test_seconds_to_bar_counts_down():
    beats = feed_in_blocks(TempoTracker(RATE), kick_track(RATE, 30, bpm=138))
    tail = [b.seconds_to_bar for b in beats[-20:]]
    assert all(t is not None and 0 <= t <= 4 * 60 / 138 + 0.05 for t in tail)


def test_no_kick_means_no_tempo():
    assert feed_in_blocks(TempoTracker(RATE), pads_only(RATE, 20))[-1].bpm is None


def test_blend_of_two_tempos_drops_confidence():
    mix = kick_track(RATE, 20, bpm=128) * 0.6 + kick_track(RATE, 20, bpm=134) * 0.6
    tail = feed_in_blocks(TempoTracker(RATE), mix)[-50:]
    assert min(b.confidence for b in tail) < CONFIDENT
