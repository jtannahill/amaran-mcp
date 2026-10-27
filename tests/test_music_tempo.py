"""Tempo, bar and phrase tracking on synthetic kicks."""

import numpy as np
import pytest

from amaran_mcp.music.tempo import TempoTracker
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


def test_clashing_blend_stays_within_its_tempos():
    """Two unmatched tempos at once (rare: DJs beatmatch): the reading stays
    within the blend's range, so colour moves still land near the bars."""
    mix = kick_track(RATE, 20, bpm=128) * 0.6 + kick_track(RATE, 20, bpm=134) * 0.6
    tail = [b.bpm for b in feed_in_blocks(TempoTracker(RATE), mix)[-50:] if b.bpm]
    assert tail and all(127 <= bpm <= 135 for bpm in tail)


def test_pulse_tells_kick_from_held_sub_bass():
    """Seen on a real episode: tempo confidence is slow, bass level is fooled by sub-bass."""
    from music_fixtures import sub_bass_pads
    t = TempoTracker(RATE)
    groove = feed_in_blocks(t, kick_track(RATE, 30, bpm=138))
    held = feed_in_blocks(t, sub_bass_pads(RATE, 20))
    assert all(b.kick for b in groove[-40:])
    assert not any(b.kick for b in held[-300:])


def test_onset_jump_as_soon_as_the_kick_returns():
    from music_fixtures import sub_bass_pads
    t = TempoTracker(RATE)
    feed_in_blocks(t, kick_track(RATE, 30, bpm=138))
    held = feed_in_blocks(t, sub_bass_pads(RATE, 20))
    assert max(b.onset_jump for b in held[-200:]) < 2
    back = feed_in_blocks(t, kick_track(RATE, 3, bpm=138))
    first = next(i for i, b in enumerate(back) if b.onset_jump >= 2)
    assert first * 1024 / RATE <= 0.7


def test_steady_groove_has_no_onset_jump():
    beats = feed_in_blocks(TempoTracker(RATE), kick_track(RATE, 30, bpm=138))
    assert max(b.onset_jump for b in beats[-400:]) < 2
