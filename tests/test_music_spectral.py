"""Per-second spectral summary: band energies, flux, flatness, centroid."""

import numpy as np

from amaran_mcp.music.spectral import BANDS, Spectral
from music_fixtures import feed_in_blocks, kick_track, riser

RATE = 24000


def seconds_of(samples):
    s = Spectral(RATE)
    rows = [r for r in feed_in_blocks(s, samples) if r is not None]
    return np.array(rows)


def test_one_row_per_second_with_documented_width():
    rows = seconds_of(kick_track(RATE, 10))
    assert rows.shape == (9, Spectral.WIDTH) or rows.shape == (10, Spectral.WIDTH)
    assert Spectral.WIDTH == len(BANDS) + 3


def test_riser_lifts_the_high_bands_against_the_groove():
    rows = seconds_of(np.concatenate([kick_track(RATE, 30), riser(RATE, 16)]))
    high = len(BANDS) - 1
    assert rows[-1, high] > rows[25, high] + 0.3
