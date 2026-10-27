"""Section features, context windows and scoring."""

import sys
from pathlib import Path

import numpy as np

from amaran_mcp.music import sections
from music_fixtures import make_features

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import score_sections  # noqa: E402


def test_vector_has_one_value_per_feature_in_order():
    f = make_features(energy=0.7, tension=0.4, vocal=0.5)
    v = sections.vector(f)
    assert len(v) == len(sections.FEATURES)
    assert v[sections.FEATURES.index("energy")] == 0.7
    assert v[sections.FEATURES.index("tension")] == 0.4


def test_window_is_causal_and_documented_length():
    seq = np.arange(30 * len(sections.FEATURES), dtype=float).reshape(30, len(sections.FEATURES))
    w = sections.window(seq, 20)
    assert w.shape == (sections.WINDOW_SIZE,)
    n = len(sections.FEATURES)
    assert np.allclose(w[:n], seq[20])                       # last value
    assert np.allclose(w[n : 2 * n], seq[5:21].mean(axis=0))  # mean over the last 16 s
    assert np.allclose(w[2 * n : 3 * n], seq[20] - seq[5])    # change over the window
    early = sections.window(seq, 0)                          # padded at the start
    assert np.allclose(early[:n], seq[0])


def test_perfect_prediction_scores_one():
    truth = ["groove"] * 20 + ["breakdown"] * 20 + ["climax"] * 20
    result = score_sections.score(truth, truth, true_releases=[40.0], predicted_releases=[40.0])
    assert result["frame_accuracy"] == 1.0
    assert result["boundaries"]["breakdown"]["f"] == 1.0 and result["release_f"] == 1.0


def test_three_seconds_late_misses_the_boundary():
    truth = ["groove"] * 20 + ["breakdown"] * 20 + ["groove"] * 20
    late = ["groove"] * 23 + ["breakdown"] * 20 + ["groove"] * 17
    result = score_sections.score(truth, late, true_releases=[], predicted_releases=[])
    assert result["boundaries"]["breakdown"]["f"] == 0.0
    assert 0.85 < result["frame_accuracy"] < 1.0
