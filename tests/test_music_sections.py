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


def test_numpy_inference_matches_scikit_learn():
    from sklearn.neural_network import MLPClassifier
    import train_sections
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    y = np.array(["groove", "breakdown", "climax"])[rng.integers(0, 3, 300)]
    means, scales = X.mean(axis=0), X.std(axis=0)
    clf = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=50, random_state=0).fit((X - means) / scales, y)
    model = sections.Model(train_sections.export(clf, means, scales))
    for x in X[:20]:
        ours = model.proba(x)
        theirs = dict(zip(clf.classes_, clf.predict_proba(((x - means) / scales)[None])[0]))
        assert all(abs(ours[c] - theirs[c]) < 1e-6 for c in clf.classes_)


def test_decoder_holds_and_respects_musical_order():
    d = sections.SectionDecoder()
    climax = {"groove": 0.05, "breakdown": 0.05, "build": 0.0, "climax": 0.9}
    assert [d.step(climax) for _ in range(5)] == ["groove"] * 5   # groove cannot jump to climax
    breakdown = {"groove": 0.1, "breakdown": 0.85, "build": 0.05, "climax": 0.0}
    seq = [d.step(breakdown) for _ in range(4)]
    assert seq[0] == "groove" and seq[-1] == "breakdown"          # after holding 2 s
    assert d.step(climax) == "breakdown" and d.step(climax) == "climax"
    assert d.released                                             # breakdown -> climax is a release


def test_unsure_model_changes_nothing():
    d = sections.SectionDecoder()
    unsure = {"groove": 0.3, "breakdown": 0.4, "build": 0.2, "climax": 0.1}
    assert [d.step(unsure) for _ in range(10)] == ["groove"] * 10
