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
    seq = np.arange(30 * sections.ROW_WIDTH, dtype=float).reshape(30, sections.ROW_WIDTH)
    w = sections.window(seq, 20)
    assert w.shape == (sections.WINDOW_SIZE,)
    n = sections.ROW_WIDTH
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
    intro_only = {"intro": 0.9, "groove": 0.05, "breakdown": 0.05, "build": 0.0, "climax": 0.0}
    assert [d.step(intro_only) for _ in range(5)] == ["groove"] * 5   # never back to an intro
    climax = {"groove": 0.05, "breakdown": 0.05, "build": 0.0, "climax": 0.9}
    breakdown = {"groove": 0.1, "breakdown": 0.85, "build": 0.05, "climax": 0.0}
    seq = [d.step(breakdown) for _ in range(4)]
    assert seq[0] == "groove" and seq[-1] == "breakdown"          # after holding 2 s
    steps = next(n for n in range(1, 30) if d.step(climax) == "climax")  # a section lasts a while
    assert sections.MIN_STAY - 2 <= steps <= sections.MIN_STAY + sections.HOLD_SECONDS
    assert d.released                                             # breakdown -> climax is a release


def test_unsure_model_changes_nothing():
    d = sections.SectionDecoder()
    unsure = {"groove": 0.3, "breakdown": 0.4, "build": 0.2, "climax": 0.1}
    assert [d.step(unsure) for _ in range(10)] == ["groove"] * 10


def test_balanced_accuracy_does_not_reward_always_groove():
    truth = ["groove"] * 60 + ["breakdown"] * 20 + ["build"] * 10 + ["climax"] * 10
    always = ["groove"] * 100
    result = score_sections.score(truth, always, [], [])
    assert result["frame_accuracy"] == 0.6
    assert result["balanced_accuracy"] == 0.25


def test_groove_may_go_straight_to_climax_without_a_release():
    d = sections.SectionDecoder()
    climax = {"groove": 0.05, "breakdown": 0.05, "build": 0.0, "climax": 0.9}
    seq = [d.step(climax) for _ in range(3)]
    assert seq[-1] == "climax" and not d.released



def test_tree_export_matches_scikit_learn():
    from sklearn.ensemble import HistGradientBoostingClassifier
    import train_sections
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 10))
    y = np.array(["groove", "breakdown", "build", "climax"])[(X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0.5)]
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    model = sections.Model(train_sections.export_trees(clf))
    ours = model.proba_batch(X[:50])
    theirs = clf.predict_proba(X[:50])
    order = [model.classes.index(c) for c in clf.classes_]
    assert np.allclose(ours[:, order], theirs, atol=1e-6)
    single = model.proba(X[0])
    assert abs(single[clf.classes_[0]] - theirs[0, 0]) < 1e-6


def test_pooled_boundary_score_does_not_reward_predicting_nothing():
    import train_sections
    # Two tracks: one with a build the rules never call, one with no build at all.
    with_build = score_sections.score(["groove"] * 20 + ["build"] * 10, ["groove"] * 30, [], [])
    without = score_sections.score(["groove"] * 30, ["groove"] * 30, [], [])
    pooled = train_sections.summarise([with_build, without])
    assert pooled["build_f"] == 0.0  # per-track averaging would give 0.5


def test_spectrogram_network_numpy_matches_torch():
    import torch
    import train_cnn
    torch.manual_seed(0)
    net = train_cnn.SectionNet(classes=["breakdown", "build", "climax", "groove"]).eval()
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(3, sections.MEL_FRAMES, 64)).astype(np.float32) - 5
    weights = train_cnn.export_cnn(net, level_mean=-5.0, level_scale=2.0)
    model = sections.Model(weights)
    ours = model.proba_batch(windows)
    with torch.no_grad():
        theirs = torch.softmax(net(torch.from_numpy(train_cnn.prepare(windows, -5.0, 2.0))), dim=1).numpy()
    assert np.allclose(ours, theirs, atol=1e-5)
    assert model.input == "mel"
