"""Converting SSME .mat annotations to music mode's label format."""

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import ssme  # noqa: E402


def test_mat_converts_to_segments(tmp_path):
    mat = tmp_path / "track.mat"
    savemat(mat, {"drop": np.array([[90.0]]), "build_start": np.array([[40.0, 190.0]]),
                  "build_end": np.array([[52.0, 201.0]]), "break_start": np.array([[60.0]]),
                  "break_end": np.array([[80.0]])})
    labels = ssme.labels(mat, duration=240.0, bpm=128)
    names = [(round(l["start"]), round(l["end"]), l["section"]) for l in labels]
    assert (40, 52, "build") in names and (190, 201, "build") in names
    assert (60, 80, "breakdown") in names
    climax = next(l for l in labels if l["section"] == "climax")
    assert round(climax["start"]) == 90 and abs(climax["end"] - (90 + 32 * 4 * 60 / 128)) < 0.5
    assert labels[0]["start"] == 0 and labels[-1]["end"] == 240.0  # the gaps are groove
    assert ssme.releases(mat) == [90.0]


def test_unequal_arrays_skip_that_event(tmp_path, caplog):
    mat = tmp_path / "bad.mat"
    savemat(mat, {"drop": np.array([[100.0]]), "build_start": np.array([[40.0, 70.0]]),
                  "build_end": np.array([[52.0]])})
    labels = ssme.labels(mat, duration=200.0, bpm=128)
    assert not any(l["section"] == "build" for l in labels)
    assert any(l["section"] == "climax" for l in labels)
    assert "build" in caplog.text


def test_missing_field_means_no_events_of_that_kind(tmp_path):
    mat = tmp_path / "sparse.mat"
    savemat(mat, {"break_start": np.array([[10.0]]), "break_end": np.array([[30.0]])})
    labels = ssme.labels(mat, duration=100.0, bpm=128)
    assert {l["section"] for l in labels} == {"groove", "breakdown"}


def test_short_gap_before_a_drop_belongs_to_the_build(tmp_path):
    """Seen in SSME track 100189023: build 70-85 s, drop at 87 s."""
    mat = tmp_path / "gap.mat"
    savemat(mat, {"drop": np.array([[87.0]]), "build_start": np.array([[70.0]]), "build_end": np.array([[85.0]])})
    labels = ssme.labels(mat, duration=200.0, bpm=128)
    build = next(l for l in labels if l["section"] == "build")
    assert build["end"] == 87.0
    assert not any(l["section"] == "groove" and 85 <= l["start"] < 87 for l in labels)
