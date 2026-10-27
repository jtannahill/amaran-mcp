"""tools/align_versions.py: finding where moments of one cut sit in another version."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import align_versions  # noqa: E402


def song(frames, seed=3):
    """A stand-in for a log-mel sequence: smooth random features, 10 frames a second."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(frames, 16)).cumsum(axis=0)
    return x - x.mean(axis=0)


def test_a_cut_that_drops_a_stretch_maps_back_to_the_full_version():
    full = song(3000)                                     # 300 s
    cut = np.concatenate([full[200:1200], full[1600:2800]])  # starts 20 s in, skips 40 s
    mapping = align_versions.mapping(cut, full, window=80, step=20)
    assert abs(align_versions.map_time(mapping, 10.0) - 30.0) < 1.0     # before the skip
    assert abs(align_versions.map_time(mapping, 150.0) - 210.0) < 1.0   # after the skip


def test_moments_outside_the_matched_part_are_dropped():
    full = song(3000)
    cut = np.concatenate([song(300, seed=9), full[500:2000]])  # 30 s of another track first
    mapping = align_versions.mapping(cut, full, window=80, step=20)
    assert align_versions.map_time(mapping, 5.0) is None
    assert abs(align_versions.map_time(mapping, 60.0) - 80.0) < 1.0


def test_a_repeated_section_maps_to_the_repeat_that_keeps_time_moving_forward():
    # Trance repeats itself: in full = A B A C the cut's second A must map to the
    # second A in full, not jump back to the first.
    a, b, c = song(500, seed=1), song(500, seed=2), song(500, seed=4)
    full = np.concatenate([a, b, a, c])
    cut = np.concatenate([b, a, c])
    mapping = align_versions.mapping(cut, full, window=80, step=20)
    assert abs(align_versions.map_time(mapping, 75.0) - 125.0) < 1.0   # in A, after B
    assert abs(align_versions.map_time(mapping, 125.0) - 175.0) < 1.0  # in C


def test_a_mark_just_after_a_splice_takes_the_far_side_offset():
    # The cut skips 40 s at 100 s in; a mark 1 s after the splice belongs after the skip.
    full = song(3000)
    cut = np.concatenate([full[0:1000], full[1400:2800]])
    found = align_versions.mapping(cut, full, window=80, step=20)
    moved = align_versions.map_time(found, 101.0, cut=cut, full=full)
    assert abs(moved - 141.0) < 1.0


def test_a_full_version_shorter_than_one_stretch_matches_nothing():
    assert align_versions.mapping(song(300), song(50), window=80, step=20) == []
