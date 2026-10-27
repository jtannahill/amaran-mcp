"""Music analysis on synthetic audio."""

import numpy as np

from amaran_mcp.music.analysis import Analyzer, merge_events
from music_fixtures import (arrangement, feed_in_blocks, kick_track, make_features, pads_only, silence,
                            speech_like, speech_over, sub_bass_pads, bass_swell)

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


def test_breakdown_with_sub_bass_still_detected():
    """Seen on a real episode: the kick goes but the sub-bass holds."""
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 30, bpm=138))
    held = feed_in_blocks(a, sub_bass_pads(RATE, 20))
    assert at(held, 12).section == "breakdown"
    back = feed_in_blocks(a, kick_track(RATE, 6, bpm=138))
    releases = [i * 1024 / RATE for i, f in enumerate(back) if f.release]
    assert len(releases) == 1 and releases[0] <= 2.0


def test_bass_swell_in_breakdown_is_not_a_release():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 30, bpm=138))
    feed_in_blocks(a, sub_bass_pads(RATE, 15))
    swell = feed_in_blocks(a, bass_swell(RATE, 12))
    assert not any(f.release for f in swell)
    assert swell[-1].section == "breakdown"


class StubModel:
    def __init__(self, section):
        self.section = section

    def proba(self, x):
        return {s: (0.9 if s == self.section else 0.025) for s in ("intro", "groove", "breakdown", "build", "climax")}


def test_model_cannot_force_an_impossible_section():
    a = Analyzer(RATE, model=StubModel("intro"))  # nothing goes back to an intro
    fs = feed_in_blocks(a, kick_track(RATE, 30))
    assert "intro" not in {f.section for f in fs[-300:]}


def test_model_proposal_is_followed_when_allowed():
    a = Analyzer(RATE, model=StubModel("breakdown"))
    fs = feed_in_blocks(a, kick_track(RATE, 30))
    assert fs[-1].section == "breakdown"


def test_a_sure_model_ends_an_intro_the_kick_test_missed():
    # A rolling bass hides the kick from the rules (seen on a real track: four
    # minutes of "intro"); a model sure of a build or a drop ends the intro.
    a = Analyzer(RATE, model=StubModel("build"))
    fs = feed_in_blocks(a, pads_only(RATE, 30))
    assert fs[-1].section == "build"


def test_a_model_calling_groove_or_breakdown_leaves_an_intro_alone():
    # The model never saw an intro (SSME has none), so it calls one groove or
    # breakdown: only the kick may end an intro then.
    for section in ("groove", "breakdown"):
        a = Analyzer(RATE, model=StubModel(section))
        fs = feed_in_blocks(a, pads_only(RATE, 30))
        assert fs[-1].section == "intro", section


class StubMelModel(StubModel):
    input = "mel"

    def __init__(self, section):
        super().__init__(section)
        self.shapes = []

    def proba(self, x):
        self.shapes.append(np.shape(x))
        return super().proba(x)


def test_spectrogram_model_gets_sixteen_seconds_of_mel():
    from amaran_mcp.music import sections
    stub = StubMelModel("breakdown")
    a = Analyzer(RATE, model=stub)
    fs = feed_in_blocks(a, kick_track(RATE, 30))
    assert stub.shapes and set(stub.shapes) == {(sections.MEL_FRAMES, 64)}
    assert fs[-1].section == "breakdown"


class StubBothModel(StubModel):
    input = "both"

    def __init__(self, section):
        super().__init__(section)
        self.inputs = []

    def proba(self, x):
        self.inputs.append((np.shape(x[0]), np.shape(x[1])))
        return super().proba(x)


def test_ensemble_model_gets_features_and_mel():
    from amaran_mcp.music import sections
    stub = StubBothModel("breakdown")
    a = Analyzer(RATE, model=stub)
    fs = feed_in_blocks(a, kick_track(RATE, 30))
    assert stub.inputs and set(stub.inputs) == {((sections.WINDOW_SIZE,), (sections.MEL_FRAMES, 64))}
    assert fs[-1].section == "breakdown"


class SureThenUnsure(StubModel):
    """Sure of one section for its first `sure` calls, then sure of nothing."""

    def __init__(self, section, sure):
        super().__init__(section)
        self.calls, self.sure = 0, sure

    def proba(self, x):
        self.calls += 1
        if self.calls <= self.sure:
            return super().proba(x)
        return {s: 0.2 for s in ("intro", "groove", "breakdown", "build", "climax")}


def test_rules_do_not_undo_a_section_before_its_minimum():
    # Seen on Miss You at 1:48: the model entered a build, went unsure two seconds
    # later, and the rules, hearing the kick, sent it straight back to a drop.
    from amaran_mcp.music import sections
    a = Analyzer(RATE, model=SureThenUnsure("build", sure=12))
    fs = feed_in_blocks(a, kick_track(RATE, 40))
    names = [f.section for f in fs]
    assert "build" in names
    start = names.index("build")
    lasted = next((i for i in range(start, len(names)) if names[i] != "build"), len(names)) - start
    assert lasted * 1024 / RATE >= sections.MIN_STAY - 0.5


class Sequence(StubModel):
    """Sure of one section for the first `first` calls, then of another."""

    def __init__(self, first_section, first, then_section):
        super().__init__(first_section)
        self.calls, self.first, self.then = 0, first, then_section

    def proba(self, x):
        self.calls += 1
        self.section = self.section if self.calls <= self.first else self.then
        return super().proba(x)


def test_a_rejected_proposal_does_not_hold_an_intro_back():
    # A model sure of a breakdown in an intro is ignored (an intro ends with a
    # build or a drop); it must not cost the next, valid proposal 8 s.
    a = Analyzer(RATE, model=Sequence("breakdown", 6, "build"))
    fs = feed_in_blocks(a, pads_only(RATE, 40))
    names = [f.section for f in fs]
    assert "build" in names
    assert names.index("build") * 1024 / RATE <= 14


# ── Between songs: each song is heard fresh, as training heard every track ─────

def test_a_new_tempo_after_a_gap_is_not_judged_at_the_old_one():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 120, bpm=138))
    feed_in_blocks(a, silence(RATE, 15))
    after = feed_in_blocks(a, kick_track(RATE, 30, bpm=170))
    assert any(f.section in ("groove", "build", "climax", "breakdown") for f in after[: int(10 * RATE / 1024)])


def test_the_same_song_after_a_long_silence_does_not_fake_a_build_or_drop():
    a = Analyzer(RATE)
    first = feed_in_blocks(a, kick_track(RATE, 150, bpm=138))
    feed_in_blocks(a, silence(RATE, 120))
    second = feed_in_blocks(a, kick_track(RATE, 60, bpm=138))
    assert {f.section for f in second} <= {f.section for f in first}
    assert not any(f.release for f in second)


def test_a_short_pause_inside_a_song_keeps_its_tempo():
    a = Analyzer(RATE)
    feed_in_blocks(a, kick_track(RATE, 60, bpm=138))
    tempo = a.tempo
    feed_in_blocks(a, silence(RATE, 1.7))  # one bar of silence before a drop
    feed_in_blocks(a, kick_track(RATE, 5, bpm=138))
    assert a.tempo is tempo and tempo.locked
