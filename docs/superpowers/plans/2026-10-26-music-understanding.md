# Music Understanding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Music mode understands trance structure (tempo and phrases, chords, vocal, tension, sections) and renders the owner's breakdown bloom, on top of the music mode already built.

**Architecture:** New pure analysis modules (`tempo.py`, `harmony.py`, `timbre.py`) feed an extended `Analyzer` whose `Features` carry section, events (bar, phrase, chord change, lift, release) and continuous measures (vocal, tension, openness, harmony balance). The renderer gains per-section behaviour and cool/warm chord colour sets per mood. A clip recorder and replay tool make tuning on real episodes repeatable offline.

**Tech Stack:** Python 3.12, numpy only for analysis; existing capture helper, runner, writer and MCP tools.

**Spec:** `docs/superpowers/specs/2026-10-26-music-mode-design.md` (section "Revision: musical understanding" and the owner decisions after it). Builds on `docs/superpowers/plans/2026-10-26-music-mode.md`, Tasks 1 to 7 complete.

## Global Constraints

- Anti-goals stay hard limits in the renderer: no flash, no hard cut, no per-beat pulse; brightness within the mood's floor..ceiling (plus offset); saturation never above the mood's maximum. The existing slew limits (hue 30 deg/s, saturation 20/s, brightness 12/s) stay.
- The bloom needs a chord lift during a breakdown; a sung vocal at that moment makes it bigger. A voice outside a breakdown never blooms; host speech over music does nothing.
- Continuous mixes: no outro section; "intro" is only the first music after silence; DJ blends count as groove.
- Structural colour moves (chord washes, section changes) start on the next bar or phrase boundary when tempo is known (bar if the next phrase is more than about 8 s away), immediately when it is not.
- Breakdowns start pale for their first bars, then colour builds with the harmony; the first version's 25% breakdown cap is removed.
- Chord stops within a cool or warm set sit 20 to 40 degrees apart; the major/minor balance over the last 8 to 16 bars picks the set; the chord root picks the stop.
- Tuning clips are recorded only to `CONFIG_DIR / "clips"` (outside the repo) and never uploaded.
- numpy only for analysis (no librosa, no ML model). No em-dashes or emojis in files.

## Review Focus

1. Host speech over an intro or a groove (A State of Trance voice-overs): no bloom, no section change caused by the speech. (Task 4 test, Task 5 test)
2. A DJ blend where two tracks at slightly different tempos overlap: tempo confidence drops and colour moves fall back to time-based starts instead of drifting off the bars; nothing crashes. (Task 2 test, Task 5 test)
3. A long breakdown with no chord change for 30 s or more: the light keeps drifting slowly within the chord's colour, never freezes. (Task 5 test)
4. A quiet breakdown (pads around -45 dBFS): still music, never "silence", so the light does not settle back mid-breakdown. (Task 4 test)
5. A key change up a whole step in the final build: detected as a lift-like rise only if it lands on a major chord; no bloom outside a breakdown. (Task 3 test, Task 5 test)

---

## File Structure

- `src/amaran_mcp/music/tempo.py`: onset envelope, BPM, beat phase, bar and phrase boundaries.
- `src/amaran_mcp/music/harmony.py`: chroma, chord matching, quality, balance, chord change and lift events.
- `src/amaran_mcp/music/timbre.py`: vocal estimate, tension, openness.
- `src/amaran_mcp/music/analysis.py`: modify; combines the above, new sections, new `Features`.
- `src/amaran_mcp/music/moods.py`, `render.py`: modify; cool/warm sets, per-section behaviour, bloom, boundary-timed moves.
- `src/amaran_mcp/music/runner.py`: modify; merge events between ticks, `--log-features`.
- `src/amaran_mcp/music/clips.py`: record and replay.
- `src/amaran_mcp/__init__.py`: CLI `music-record`, `music-replay`, `music --log-features`.
- Tests: `tests/test_music_tempo.py`, `tests/test_music_harmony.py`, `tests/test_music_timbre.py`, `tests/test_music_clips.py`; modify `tests/test_music_analysis.py`, `tests/test_music_render.py`, `tests/test_music_runner.py`; extend `tests/music_fixtures.py`.

New fixtures in `tests/music_fixtures.py` (each returns mono float32): `chord(rate, seconds, root, minor, octave=4)` (sustained triad with soft attack), `progression(rate, chords: list[tuple[str, bool]], seconds_each)`, `sung(rate, seconds, notes)` (harmonic series with formants and 5.5 Hz vibrato, sustained notes; tests use female-range notes, 330 to 880 Hz), `speech_over(rate, seconds, bed)` (speech_like mixed over a bed), `riser(rate, seconds)` (noise sweep with rising high-pass and accelerating snare hits), `arrangement(rate)` (silence 2 s, intro 16 s hats and pads, groove 30 s, breakdown 40 s chords Am-F-C-G, build 16 s riser, climax 30 s, groove 10 s) returning `(samples, marks)` where marks maps section name to (start_s, end_s).

---

### Task 1: Record and replay clips

**Files:**
- Create: `src/amaran_mcp/music/clips.py`, `tests/test_music_clips.py`
- Modify: `src/amaran_mcp/__init__.py`

**Interfaces:**
- Consumes: `capture.SystemAudio` (existing).
- Produces:

```python
CLIPS_DIR = CONFIG_DIR / "clips"
async def record(seconds: float, name: str, source: Source | None = None) -> Path  # 16-bit mono WAV in CLIPS_DIR
def load(path: Path) -> tuple[np.ndarray, int]                                     # float32 mono, rate
def replay(path: Path, every: float = 1.0) -> list[dict]   # one row per `every` seconds of features (dataclass asdict + "t")
```

CLI: `amaran-mcp music-record SECONDS NAME` (prints the path), `amaran-mcp music-replay PATH` (prints a compact table: t, section, tempo, chord, balance, vocal, tension, openness, events).

- [ ] **Step 1: Write the failing tests**

```python
def test_record_writes_wav_in_clips_dir(home):
    path = asyncio.run(clips.record(1.0, "test", source=FakeSource([kick_track(24000, 2)])))
    assert path.parent == clips.CLIPS_DIR and path.suffix == ".wav"
    samples, rate = clips.load(path)
    assert rate == 24000 and abs(len(samples) - 24000) <= 1024

def test_replay_reports_sections_per_second(home):
    path = write_wav(home, kick_track(24000, 12), 24000)  # test helper
    rows = clips.replay(path)
    assert len(rows) in (11, 12) and rows[-1]["section"] in ("intro", "groove")
    assert {"t", "section", "energy"} <= rows[0].keys()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_clips.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `clips.py` and the two CLI commands** (`wave` module from the standard library)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_clips.py -v`
Expected: PASS

- [ ] **Step 5: Commit** `Add music clip recording and replay for tuning`

---

### Task 2: Tempo, bars and phrases

**Files:**
- Create: `src/amaran_mcp/music/tempo.py`, `tests/test_music_tempo.py`

**Interfaces:**
- Produces:

```python
@dataclass
class Beat:
    bpm: float | None        # None until confident
    confidence: float        # 0..1
    bar: bool                # a bar boundary fell in this block
    phrase: bool             # an 8-bar boundary fell in this block
    seconds_to_bar: float | None
    seconds_to_phrase: float | None

class TempoTracker:
    def __init__(self, rate: int) -> None: ...
    def feed(self, samples: np.ndarray) -> Beat: ...
```

Onset envelope from low-band spectral flux per block; BPM from the autocorrelation of the last ~8 s of envelope, searched in 100 to 150 BPM; beat phase from the envelope peak alignment; bar = 4 beats; phrase = 8 bars counted from the first confident downbeat. Confidence falls when the autocorrelation peak weakens; below 0.4, `bpm` is None and no boundaries are reported.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("bpm", [125, 138])
def test_tempo_locks_within_one_bpm(bpm):
    beats = feed_in_blocks(TempoTracker(24000), kick_track(24000, 30, bpm=bpm))
    assert beats[-1].bpm == pytest.approx(bpm, abs=1)

def test_bars_every_four_beats_and_phrases_every_eight_bars():
    beats = feed_in_blocks(TempoTracker(24000), kick_track(24000, 80, bpm=138))
    bar_times = [i * 1024 / 24000 for i, b in enumerate(beats) if b.bar]
    gaps = np.diff(bar_times[3:])
    assert np.allclose(gaps, 4 * 60 / 138, atol=0.05)
    phrases = [b for b in beats if b.phrase]
    assert 1 <= len(phrases) <= 3   # 80 s at 138 BPM is about 5.75 phrases of 13.9 s, minus lock-in

def test_no_kick_means_no_tempo():
    assert feed_in_blocks(TempoTracker(24000), pads_only(24000, 20))[-1].bpm is None

def test_blend_of_two_tempos_drops_confidence():
    mix = kick_track(24000, 20, bpm=128) * 0.6 + kick_track(24000, 20, bpm=134) * 0.6
    tail = feed_in_blocks(TempoTracker(24000), mix)[-50:]
    assert min(b.confidence for b in tail) < 0.4
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_tempo.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `TempoTracker`**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_tempo.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit** `Add tempo, bar and phrase tracking`

---

### Task 3: Harmony

**Files:**
- Create: `src/amaran_mcp/music/harmony.py`, `tests/test_music_harmony.py`
- Modify: `tests/music_fixtures.py` (chord, progression)

**Interfaces:**
- Produces:

```python
@dataclass
class Harmony:
    chord: str | None        # e.g. "Am", "F", None when confidence is low
    root: int | None         # pitch class 0..11 (C = 0)
    major: bool | None
    confidence: float        # 0..1
    balance: float           # -1 all minor .. +1 all major, over the last ~12 bars (or ~25 s without tempo)
    change: bool             # the chord changed in this block (after smoothing)
    lift: bool               # a change onto a major chord, or balance rising by >= 0.3 within ~2 s, confidence >= 0.5

class HarmonyTracker:
    def __init__(self, rate: int) -> None: ...
    def feed(self, samples: np.ndarray) -> Harmony: ...
```

12-bin chroma from the FFT magnitude (bins mapped to pitch classes, 55 Hz to 2 kHz), smoothed over ~1 s; cosine match against 24 triad templates; a change needs the new best chord to hold for ~0.5 s.

- [ ] **Step 1: Write the failing tests**

```python
def test_identifies_a_minor_progression():
    seq = [("A", True), ("F", False), ("C", False), ("G", False)]
    hs = feed_in_blocks(HarmonyTracker(24000), progression(24000, seq, 4.0))
    seen = [h.chord for h in hs if h.change]
    assert seen[:4] == ["Am", "F", "C", "G"]

def test_lift_on_minor_to_major():
    hs = feed_in_blocks(HarmonyTracker(24000), progression(24000, [("A", True), ("C", False)], 5.0))
    lifts = [i for i, h in enumerate(hs) if h.lift]
    assert len(lifts) == 1 and lifts[0] * 1024 / 24000 >= 5.0

def test_whole_step_key_change_onto_minor_is_not_a_lift():
    hs = feed_in_blocks(HarmonyTracker(24000), progression(24000, [("A", True), ("B", True)], 5.0))
    assert not any(h.lift for h in hs)

def test_noise_has_no_confident_chord():
    hs = feed_in_blocks(HarmonyTracker(24000), riser(24000, 8))
    assert all(h.chord is None or h.confidence < 0.5 for h in hs[-40:])

def test_balance_tracks_major_minor_mix():
    minor = feed_in_blocks(HarmonyTracker(24000), progression(24000, [("A", True), ("D", True)] * 3, 4.0))
    assert minor[-1].balance < -0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_harmony.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `HarmonyTracker` and the chord fixtures**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_harmony.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit** `Add chord, balance and lift tracking`

---

### Task 4: Vocal, tension and openness

**Files:**
- Create: `src/amaran_mcp/music/timbre.py`, `tests/test_music_timbre.py`
- Modify: `tests/music_fixtures.py` (sung, speech_over, riser)

**Interfaces:**
- Produces:

```python
@dataclass
class Timbre:
    vocal: float     # 0..1, smoothed (rise ~0.7 s, fall ~1.5 s)
    tension: float   # 0..1, relative to the last ~30 s
    openness: float  # 0..1, spectral centroid within the track's recent range
    level_db: float  # block level in dBFS, for the quiet-breakdown rule

class TimbreTracker:
    def __init__(self, rate: int) -> None: ...
    def feed(self, samples: np.ndarray) -> Timbre: ...
```

Vocal: peak-tracked harmonic energy in 200 Hz to 4 kHz that shows pitch modulation at 4 to 7 Hz or glides, relative to the sustained bed (a slow median of the band). Tension: rising spectral flatness, rising high-band share and rising onset density, each against its 30 s average, combined and smoothed over ~2 s. Openness: centroid normalised to its 30 s min..max.

- [ ] **Step 1: Write the failing tests**

```python
def test_sung_vocal_over_pads_reads_high():
    mix = pads_only(24000, 12) + sung(24000, 12, notes=[440, 494, 523])
    assert np.mean([t.vocal for t in feed_in_blocks(TimbreTracker(24000), mix)[-100:]]) > 0.6

def test_static_pads_read_low():
    assert np.mean([t.vocal for t in feed_in_blocks(TimbreTracker(24000), pads_only(24000, 12))[-100:]]) < 0.2

def test_riser_raises_tension_steadily():
    ts = feed_in_blocks(TimbreTracker(24000), np.concatenate([kick_track(24000, 30), riser(24000, 16)]))
    build = [t.tension for t in ts[-300:]]
    assert build[-1] > 0.7 and build[-1] - build[0] > 0.4

def test_quiet_breakdown_level_is_reported():
    quiet = pads_only(24000, 10) * 0.05   # about -45 dBFS
    assert -50 < feed_in_blocks(TimbreTracker(24000), quiet)[-1].level_db < -40

def test_host_speech_over_groove_is_not_tension():
    ts = feed_in_blocks(TimbreTracker(24000), speech_over(24000, 20, kick_track(24000, 20)))
    assert max(t.tension for t in ts[-200:]) < 0.5
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_timbre.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `TimbreTracker` and the fixtures**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_timbre.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit** `Add vocal, tension and openness measures`

---

### Task 5: Sections and the new Features

**Files:**
- Modify: `src/amaran_mcp/music/analysis.py`, `tests/test_music_analysis.py`, `tests/music_fixtures.py` (arrangement)

**Interfaces:**
- Consumes: `TempoTracker`/`Beat` (Task 2), `HarmonyTracker`/`Harmony` (Task 3), `TimbreTracker`/`Timbre` (Task 4).
- Produces (replaces the first version's `Features`; `drop` becomes `release`):

```python
@dataclass
class Features:
    energy: float; bass: float; is_music: bool; digital_zero: bool
    section: str             # "silence" | "intro" | "groove" | "breakdown" | "build" | "climax"
    section_seconds: float   # time in the current section
    release: bool            # build or breakdown -> climax (kick returns with energy)
    bloom: bool              # harmony.lift while section == "breakdown"
    beat: Beat
    harmony: Harmony
    timbre: Timbre

def merge_events(earlier: Features, later: Features) -> Features
    # later's values, with release, bloom, beat.bar, beat.phrase and harmony.change/lift OR-ed in
```

Sections: intro = music since silence without the kick for its first stretch; groove = kick and bass present at usual energy; breakdown = kick gone for 4+ bars (or 8+ s without tempo) after groove or climax; build = tension above 0.6 and rising for 4+ bars; climax = after a release, until energy falls back to its groove level for 8 bars. Silence keeps the first version's rule, but a quiet passage above -55 dBFS that is still music is never silence.

- [ ] **Step 1: Replace the breakdown/drop test and write the section tests**

```python
def test_arrangement_sections_in_order():
    samples, marks = arrangement(24000)
    fs = feed_in_blocks(Analyzer(24000), samples)
    at = lambda s: fs[int(s * 24000 / 1024)].section
    assert at(marks["intro"][1] - 2) == "intro"
    assert at(marks["groove"][1] - 2) == "groove"
    assert at(marks["breakdown"][0] + 15) == "breakdown"
    assert at(marks["build"][1] - 2) == "build"
    assert at(marks["climax"][0] + 10) == "climax"
    assert sum(f.release for f in fs) == 1

def test_bloom_only_on_lift_inside_breakdown():
    samples, marks = arrangement(24000)          # breakdown is Am-F-C-G: Am -> C... lifts happen there
    fs = feed_in_blocks(Analyzer(24000), samples)
    blooms = [i * 1024 / 24000 for i, f in enumerate(fs) if f.bloom]
    assert blooms and all(marks["breakdown"][0] <= t <= marks["build"][0] for t in blooms)

def test_host_speech_over_intro_changes_nothing():
    a, b = Analyzer(24000), Analyzer(24000)
    bed = kick_track(24000, 30)
    plain = feed_in_blocks(a, bed)
    talked = feed_in_blocks(b, speech_over(24000, 30, bed))
    assert [f.section for f in plain[-200:]] == [f.section for f in talked[-200:]]
    assert not any(f.bloom for f in talked)

def test_quiet_breakdown_is_not_silence():
    a = Analyzer(24000)
    feed_in_blocks(a, kick_track(24000, 30))
    fs = feed_in_blocks(a, pads_only(24000, 25) * 0.05)
    assert all(f.section != "silence" for f in fs)

def test_merge_events_keeps_one_tick_events():
    ...  # two Features, the earlier with bloom=True and beat.bar=True, the later without: merged has both True
```

Keep `test_kick_track_is_music_with_steady_energy`, `test_speech_is_not_music` and `test_silence_after_ten_seconds` unchanged.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run pytest tests/test_music_analysis.py -v`
Expected: new tests FAIL; the three kept tests PASS

- [ ] **Step 3: Implement the sections, `Features` and `merge_events`; build the `arrangement` fixture**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit** `Add trance sections, bloom events and merged features`

---

### Task 6: Renderer behaviour by section

**Files:**
- Modify: `src/amaran_mcp/music/moods.py`, `src/amaran_mcp/music/render.py`, `tests/test_music_render.py`

**Interfaces:**
- Consumes: new `Features` (Task 5).
- Produces: `Mood` gains `cool: tuple[tuple[float, float], ...]` and `warm: tuple[tuple[float, float], ...]` (hue, saturation stops 20 to 40 degrees apart; ethereal cool around 220 to 270, warm around 290 to 340 plus a soft gold near 40); `Renderer.step(f, dt)` unchanged in signature.

Behaviour (all through the existing slew limits):
- intro: brightness in the lower third of the range, drift as groove.
- groove: the first version's drift and breathing.
- breakdown: first 2 bars (or 4 s) pale (saturation <= 25); then colour family from `harmony.balance` (< 0 cool, >= 0 warm) and stop from `harmony.root` (root mod number of stops); a chord change moves to its stop starting at the next bar (immediately without tempo); with no chord change for 30 s, drift slowly within the family.
- bloom: on `f.bloom`, over the next ~4 s: hue toward the warm family, brightness toward the ceiling by 30 to 60% of the remaining headroom (scaled by `timbre.vocal`), saturation eased down by up to 15 (luminous); eases back over ~4 s after the vocal falls below 0.3 or 2 phrases pass.
- build: drift speed falls toward 0 as tension rises; saturation toward the mood maximum; brightness held.
- release and climax: brightness toward the ceiling, saturation full, drift resumes faster (x1.3).
- openness (groove, climax, breakdown after its pale start): up to 15 points less saturation and up to 10 points more brightness at full openness, within the range: a filter opening reads as luminous and airy.

- [ ] **Step 1: Replace `test_breakdown_goes_pale` and the drop tests; write behaviour tests**

```python
def test_breakdown_starts_pale_then_takes_chord_colour():
    r = Renderer(MOODS["ethereal"])
    run(r, groove_f(), 10)
    early = run(r, breakdown_f(seconds=1, chord_root=9, balance=-0.8), 3)[-1]
    assert early.saturation <= 25
    later = run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.8), 6)[-1]
    assert later.saturation > 35 and 210 <= later.hue <= 280      # cool family

def test_major_balance_moves_to_warm_family():
    r = Renderer(MOODS["ethereal"])
    run(r, breakdown_f(seconds=8, chord_root=0, balance=0.8), 10)
    hue = run(r, breakdown_f(seconds=12, chord_root=0, balance=0.8), 4)[-1].hue
    assert 285 <= hue <= 345 or 30 <= hue <= 50

def test_bloom_swells_and_eases_back():
    r = Renderer(MOODS["ethereal"])
    base = run(r, breakdown_f(seconds=8, balance=-0.5), 8)[-1]
    peak = max(o.brightness for o in [r.step(breakdown_f(seconds=9, bloom=True, vocal=0.9), 1 / 8)]
               + run(r, breakdown_f(seconds=10, vocal=0.9), 4))
    assert peak >= base.brightness + 8
    after = run(r, breakdown_f(seconds=20, vocal=0.0), 8)[-1]
    assert after.brightness <= base.brightness + 3

def test_build_gathers_drift_stops():
    r = Renderer(MOODS["ethereal"])
    hues = [o.hue for o in run(r, build_f(tension=0.95), 12)]
    assert max(hues[-16:]) - min(hues[-16:]) < 2

def test_chord_change_waits_for_bar():
    r = Renderer(MOODS["ethereal"])
    run(r, breakdown_f(seconds=8, chord_root=9, balance=-0.8), 6)
    before = r.step(breakdown_f(seconds=9, chord_root=5, balance=-0.8, change=True, seconds_to_bar=1.0), 1 / 8).hue
    held = run(r, breakdown_f(seconds=9.2, chord_root=5, balance=-0.8, seconds_to_bar=0.7), 0.5)[-1].hue
    assert abs(held - before) < 1.5                                 # waits for the bar

def test_intro_is_restrained():
    r = Renderer(MOODS["ethereal"])
    low, high = MOODS["ethereal"].brightness
    assert run(r, intro_f(energy=1.0), 20)[-1].brightness <= low + (high - low) / 3 + 1e-9

def test_release_opens_to_the_ceiling():
    r = Renderer(MOODS["ethereal"])
    run(r, build_f(tension=0.9), 8)
    outs = [r.step(climax_f(release=True), 1 / 8)] + run(r, climax_f(), 6)
    assert outs[-1].brightness >= MOODS["ethereal"].brightness[1] - 3

def test_openness_reads_luminous():
    closed = run(Renderer(MOODS["ethereal"]), groove_f(openness=0.0), 15)[-1]
    open_ = run(Renderer(MOODS["ethereal"]), groove_f(openness=1.0), 15)[-1]
    assert open_.saturation < closed.saturation and open_.brightness > closed.brightness

def test_long_breakdown_without_changes_keeps_drifting():
    r = Renderer(MOODS["ethereal"])
    hues = [o.hue for o in run(r, breakdown_f(seconds=40, chord_root=9, balance=-0.8), 30)]
    assert max(hues) - min(hues) > 3
```

Keep the slew, bounds and brightness-offset tests unchanged. `intro_f`, `groove_f`, `breakdown_f`, `build_f`, `climax_f` are test helpers building `Features` with the named fields.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_render.py -v`
Expected: new tests FAIL; kept tests PASS

- [ ] **Step 3: Implement the mood sets and the per-section renderer**

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_music_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit** `Render trance sections: chord colour, bloom, build and release`

---

### Task 7: Runner integration, feature log, live tuning

**Files:**
- Modify: `src/amaran_mcp/music/runner.py`, `src/amaran_mcp/__init__.py`, `tests/test_music_runner.py`, `README.md`

**Interfaces:**
- Consumes: `Features`, `merge_events` (Task 5); renderer (Task 6); `clips` (Task 1).
- Produces: `RunnerConfig.log_features: Path | None`; CLI `music --log-features PATH`.

- [ ] **Step 1: Update runner tests** for the new fields (replace `_drop` with `merge_events`) and add:

```python
def test_feature_log_writes_one_line_per_tick(runner, tmp_path):
    log = tmp_path / "features.jsonl"
    asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=5, log_features=log),
                           source=FakeSource([kick_track(RATE, 10), silence(RATE, 20)]),
                           light=lambda: FakeLight(), music=lambda: None))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) > 50 and {"t", "section", "output"} <= rows[0].keys()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_music_runner.py -v`
Expected: the new test FAILS; updated tests PASS after Step 3

- [ ] **Step 3: Implement** event merging between ticks and the feature log; run the whole suite

Run: `uv run pytest -q`
Expected: all PASS

- [ ] **Step 4: Record and tune on real episodes with the owner**

Record 3 to 5 clips of 2 to 4 minutes each with `music-record` around breakdowns the owner picks (ASOT, Find Your Harmony, FSOE). The owner named the vocal set: JES and other female vocals on Armin van Buuren tracks from the last ten years; include at least two of those so the vocal estimate is tuned on them. Clips are tuning material only (no model is trained) and stay in `CONFIG_DIR / "clips"`. Replay each with `music-replay`, compare the sections, lifts and blooms with where the owner hears them, and tune thresholds and time constants until they line up. Then run music mode live on an episode with the owner watching and ask: does the breakdown follow the chords, does the bloom land when the chords lift and the vocal comes in, does the build gather and the release open, and does anything flash or pulse. Re-run `uv run pytest -q` after every tuning change.

- [ ] **Step 5: README** "Music mode" section updated: sections, the bloom, continuous mixes, clip recording (local only) and the feature log.

- [ ] **Step 6: Commit and push**

```bash
git add -A && git commit -m "Integrate trance understanding into music mode; tune on real episodes"
git push origin music-mode
```
