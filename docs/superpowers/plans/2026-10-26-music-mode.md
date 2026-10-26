# Music Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A background "music mode" that listens to what the Mac is playing and moves the amaran PT2c through slow, ethereal colour, started and stopped from Claude.

**Architecture:** A small Swift helper captures system audio with a Core Audio process tap and streams PCM to a Python process. The Python side analyses energy and song structure (numpy), renders slew-limited HSI colour from a mood palette, and drives the light through the existing `backends.current_light()`. The MCP server spawns it as a detached process and talks to it through files in `~/.config/amaran-mcp/`.

**Tech Stack:** Python 3.12 (uv), numpy, sounddevice (microphone fallback only), Swift 6 / Core Audio process taps (macOS 14.2+), osascript for the Music app, existing amaran-mcp backends.

**Spec:** `docs/superpowers/specs/2026-10-26-music-mode-design.md`

## Global Constraints

- macOS, one light (the configured PT2c). Process taps need macOS 14.2 or later.
- Never open a Bluetooth input device (AirPods): opening one switches them to the call codec. This holds even when it is the default input.
- Anti-goals are hard limits enforced in the renderer: no flash, no hard cut, no visible per-beat pulse; brightness never outside the mood's floor..ceiling (default 25% to 70%); saturation never above the mood's maximum.
- Timing values from the spec: energy rise about 1.5 s, fall about 4 s; silence after about 10 s; exit after 15 minutes of continuous silence; drop crossfade 2 to 3 s; palette loop 45 to 90 s; breakdown saturation 20 to 30%; settle fade about 5 s; updates about 8/s through amaran Desktop, about 3/s over Bluetooth; Apple Music polled every few seconds.
- The Music app is read only if it is already running (never launch it).
- Files the process shares live in `CONFIG_DIR` (`~/.config/amaran-mcp`), written with `config._write_private` (mode 600).
- House style: no em-dashes and no emojis in any file content. Logs go to stderr or the log file, never stdout of the MCP server.
- Change from the spec (builder's call, flagged to the owner at handoff): the ScreenCaptureKit fallback is dropped. If the tap cannot be created, music mode uses the built-in microphone fallback instead.

## Review Focus

1. Music app not running, or Automation permission refused: track info is simply absent; Music is never launched and nothing crashes. (Task 5 test)
2. Capture permission refused: the tap delivers exact zeros. If the Music app says it is playing while capture has been exact zeros for 10 s, music mode exits with a message naming System Settings, Privacy and Security, Screen and System Audio Recording, Terminal; otherwise it just reports "listening". (Task 6 test)
3. amaran Desktop quits (or starts) mid-session: the next tick goes through whichever backend `current_light()` picks; a `DesktopError` or `LinkError` on one tick is logged and retried, never fatal. (Task 6 test)
4. `start_music_mode` while it is already running: the old process is stopped and one new instance runs with the new mood. (Task 7 test)
5. Stale PID file after a crash or reboot: status reports not running and start works. (Task 7 test)

---

## File Structure

- `src/amaran_mcp/music/__init__.py`: package marker, re-exports nothing.
- `src/amaran_mcp/music/systemtap.swift`: Swift helper source (system audio capture).
- `src/amaran_mcp/music/Info.plist`: embedded into the helper (audio capture usage string).
- `src/amaran_mcp/music/capture.py`: building/launching the helper, reading PCM, and the microphone fallback.
- `src/amaran_mcp/music/analysis.py`: pure audio features and section detection.
- `src/amaran_mcp/music/moods.py`: the three moods.
- `src/amaran_mcp/music/render.py`: pure renderer from features to slew-limited HSI.
- `src/amaran_mcp/music/applemusic.py`: Music app reader.
- `src/amaran_mcp/music/runner.py`: the music-mode loop.
- `src/amaran_mcp/music/control.py`: start/stop/status from other processes.
- `src/amaran_mcp/__main__.py`: `python -m amaran_mcp` entry (needed to spawn the runner).
- Modify `src/amaran_mcp/__init__.py` (CLI), `src/amaran_mcp/server.py` (tools), `src/amaran_mcp/light.py` (`LightBase.apply`), `README.md`.
- Tests: `tests/test_music_analysis.py`, `tests/test_music_render.py`, `tests/test_music_capture.py`, `tests/test_music_applemusic.py`, `tests/test_music_runner.py`, `tests/test_music_control.py`, plus `tests/music_fixtures.py` (synthetic audio).

---

### Task 1: System audio helper and the AirPlay risk check

The spec makes this the first thing built: prove a system-wide tap hears normal playback and AirPlay Receiver audio before anything else.

**Files:**
- Create: `src/amaran_mcp/music/__init__.py`, `src/amaran_mcp/music/systemtap.swift`, `src/amaran_mcp/music/Info.plist`, `src/amaran_mcp/music/capture.py` (build and probe parts only)
- Modify: `src/amaran_mcp/__init__.py` (add `music-probe` command)
- Test: `tests/test_music_capture.py`

**Interfaces:**
- Produces: helper protocol: stdout line 1 is `RATE <int>\n`, then raw little-endian float32 mono samples until SIGTERM. Exit code 3 with a message on stderr if the tap cannot be created.
- Produces: `capture.helper_path() -> Path` (builds `CONFIG_DIR / "bin" / "amaran-systemtap"` with `swiftc -O` when missing or when the source's sha256 differs from `CONFIG_DIR / "bin" / "amaran-systemtap.sha256"`; raises `CaptureError` with the compiler output on failure).
- Produces: `class CaptureError(RuntimeError)`.

- [ ] **Step 1: Write the failing test**

```python
def test_helper_builds_and_reports_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path))
    from amaran_mcp.music import capture
    importlib.reload(capture)
    path = capture.helper_path()
    assert path.exists() and os.access(path, os.X_OK)
    assert capture.helper_path() == path  # second call does not rebuild
    assert (tmp_path / "bin" / "amaran-systemtap.sha256").read_text().strip()
```

Mark it `@pytest.mark.skipif(shutil.which("swiftc") is None, reason="needs Xcode")`.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_music_capture.py -v`
Expected: FAIL (module `amaran_mcp.music.capture` not found)

- [ ] **Step 3: Write the helper `systemtap.swift`**

Core Audio steps, in order: `CATapDescription(monoGlobalTapButExcludeProcesses: [])` with `isPrivate = true` and `muteBehavior = .unmuted`; `AudioHardwareCreateProcessTap`; read the tap's `kAudioTapPropertyUID` and `kAudioTapPropertyFormat`; create a private aggregate device (`AudioHardwareCreateAggregateDevice`) whose `kAudioAggregateDeviceTapListKey` holds that UID with `kAudioAggregateDeviceTapAutoStartKey` true; `AudioDeviceCreateIOProcIDWithBlock` writing the input buffer's float32 samples to stdout (mix channels to mono if the format has more than one); `AudioDeviceStart`. Print `RATE <sample rate>` first. On SIGTERM or SIGINT, stop the device, destroy the IOProc, the aggregate device and the tap, then exit 0. Any OSStatus failure during setup: message to stderr, exit 3.

`Info.plist` holds `CFBundleIdentifier` `com.jtannahill.amaran-systemtap` and `NSAudioCaptureUsageDescription` "amaran-mcp listens to what this Mac is playing to move your light with the music." Build flags: `swiftc -O systemtap.swift -o <out> -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist`.

- [ ] **Step 4: Implement `helper_path()` in `capture.py`**, run the test

Run: `uv run pytest tests/test_music_capture.py -v`
Expected: PASS

- [ ] **Step 5: Add `amaran-mcp music-probe [SECONDS]`**

Runs the helper for SECONDS (default 10) and prints one line per second: RMS in dBFS and peak. Prints the helper's stderr if it exits early.

- [ ] **Step 6: Manual check with the owner (stop here if it fails)**

1. `afplay /System/Library/Sounds/Submarine.aiff` in a loop while running `uv run amaran-mcp music-probe 10`, with AirPods as the output. Expected: macOS asks Terminal for system audio recording permission once; afterwards RMS rises well above -60 dBFS while the sound plays and the AirPods keep normal audio quality.
2. Ask the owner to AirPlay music from the iPhone to this Mac, then run the probe again. Expected: RMS follows the music.

If step 2 shows silence, stop and report to the owner before Task 2; the iPhone path in the spec depends on it.

- [ ] **Step 7: Commit**

```bash
git add src/amaran_mcp/music tests/test_music_capture.py src/amaran_mcp/__init__.py
git commit -m "Add system audio capture helper and music-probe"
```

---

### Task 2: Audio analysis

**Files:**
- Create: `src/amaran_mcp/music/analysis.py`, `tests/music_fixtures.py`, `tests/test_music_analysis.py`
- Modify: `pyproject.toml` (`uv add numpy`)

**Interfaces:**
- Produces:

```python
@dataclass
class Features:
    energy: float        # 0..1, smoothed (rise ~1.5 s, fall ~4 s), relative to long-term level
    bass: float          # 0..1, smoothed bass-band level relative to its long-term level
    is_music: bool       # sustained bass energy present (speech and calls are not music)
    section: str         # "silence" | "breakdown" | "playing"
    drop: bool           # True on exactly one update when a drop is detected
    digital_zero: bool   # every sample in the last block was exactly 0.0

class Analyzer:
    def __init__(self, rate: int) -> None: ...
    def feed(self, samples: np.ndarray) -> Features: ...  # any block length; mono float32
```

- `tests/music_fixtures.py` produces `kick_track(rate, seconds, bpm=125, pads=True) -> np.ndarray`, `pads_only(rate, seconds)`, `speech_like(rate, seconds)` (band-limited 300 to 3000 Hz noise in syllable-length bursts, no bass), `silence(rate, seconds)`, and `feed_in_blocks(analyzer, samples, block=1024) -> list[Features]`.

- [ ] **Step 1: Write the failing tests** (rate 24000)

```python
def test_kick_track_is_music_with_steady_energy():
    fs = feed_in_blocks(Analyzer(24000), kick_track(24000, 30))
    tail = fs[len(fs) // 2:]
    assert all(f.is_music for f in tail)
    assert all(f.section == "playing" for f in tail)
    energies = [f.energy for f in tail]
    assert max(energies) - min(energies) < 0.15  # breathes, does not pulse per kick

def test_speech_is_not_music():
    fs = feed_in_blocks(Analyzer(24000), speech_like(24000, 20))
    assert not any(f.is_music for f in fs[len(fs) // 2:])

def test_silence_after_ten_seconds():
    a = Analyzer(24000)
    feed_in_blocks(a, kick_track(24000, 20))
    fs = feed_in_blocks(a, silence(24000, 12))
    seconds = [i * 1024 / 24000 for i, f in enumerate(fs) if f.section == "silence"]
    assert seconds and 8.0 <= seconds[0] <= 11.0
    assert fs[-1].digital_zero

def test_breakdown_then_one_drop():
    a = Analyzer(24000)
    feed_in_blocks(a, kick_track(24000, 40))
    breakdown = feed_in_blocks(a, pads_only(24000, 16))
    assert breakdown[-1].section == "breakdown"
    after = feed_in_blocks(a, kick_track(24000, 10))
    assert sum(f.drop for f in after) == 1
    assert sum(f.drop for f in breakdown) == 0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_analysis.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `Analyzer`**

Per block: RMS, FFT band energies (bass 40 to 150 Hz, mid 150 to 2000 Hz, high 2000 Hz+), spectral flux. Asymmetric one-pole followers with the spec's rise and fall times; long-term reference levels over about 30 s for normalisation. Music: bass follower above a floor relative to the long-term level for about 2 s. Silence: RMS below about -60 dBFS for 10 s. Breakdown: music present but bass below about 40% of its long-term level for about 4 s. Drop: on leaving a breakdown when bass returns above about 80% of its long-term level; fire once.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_music_analysis.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/amaran_mcp/music/analysis.py tests/music_fixtures.py tests/test_music_analysis.py
git commit -m "Add music analysis: energy, music detection, breakdowns and drops"
```

---

### Task 3: Moods and renderer

**Files:**
- Create: `src/amaran_mcp/music/moods.py`, `src/amaran_mcp/music/render.py`, `tests/test_music_render.py`

**Interfaces:**
- Consumes: `analysis.Features` (Task 2).
- Produces:

```python
@dataclass(frozen=True)
class Mood:
    name: str
    anchors: tuple[tuple[float, float], ...]  # (hue degrees, saturation %) palette stops
    brightness: tuple[float, float]           # floor, ceiling in %
    max_saturation: float
    loop_seconds: float                       # one trip around the palette

MOODS: dict[str, Mood]   # "ethereal" (default), "deep", "dawn"
DEFAULT_MOOD = "ethereal"

@dataclass
class Output:
    hue: float; saturation: float; brightness: float

class Renderer:
    def __init__(self, mood: Mood, start_anchor: int = 0, brightness_offset: float = 0.0) -> None: ...
    def step(self, f: Features, dt: float) -> Output: ...
    def jump_to_anchor(self, index: int) -> None: ...   # crossfades, used on track change
    def current_target_hue_after_drop(self) -> float: ...  # hue of the next anchor; for tests
    brightness_offset: float                           # set by "brighter"/"dimmer"; shifts floor..ceiling
```

Mood values (tuning may adjust): ethereal anchors deep blue (225, 70), violet (270, 60), soft teal (185, 45), misty white (220, 12); brightness 25..70; max saturation 75; loop 75 s. deep: indigo (245, 80), navy (220, 85), magenta-violet (290, 65); 20..55; 85; loop 90 s. dawn: lavender (265, 35), peach (25, 45), pale gold (45, 40), warm white (35, 10); 30..70; 55; loop 60 s.

Slew limits (per second): hue 30 degrees, saturation 20 points, brightness 12 points. They are the anti-goal guard.

- [ ] **Step 1: Write the failing tests**

```python
def run(renderer, features, seconds, hz=8):
    return [renderer.step(features, 1 / hz) for _ in range(int(seconds * hz))]

def test_slew_limits_hold_under_chaotic_input():
    r = Renderer(MOODS["ethereal"])
    rng = random.Random(1)
    outs = []
    for _ in range(8 * 120):
        f = Features(energy=rng.random(), bass=rng.random(), is_music=True,
                     section=rng.choice(["playing", "breakdown"]), drop=rng.random() < 0.02,
                     digital_zero=False)
        outs.append(r.step(f, 1 / 8))
    for a, b in zip(outs, outs[1:]):
        assert abs(b.brightness - a.brightness) <= 12 / 8 + 1e-9
        assert abs(b.saturation - a.saturation) <= 20 / 8 + 1e-9
        assert min(abs(b.hue - a.hue), 360 - abs(b.hue - a.hue)) <= 30 / 8 + 1e-9

def test_bounds_never_leave_the_mood():
    for mood in MOODS.values():
        r = Renderer(mood)
        for f in (LOUD, QUIET, BREAKDOWN, DROP):
            for o in run(r, f, 20):
                assert mood.brightness[0] - 1e-9 <= o.brightness <= mood.brightness[1] + 1e-9
                assert o.saturation <= mood.max_saturation + 1e-9

def test_breakdown_goes_pale():
    r = Renderer(MOODS["ethereal"])
    run(r, LOUD, 20)
    assert run(r, BREAKDOWN, 8)[-1].saturation <= 30

def test_drop_reaches_next_anchor_within_three_seconds():
    r = Renderer(MOODS["ethereal"])
    run(r, LOUD, 10)
    target = r.current_target_hue_after_drop()  # helper exposed for tests
    outs = run(r, DROP_ONCE_THEN_LOUD, 3.5)
    assert min(abs(outs[-1].hue - target), 360 - abs(outs[-1].hue - target)) < 5

def test_brightness_offset_shifts_range():
    r = Renderer(MOODS["ethereal"], brightness_offset=15)
    assert all(40 - 1e-9 <= o.brightness <= 85 + 1e-9 for o in run(r, LOUD, 30))
```

(`LOUD`, `QUIET`, `BREAKDOWN`, `DROP`, `DROP_ONCE_THEN_LOUD` are module-level fixtures: `DROP_ONCE_THEN_LOUD` yields `drop=True` on the first step only.) The offset range is clamped to 10..100.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_render.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `moods.py` and `Renderer`**

Target hue drifts around the anchors over `loop_seconds`, up to 30% faster at full energy; a drop advances the target to the next anchor. Target brightness = floor + energy x (ceiling - floor) after offset. Target saturation = anchor saturation scaled by 0.6 + 0.4 x energy, capped at `max_saturation`, and pulled to 25 in a breakdown. Then move the output toward the target, clamped by the slew limits (hue along the shorter arc).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_music_render.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/amaran_mcp/music/moods.py src/amaran_mcp/music/render.py tests/test_music_render.py
git commit -m "Add music moods and slew-limited renderer"
```

---

### Task 4: Capture sources

**Files:**
- Modify: `src/amaran_mcp/music/capture.py`
- Modify: `pyproject.toml` (`uv add sounddevice`)
- Test: `tests/test_music_capture.py`

**Interfaces:**
- Consumes: `helper_path()` (Task 1).
- Produces:

```python
class Source(Protocol):
    rate: int
    name: str                                  # "system audio" or the microphone's name
    async def blocks(self) -> AsyncIterator[np.ndarray]: ...   # mono float32
    async def close(self) -> None: ...

class SystemAudio:   # runs the helper; raises CaptureError if it exits 3 or never prints RATE
    def __init__(self, command: list[str] | None = None) -> None: ...  # command overridable for tests
async def open_source() -> Source:   # SystemAudio, else built-in microphone, else CaptureError
def pick_microphone(devices: list[dict]) -> int | None:  # sounddevice.query_devices() shape
```

`pick_microphone` returns the index of an input device whose name contains "MacBook" or "Built-in" and "Microphone"; it never returns a device whose name matches AirPods, Beats, Bluetooth or "Hands-Free", even if it is the default.

- [ ] **Step 1: Write the failing tests**

```python
def test_system_audio_reads_header_and_samples(tmp_path):
    fake = tmp_path / "fake_helper.py"
    fake.write_text(
        "import sys, struct\n"
        "sys.stdout.buffer.write(b'RATE 48000\\n')\n"
        "sys.stdout.buffer.write(struct.pack('<4f', 0.1, -0.1, 0.2, -0.2) * 512)\n"
        "sys.stdout.buffer.flush()\n")
    src = SystemAudio(command=[sys.executable, "-I", str(fake)])
    blocks = asyncio.run(collect(src, 2))
    assert src.rate == 48000 and blocks[0].dtype == np.float32
    assert np.isclose(blocks[0][:4], [0.1, -0.1, 0.2, -0.2]).all()

def test_system_audio_reports_tap_failure(tmp_path):
    fake = tmp_path / "fails.py"
    fake.write_text("import sys; print('tap failed: -1', file=sys.stderr); sys.exit(3)\n")
    with pytest.raises(CaptureError, match="tap failed"):
        asyncio.run(collect(SystemAudio(command=[sys.executable, "-I", str(fake)]), 1))

def test_microphone_never_picks_airpods():
    devices = [
        {"name": "James's AirPods Pro", "max_input_channels": 1},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    assert pick_microphone(devices) == 1
    assert pick_microphone(devices[:1]) is None
```

(`collect(src, n)` is a small test helper returning the first n blocks and closing the source.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_capture.py -v`
Expected: the three new tests FAIL

- [ ] **Step 3: Implement `SystemAudio`, `MicAudio` (sounddevice `InputStream` on the picked index, blocks via an asyncio queue), `pick_microphone`, `open_source`**

`open_source` tries `SystemAudio(); await first block` and falls back to the microphone on `CaptureError`, logging why.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_music_capture.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/amaran_mcp/music/capture.py tests/test_music_capture.py
git commit -m "Add system audio and microphone sources; never open Bluetooth inputs"
```

---

### Task 5: Apple Music reader

**Files:**
- Create: `src/amaran_mcp/music/applemusic.py`, `tests/test_music_applemusic.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class Track:
    name: str; artist: str; playing: bool

def now_playing(run: Callable[[list[str]], subprocess.CompletedProcess] = _run) -> Track | None
def anchor_for(track: Track, anchors: int) -> int   # stable: sha256 of "name|artist" mod anchors
```

The AppleScript first checks `application "Music" is running` and returns an empty result if not, so Music is never launched. Output is one line `state<TAB>name<TAB>artist`. Timeout 3 s. Any non-zero exit (including Automation refused, -1743) or timeout returns None.

- [ ] **Step 1: Write the failing tests**

```python
def fake(stdout="", code=0):
    return lambda args: subprocess.CompletedProcess(args, code, stdout, "")

def test_playing_track():
    t = now_playing(fake("playing\tAdagio for Strings\tTiesto\n"))
    assert t == Track("Adagio for Strings", "Tiesto", True)

def test_not_running_or_refused_is_none():
    assert now_playing(fake("")) is None
    assert now_playing(fake("", code=1)) is None  # e.g. Automation refused (-1743)

def test_script_never_launches_music():
    seen = []
    now_playing(lambda args: seen.append(args) or subprocess.CompletedProcess(args, 0, "", ""))
    assert 'application "Music" is running' in " ".join(seen[0])

def test_anchor_is_stable():
    t = Track("Adagio for Strings", "Tiesto", True)
    assert anchor_for(t, 4) == anchor_for(t, 4) and 0 <= anchor_for(t, 4) < 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_applemusic.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `now_playing` and `anchor_for`**

When osascript fails with error -1743 (Automation refused), log once to stderr: "To show track names, allow Terminal to control Music in System Settings, Privacy and Security, Automation." Add a test that the message is logged once across two refused calls (use `caplog`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_music_applemusic.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/amaran_mcp/music/applemusic.py tests/test_music_applemusic.py
git commit -m "Add Apple Music now-playing reader"
```

---

### Task 6: The music-mode loop

**Files:**
- Create: `src/amaran_mcp/music/runner.py`, `tests/test_music_runner.py`
- Modify: `src/amaran_mcp/light.py` (add `LightBase.apply`, make `recall_scene` use it)

**Interfaces:**
- Consumes: `Analyzer`, `Features` (Task 2); `MOODS`, `Renderer` (Task 3); `Source`, `open_source` (Task 4); `now_playing`, `anchor_for` (Task 5); `backends.current_light()`; `LightBase.set_color`, `LightBase.read_status`.
- Produces:

```python
# light.py
async def LightBase.apply(self, state: LightState) -> None   # power off, or white/colour/effect as recall_scene did

# runner.py
STATUS_PATH = CONFIG_DIR / "music.json"     # written by the runner, read by control
CONTROL_PATH = CONFIG_DIR / "music_control.json"  # {"brightness_offset": float}, read each second

@dataclass
class RunnerConfig:
    mood: str = "ethereal"
    silence_exit_seconds: float = 15 * 60
    settle_seconds: float = 5.0
    music_poll_seconds: float = 3.0

async def run(config: RunnerConfig, source: Source | None = None,
              light: Callable[[], LightBase] = current_light,
              clock: Callable[[], float] = time.monotonic,
              music: Callable[[], Track | None] = now_playing) -> int   # exit code
```

Status JSON fields: `pid`, `mood`, `state` ("listening" | "playing" | "settling"), `track` (`{"name", "artist"}` or null), `source` (source name), `backend`, `started_at` (ISO time), `message` (last warning or null).

Behaviour: read the light's state once at start (`read_status`, kept as a `LightState`). Each tick (1/8 s when `light().backend == "amaran Desktop"`, else 1/3 s): pull audio into the analyzer, step the renderer, send `set_color(hue, saturation, brightness)`. On `section == "silence"`: fade from the current output to the start state over `settle_seconds`, then `apply(start_state)` and state "listening" (no further light writes until music returns). On `silence_exit_seconds` of continuous silence: exit 0. On SIGTERM: settle the same way, then exit 0. A track change from Apple Music calls `renderer.jump_to_anchor(anchor_for(track, len(mood.anchors)))`. A `DesktopError` or `LinkError` on a tick: log, set `message`, continue. Exact zeros for 10 s while `music().playing` is true: write the permission message to status and stderr, exit 2.

- [ ] **Step 1: Write the failing tests** (fake source yields fixture audio from Task 2; fake light records calls; fake clock advances per tick)

```python
def test_plays_then_settles_back_to_start_state():
    light = FakeLight(start=LightState(True, "white", 80, 5400, 0))
    code = asyncio.run(run(RunnerConfig(silence_exit_seconds=30),
                           source=FakeSource([kick_track(24000, 20), silence(24000, 45)]),
                           light=lambda: light, clock=light.clock, music=lambda: None))
    assert code == 0
    assert any(c[0] == "set_color" for c in light.calls)
    assert light.calls[-1] == ("apply", LightState(True, "white", 80, 5400, 0))
    assert json.loads(STATUS_PATH.read_text())["state"] in ("listening", "settling")

def test_backend_error_on_one_tick_is_not_fatal():
    light = FakeLight(fail_on_calls={3: DesktopError("Desktop quit")})
    code = asyncio.run(run(RunnerConfig(silence_exit_seconds=10),
                           source=FakeSource([kick_track(24000, 10), silence(24000, 25)]),
                           light=lambda: light, clock=light.clock, music=lambda: None))
    assert code == 0 and len(light.calls) > 10

def test_zeros_while_music_app_playing_exits_with_permission_message():
    light = FakeLight()
    code = asyncio.run(run(RunnerConfig(), source=FakeSource([silence(24000, 15)]),
                           light=lambda: light, clock=light.clock,
                           music=lambda: Track("x", "y", True)))
    assert code == 2
    assert "Screen and System Audio Recording" in json.loads(STATUS_PATH.read_text())["message"]

def test_track_change_jumps_palette_anchor(monkeypatch):
    ...  # two tracks via the fake music callable; assert Renderer.jump_to_anchor called with anchor_for(track, 4)

def test_brightness_offset_from_control_file():
    ...  # write CONTROL_PATH {"brightness_offset": 15} before run; assert every set_color brightness >= 40
```

Also add to `tests/test_offline.py`: `test_apply_restores_white_and_power_off` covering `LightBase.apply` through a stub subclass.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_runner.py tests/test_offline.py -v`
Expected: new tests FAIL

- [ ] **Step 3: Implement `LightBase.apply` and `runner.run`**

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/amaran_mcp/light.py src/amaran_mcp/music/runner.py tests/test_music_runner.py tests/test_offline.py
git commit -m "Add the music-mode loop with settling, silence exit and status file"
```

---

### Task 7: Start, stop and MCP tools

**Files:**
- Create: `src/amaran_mcp/music/control.py`, `src/amaran_mcp/__main__.py`, `tests/test_music_control.py`
- Modify: `src/amaran_mcp/__init__.py` (CLI `music [--mood NAME]` runs `runner.run` in the foreground; `music-stop`), `src/amaran_mcp/server.py`

**Interfaces:**
- Consumes: `runner.STATUS_PATH`, `runner.CONTROL_PATH`, `RunnerConfig`, `MOODS`.
- Produces:

```python
PID_PATH = CONFIG_DIR / "music.pid"
LOG_PATH = CONFIG_DIR / "music.log"

def status() -> dict | None          # status JSON if the PID is alive, else None (and stale files removed)
def start(mood: str = "ethereal", spawn: Callable[[list[str]], int] = _spawn) -> dict
def stop(timeout: float = 8.0) -> bool  # SIGTERM, wait for exit (runner settles the light), True if one was running
def set_brightness(percent: float) -> float  # writes CONTROL_PATH offset = percent - mood midpoint, returns offset
```

`_spawn` runs `[sys.executable, "-m", "amaran_mcp", "music", "--mood", mood]` with `start_new_session=True`, stdin from `/dev/null`, stdout and stderr appended to `LOG_PATH`, and returns the PID, which `start` writes to `PID_PATH`. `start` with an unknown mood raises `ValueError` listing the moods. `start` while running calls `stop()` first.

MCP tools in `server.py`:
- `start_music_mode(mood: str = "ethereal") -> dict`: "Start music mode: the light follows what this Mac is playing (including AirPlay from an iPhone) with slow, ethereal colour. Moods: ethereal, deep, dawn."
- `stop_music_mode() -> str`: "Stop music mode and return the light to how it was."
- `get_light` gains `music_mode`: `control.status()` or `{"running": false}`. When music mode is running and the backend is Bluetooth, `get_light` returns the music status without connecting to the light (the runner holds the connection).
- `set_white`, `set_color`, `set_effect`, `stop_effect`, `recall_scene`, `set_power` call `control.stop()` first when music mode is running. `set_brightness` while running calls `control.set_brightness(percent)` instead and returns "music mode brightness range moved to X..Y%".

- [ ] **Step 1: Write the failing tests**

```python
def test_start_writes_pid_and_restarts_with_new_mood(home, monkeypatch):
    spawned = []
    monkeypatch.setattr(control, "_alive", lambda pid: pid in spawned)
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: spawned.clear() or True)
    control.start("deep", spawn=lambda argv: spawned.append(4242) or 4242)
    assert control.PID_PATH.read_text().strip() == "4242"
    control.start("dawn", spawn=lambda argv: spawned.append(4343) or 4343)
    assert spawned == [4343]

def test_stale_pid_reports_not_running(home):
    control.PID_PATH.write_text("999999")
    assert control.status() is None and not control.PID_PATH.exists()

def test_unknown_mood_lists_moods(home):
    with pytest.raises(ValueError, match="ethereal, deep, dawn"):
        control.start("disco", spawn=lambda argv: 1)

def test_set_white_stops_music_first(monkeypatch):
    order = []
    monkeypatch.setattr(control, "status", lambda: {"state": "playing"})
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: order.append("stop") or True)
    ...  # stub current_light with a light whose set_white appends "set_white"; call server.set_white(3200)
    assert order == ["stop", "set_white"]

def test_set_brightness_moves_music_range_instead_of_stopping(monkeypatch):
    ...  # status() running with mood ethereal; server.set_brightness(60); assert CONTROL_PATH offset == 60 - 47.5 and stop not called
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_music_control.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `control.py`, `__main__.py`, CLI commands and the server changes**

- [ ] **Step 4: Run all tests and the MCP tool listing**

Run: `uv run pytest -q`
Expected: all PASS
Run: the stdio MCP client check from earlier work (list tools).
Expected: `start_music_mode` and `stop_music_mode` listed alongside the 9 existing tools.

- [ ] **Step 5: Commit**

```bash
git add src/amaran_mcp/music/control.py src/amaran_mcp/__main__.py src/amaran_mcp/__init__.py src/amaran_mcp/server.py tests/test_music_control.py
git commit -m "Add music mode start/stop, MCP tools and light ownership rules"
```

---

### Task 8: Live tuning with the owner, docs, publish

**Files:**
- Modify: `src/amaran_mcp/music/moods.py`, `src/amaran_mcp/music/analysis.py` (tuning values only), `README.md`

- [ ] **Step 1: Run music mode on real music**

Run: `uv run amaran-mcp music --mood ethereal` in the foreground with a house or trance track playing to AirPods, amaran Desktop running. Watch the tube for at least one full breakdown and drop. Then the same through AirPlay from the iPhone.

- [ ] **Step 2: Tune against the anti-goals with the owner watching**

Ask the owner: did anything flash or pulse on the beat, did breakdowns read as paler, did the drop read as a palette shift, was the desk always comfortable to work at. Adjust only time constants, thresholds and mood colours. Re-run `uv run pytest -q` after each change; slew and bounds tests must still pass.

- [ ] **Step 3: Check the Claude path end to end**

From a Claude Code session in Terminal: "start music mode", "brighter", "what's the light doing" (shows track and state), "set it to 3200K" (music stops, white applied), "start music mode deep", "stop music mode" (light returns to its starting state).

- [ ] **Step 4: Document in README**

A "Music mode" section: what it does, macOS 14.2+, the Screen and System Audio Recording permission for Terminal, AirPlay from iPhone to Mac with AirPods as the Mac's output, the three moods, the microphone fallback and that it never opens AirPods as a microphone, and the log at `~/.config/amaran-mcp/music.log`.

- [ ] **Step 5: Commit and push**

```bash
git add README.md src/amaran_mcp/music
git commit -m "Tune music mode with real music and document it"
git push origin main
```
