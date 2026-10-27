"""What gets recorded, where, and for how long."""

import importlib
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from music_fixtures import make_features

RATE = 24000
BLOCK = 1024


@pytest.fixture
def music(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import control, listener, recorder, runner
    for module in (recorder, listener, runner, control):
        importlib.reload(module)
    return SimpleNamespace(listener=listener, recorder=recorder, runner=runner)


def blocks(seconds: float, is_music: bool):
    tone = (0.1 * np.sin(np.arange(BLOCK) / 5)).astype(np.float32)
    features = make_features(is_music=is_music)
    for _ in range(int(seconds * RATE / BLOCK)):
        yield tone, features


def test_recording_ends_two_minutes_after_the_music(music):
    sessions = music.listener.Sessions()
    for block, f in blocks(20, True):
        sessions.feed(block, RATE, f)
    recorder = sessions.session.recorder
    for block, f in blocks(200, False):  # a call, a video, silence: after the music
        sessions.feed(block, RATE, f)
    assert sessions.session is None
    assert 135 <= recorder.seconds <= 145  # the music, then 120 s, not the whole 200 s


def test_nothing_is_recorded_before_music_is_heard(music):
    sessions = music.listener.Sessions()
    for block, f in blocks(60, False):
        sessions.feed(block, RATE, f)
    assert sessions.session is None and not music.recorder.RECORDINGS_DIR.exists()


def test_the_recordings_folder_is_private(music):
    music.recorder.RECORDINGS_DIR.mkdir(parents=True)
    music.recorder.RECORDINGS_DIR.chmod(0o755)  # as older versions left it
    sessions = music.listener.Sessions()
    for block, f in blocks(15, True):
        sessions.feed(block, RATE, f)
    sessions.close()
    for folder in [music.recorder.RECORDINGS_DIR, *music.recorder.RECORDINGS_DIR.iterdir()]:
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_a_full_disk_cap_stops_a_new_session_at_once(music):
    music.recorder.RECORDINGS_DIR.mkdir(parents=True)
    (music.recorder.RECORDINGS_DIR / "old.flac").write_bytes(b"x" * 2048)
    rec = music.recorder.Recorder(music.recorder.RECORDINGS_DIR / "new", cap_bytes=1024)
    assert rec.stopped and "cap" in rec.stopped
    rec.write(np.zeros(BLOCK, np.float32), RATE)
    assert rec.seconds == 0 and not (music.recorder.RECORDINGS_DIR / "new").exists()


def test_a_session_stops_at_the_cap_within_a_minute(music):
    rec = music.recorder.Recorder(music.recorder.RECORDINGS_DIR / "s", cap_bytes=20_000)
    noise = np.random.default_rng(0).standard_normal(BLOCK).astype(np.float32) * 0.3
    for _ in range(int(130 * RATE / BLOCK)):
        rec.write(noise, RATE)
        if rec.stopped:
            break
    assert rec.stopped and rec.seconds <= 125  # caught at a one-minute check, not after 30 minutes


def test_the_listener_waits_while_music_mode_records(music):
    asked = []
    sessions = music.listener.Sessions(may_open=lambda: asked.append(1) or False)
    for block, f in blocks(30, True):
        sessions.feed(block, RATE, f)
    assert sessions.session is None
    assert len(asked) <= 4  # asks every 10 s, not on every block


def test_music_mode_never_records_the_microphone(music):
    from amaran_mcp.music.capture import MicAudio

    class Mic(MicAudio):
        def __init__(self):
            self.name, self.rate = "MacBook Pro Microphone", RATE

    run = music.runner._Run(music.runner.RunnerConfig(record=True), Mic(), lambda: None, lambda: None)
    assert run.sessions is None
