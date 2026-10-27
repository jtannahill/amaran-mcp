"""The background listener: records sessions while music plays, never drives the light."""

import asyncio
import importlib
import json

import pytest

from music_fixtures import FakeSource, kick_track, silence

RATE = 24000


@pytest.fixture
def listener(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import recorder
    importlib.reload(recorder)
    from amaran_mcp.music import runner
    importlib.reload(runner)
    from amaran_mcp.music import listener as module
    importlib.reload(module)
    return module


def test_records_one_session_per_stretch_of_music(listener):
    from amaran_mcp.music.recorder import RECORDINGS_DIR
    source = FakeSource([silence(RATE, 20), kick_track(RATE, 30), silence(RATE, 150), kick_track(RATE, 30), silence(RATE, 10)])
    asyncio.run(listener.listen(source=source, quiet_close_seconds=120))
    sessions = sorted(RECORDINGS_DIR.iterdir())
    assert len(sessions) == 2
    for session in sessions:
        assert list(session.glob("audio-*.flac")) and (session / "features.jsonl").stat().st_size > 0


def test_silence_alone_records_nothing(listener):
    from amaran_mcp.music.recorder import RECORDINGS_DIR
    asyncio.run(listener.listen(source=FakeSource([silence(RATE, 60)]), quiet_close_seconds=120))
    assert not RECORDINGS_DIR.exists() or not any(RECORDINGS_DIR.iterdir())


def test_status_says_when_a_session_is_recording(listener):
    class Peek(FakeSource):
        async def blocks(self):
            async for block in super().blocks():
                yield block
            self.seen = json.loads(listener.STATUS_PATH.read_text())
    source = Peek([kick_track(RATE, 20)])
    asyncio.run(listener.listen(source=source, quiet_close_seconds=120))
    assert source.seen["recording"]


def test_marks_go_to_the_listener_when_it_records(listener, monkeypatch):
    from amaran_mcp.music import control, runner
    importlib.reload(control)
    monkeypatch.setattr(listener, "status", lambda: {"recording": "/tmp/session", "pid": 1})
    monkeypatch.setattr(control, "status", lambda: None)
    entry = control.mark("lift")
    assert entry["kind"] == "lift" and runner.MARKS_PATH.exists()


def test_music_mode_leaves_recording_to_the_listener(listener, monkeypatch):
    from amaran_mcp.music import control
    importlib.reload(control)
    monkeypatch.setattr(listener, "status", lambda: {"recording": None, "pid": 1})
    seen = []
    result = control.start("ethereal", spawn=lambda argv: seen.append(argv) or 7)
    assert "--no-record" in seen[0] and result["recording"] is False
