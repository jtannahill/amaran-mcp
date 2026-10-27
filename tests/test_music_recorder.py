"""Recording listening sessions to FLAC, and marking moments in them."""

import asyncio
import importlib
import json
import subprocess

import numpy as np
import pytest

from music_fixtures import kick_track

RATE = 24000


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import recorder as module
    importlib.reload(module)
    return module


def feed(rec, samples, block=1024):
    for i in range(0, len(samples) - block + 1, block):
        rec.write(samples[i : i + block], RATE)


def decoded_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout
    return float(out)


def test_segments_cover_the_whole_session(recorder, tmp_path):
    rec = recorder.Recorder(tmp_path / "session", segment_seconds=30)
    feed(rec, kick_track(RATE, 70))
    rec.close()
    segments = sorted((tmp_path / "session").glob("audio-*.flac"))
    assert len(segments) == 3
    assert abs(sum(decoded_seconds(p) for p in segments) - 70) < 0.2


def test_size_cap_stops_recording_without_failing(recorder, tmp_path):
    rec = recorder.Recorder(tmp_path / "session", cap_bytes=50_000, segment_seconds=5)
    feed(rec, kick_track(RATE, 30))
    rec.close()
    assert rec.stopped and "cap" in rec.stopped


def test_mark_before_any_audio_raises(recorder, tmp_path):
    rec = recorder.Recorder(tmp_path / "session")
    with pytest.raises(RuntimeError):
        rec.mark("breakdown")


def test_marks_are_saved_with_session_time(recorder, tmp_path):
    rec = recorder.Recorder(tmp_path / "session")
    feed(rec, kick_track(RATE, 10))
    rec.mark("lift", "chords open up", seconds_ago=2.0)
    rec.close()
    marks = json.loads((tmp_path / "session" / "marks.json").read_text())
    assert marks[0]["kind"] == "lift" and abs(marks[0]["t"] - 8.0) < 0.1


def test_mark_tool_says_when_not_recording(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import control, listener, runner
    importlib.reload(runner)
    importlib.reload(listener)
    importlib.reload(control)
    from amaran_mcp import server
    assert "nothing is recording" in asyncio.run(server.mark_music_moment("lift"))
