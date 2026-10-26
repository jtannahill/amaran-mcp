"""Recording clips and replaying them through the analysis."""

import asyncio
import importlib

import pytest

from music_fixtures import FakeSource, kick_track, write_wav


@pytest.fixture
def clips(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import clips as module
    importlib.reload(module)
    return module


def test_record_writes_wav_in_clips_dir(clips):
    path = asyncio.run(clips.record(1.0, "test", source=FakeSource([kick_track(24000, 2)])))
    assert path.parent == clips.CLIPS_DIR and path.suffix == ".wav"
    samples, rate = clips.load(path)
    assert rate == 24000 and abs(len(samples) - 24000) <= 1024


def test_replay_reports_sections_per_second(clips, tmp_path):
    path = write_wav(tmp_path, kick_track(24000, 12), 24000)
    rows = clips.replay(path)
    assert len(rows) in (11, 12) and rows[-1]["section"] in ("intro", "groove", "playing")
    assert {"t", "section", "energy"} <= rows[0].keys()


def test_replay_rows_keep_one_block_events(clips, tmp_path):
    from music_fixtures import arrangement
    samples, _ = arrangement(24000)
    path = write_wav(tmp_path, samples, 24000)
    rows = clips.replay(path)
    assert any(r["bloom"] for r in rows) and sum(r["release"] for r in rows) == 1
