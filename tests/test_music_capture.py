"""System audio helper build, and the capture sources."""

import asyncio
import importlib
import os
import shutil
import sys

import numpy as np

import pytest


@pytest.mark.skipif(shutil.which("swiftc") is None, reason="needs Xcode")
def test_helper_builds_and_reports_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import capture
    importlib.reload(capture)
    path = capture.helper_path()
    assert path.exists() and os.access(path, os.X_OK)
    built = path.stat().st_mtime_ns
    assert capture.helper_path() == path  # second call does not rebuild
    assert path.stat().st_mtime_ns == built
    assert (tmp_path / "bin" / "amaran-systemtap.sha256").read_text().strip()


async def collect(src, n):
    blocks = []
    try:
        async for block in src.blocks():
            blocks.append(block)
            if len(blocks) >= n:
                break
    finally:
        await src.close()
    return blocks


def test_system_audio_reads_header_and_samples(tmp_path):
    from amaran_mcp.music.capture import SystemAudio
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
    from amaran_mcp.music.capture import CaptureError, SystemAudio
    fake = tmp_path / "fails.py"
    fake.write_text("import sys; print('tap failed: -1', file=sys.stderr); sys.exit(3)\n")
    with pytest.raises(CaptureError, match="tap failed"):
        asyncio.run(collect(SystemAudio(command=[sys.executable, "-I", str(fake)]), 1))


def test_microphone_never_picks_airpods():
    from amaran_mcp.music.capture import pick_microphone
    devices = [
        {"name": "James's AirPods Pro", "max_input_channels": 1},
        {"name": "MacBook Pro Microphone", "max_input_channels": 1},
    ]
    assert pick_microphone(devices) == 1
    assert pick_microphone(devices[:1]) is None
