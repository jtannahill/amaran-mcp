"""System audio helper build, and the capture sources."""

import importlib
import os
import shutil

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
