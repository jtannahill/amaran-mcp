"""Apple Music reader, with osascript faked."""

import logging
import subprocess

from amaran_mcp.music import applemusic
from amaran_mcp.music.applemusic import Track, anchor_for, now_playing


def fake(stdout="", code=0, stderr=""):
    return lambda args: subprocess.CompletedProcess(args, code, stdout, stderr)


def test_playing_track():
    t = now_playing(fake("playing\tAdagio for Strings\tTiesto\n"))
    assert t == Track("Adagio for Strings", "Tiesto", True)


def test_paused_track_is_not_playing():
    assert now_playing(fake("paused\tAdagio for Strings\tTiesto\n")).playing is False


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


def test_automation_refusal_logged_once(caplog, monkeypatch):
    monkeypatch.setattr(applemusic, "_warned", False)
    refused = fake("", code=1, stderr="execution error: Not authorized to send Apple events to Music. (-1743)")
    with caplog.at_level(logging.WARNING):
        now_playing(refused)
        now_playing(refused)
    messages = [r.getMessage() for r in caplog.records if "Automation" in r.getMessage()]
    assert len(messages) == 1
    assert "System Settings, Privacy and Security, Automation" in messages[0]
