"""Starting, stopping and talking to music mode, and the MCP tools around it."""

import asyncio
import importlib
import json

import pytest


@pytest.fixture
def control(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import control as module, listener, recorder, runner
    for stale in (recorder, listener, runner, module):  # every path under the new home
        importlib.reload(stale)
    return module


def test_start_writes_pid_and_restarts_with_new_mood(control, monkeypatch):
    spawned = []
    monkeypatch.setattr(control, "_alive", lambda pid: pid in spawned)
    monkeypatch.setattr(control, "_ours", lambda pid: True)
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: spawned.clear() or True)
    control.start("deep", spawn=lambda argv: spawned.append(4242) or 4242)
    assert control.PID_PATH.read_text().strip() == "4242"
    control.start("dawn", spawn=lambda argv: spawned.append(4343) or 4343)
    assert spawned == [4343]
    assert control.PID_PATH.read_text().strip() == "4343"


def test_spawn_command_runs_the_package(control):
    seen = []
    control.start("deep", spawn=lambda argv: seen.append(argv) or 1)
    assert seen[0][1:] == ["-m", "amaran_mcp", "music", "--mood", "deep"]


def test_stale_pid_reports_not_running(control):
    control.PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    control.PID_PATH.write_text("999999")
    assert control.status() is None and not control.PID_PATH.exists()


def test_unknown_mood_lists_moods(control):
    with pytest.raises(ValueError, match="ethereal, deep, dawn"):
        control.start("disco", spawn=lambda argv: 1)


def test_start_publishes_the_new_mood_before_the_child_scans(control, monkeypatch):
    from amaran_mcp.music.moods import MOODS
    monkeypatch.setattr(control, "_alive", lambda pid: True)
    monkeypatch.setattr(control, "_ours", lambda pid: True)
    control.start("dawn", spawn=lambda argv: 7)
    music = control.status()
    assert music["mood"] == "dawn" and music["records"] is True and music["recording"] is None
    assert music["recorded_by"] == "music mode"
    low, high = MOODS["dawn"].brightness
    assert control.set_brightness(50) == pytest.approx(50 - (low + high) / 2)


def test_set_brightness_writes_offset_from_mood_midpoint(control, monkeypatch):
    from amaran_mcp.music import runner
    monkeypatch.setattr(control, "status", lambda: {"running": True, "mood": "ethereal"})
    assert control.set_brightness(60) == pytest.approx(60 - 47.5)
    assert json.loads(runner.CONTROL_PATH.read_text())["brightness_offset"] == pytest.approx(12.5)


class StubLight:
    backend = "amaran Desktop"

    def __init__(self, order):
        self.order = order
        from amaran_mcp.light import LightState
        self.state = LightState()

    async def set_white(self, kelvin, brightness=None, tint=0):
        self.order.append("set_white")

    async def set_brightness(self, percent):
        self.order.append("set_brightness")


def test_save_scene_stops_music_before_it_reads_the_light(control, monkeypatch):
    from amaran_mcp import server
    order = []

    class Lamp(StubLight):
        async def read_status(self):
            self.order.append("read")
            return {}

        def save_scene(self, name):
            self.order.append(("save", name))
            return {"name": name}

    monkeypatch.setattr(control, "status", lambda: {"running": True, "state": "playing", "mood": "ethereal"})
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: order.append("stop") or True)
    monkeypatch.setattr(server, "light", lambda: Lamp(order))
    asyncio.run(server.save_scene("interview"))
    assert order == ["stop", "read", ("save", "interview")]


def test_set_white_stops_music_first(control, monkeypatch):
    from amaran_mcp import server
    order = []
    monkeypatch.setattr(control, "status", lambda: {"running": True, "state": "playing", "mood": "ethereal"})
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: order.append("stop") or True)
    monkeypatch.setattr(server, "light", lambda: StubLight(order))
    asyncio.run(server.set_white(3200))
    assert order == ["stop", "set_white"]


def test_set_brightness_moves_music_range_instead_of_stopping(control, monkeypatch):
    from amaran_mcp import server
    from amaran_mcp.music import runner
    order = []
    monkeypatch.setattr(control, "status", lambda: {"running": True, "state": "playing", "mood": "ethereal"})
    monkeypatch.setattr(control, "stop", lambda timeout=8.0: order.append("stop") or True)
    monkeypatch.setattr(server, "light", lambda: StubLight(order))
    result = asyncio.run(server.set_brightness(60))
    assert order == []
    assert json.loads(runner.CONTROL_PATH.read_text())["brightness_offset"] == pytest.approx(12.5)
    assert "37.5" in result and "82.5" in result
