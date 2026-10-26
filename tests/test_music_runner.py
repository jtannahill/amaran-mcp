"""The music-mode loop, with fake audio, a fake light and a fake Music app."""

import asyncio
import importlib
import json

import numpy as np
import pytest

from amaran_mcp.desktop import DesktopError
from amaran_mcp.light import LightState
from music_fixtures import FakeSource, kick_track, silence

RATE = 24000


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path))
    from amaran_mcp import config
    importlib.reload(config)
    from amaran_mcp.music import runner as module
    importlib.reload(module)
    return module


class FakeLight:
    backend = "amaran Desktop"

    def __init__(self, start=LightState(True, "white", 60, 4000, 0), fail_on_calls=None):
        self.state = start
        self.start = start
        self.calls = []
        self.fail_on_calls = fail_on_calls or {}

    def _record(self, call):
        self.calls.append(call)
        error = self.fail_on_calls.get(len(self.calls))
        if error:
            raise error

    async def read_status(self):
        return {}

    async def set_color(self, hue, saturation, brightness=None):
        self._record(("set_color", hue, saturation, brightness))

    async def set_brightness(self, percent):
        self._record(("set_brightness", percent))

    async def set_power(self, on):
        self._record(("set_power", on))

    async def apply(self, state):
        self._record(("apply", state))


def status(runner):
    return json.loads(runner.STATUS_PATH.read_text())


def test_plays_then_settles_back_to_start_state(runner):
    light = FakeLight()
    code = asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=30),
                                  source=FakeSource([kick_track(RATE, 20), silence(RATE, 45)]),
                                  light=lambda: light, music=lambda: None))
    assert code == 0
    assert any(c[0] == "set_color" for c in light.calls)
    assert light.calls[-1] == ("apply", light.start)
    assert status(runner)["state"] in ("listening", "settling")
    assert {"section", "chord"} <= status(runner).keys()


def test_no_hard_cut_into_music(runner):
    """The first colour arrives desaturated at the floor, after dimming the white."""
    light = FakeLight(start=LightState(True, "white", 90, 4000, 0))
    asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=10),
                           source=FakeSource([kick_track(RATE, 15), silence(RATE, 25)]),
                           light=lambda: light, music=lambda: None))
    first_color = next(i for i, c in enumerate(light.calls) if c[0] == "set_color")
    dims = [c[1] for c in light.calls[:first_color] if c[0] == "set_brightness"]
    assert dims and dims == sorted(dims, reverse=True) and dims[-1] <= 26
    _, _, saturation, brightness = light.calls[first_color]
    assert saturation <= 5 and brightness <= 25 + 12 / 8  # floor plus one slew step


def test_backend_error_on_one_tick_is_not_fatal(runner):
    light = FakeLight(fail_on_calls={3: DesktopError("Desktop quit")})
    code = asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=10),
                                  source=FakeSource([kick_track(RATE, 10), silence(RATE, 25)]),
                                  light=lambda: light, music=lambda: None))
    assert code == 0 and len(light.calls) > 10


def test_zeros_while_music_app_playing_exits_with_permission_message(runner):
    from amaran_mcp.music.applemusic import Track
    light = FakeLight()
    code = asyncio.run(runner.run(runner.RunnerConfig(), source=FakeSource([silence(RATE, 15)]),
                                  light=lambda: light, music=lambda: Track("x", "y", True)))
    assert code == 2
    assert "Screen and System Audio Recording" in status(runner)["message"]


def test_track_change_jumps_palette_anchor(runner, monkeypatch):
    from amaran_mcp.music import render
    from amaran_mcp.music.applemusic import Track, anchor_for
    jumps = []
    original = render.Renderer.jump_to_anchor
    monkeypatch.setattr(render.Renderer, "jump_to_anchor",
                        lambda self, i: jumps.append(i) or original(self, i))
    tracks = iter([Track("One", "A", True)] * 3 + [Track("Two", "B", True)] * 100)
    light = FakeLight()
    asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=5, music_poll_seconds=3),
                           source=FakeSource([kick_track(RATE, 20), silence(RATE, 20)]),
                           light=lambda: light, music=lambda: next(tracks)))
    assert anchor_for(Track("One", "A", True), 4) in jumps
    assert anchor_for(Track("Two", "B", True), 4) in jumps
    assert status(runner)["track"] == {"name": "Two", "artist": "B"}


def test_brightness_offset_from_control_file(runner):
    from amaran_mcp.config import _write_private
    _write_private(runner.CONTROL_PATH, {"brightness_offset": 15})
    light = FakeLight()
    asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=10),
                           source=FakeSource([kick_track(RATE, 20), silence(RATE, 25)]),
                           light=lambda: light, music=lambda: None))
    colors = [c for c in light.calls if c[0] == "set_color"]
    assert colors and all(c[3] >= 40 - 1e-9 for c in colors)


def test_feature_log_writes_one_line_per_tick(runner, tmp_path):
    log = tmp_path / "features.jsonl"
    asyncio.run(runner.run(runner.RunnerConfig(silence_exit_seconds=5, log_features=log),
                           source=FakeSource([kick_track(RATE, 10), silence(RATE, 20)]),
                           light=lambda: FakeLight(), music=lambda: None))
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) > 50 and {"t", "section", "output"} <= rows[0].keys()
    assert any(row["output"] for row in rows)
