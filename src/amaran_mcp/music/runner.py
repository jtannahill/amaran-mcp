"""The music-mode loop: listen, analyse, and keep the light moving slowly.

Audio is read and analysed on every block; time is counted in audio, so the
loop runs at the music's pace. Light commands go to a writer task that only
ever sends the newest one, so a slow light (a Bluetooth reconnect) never
holds up reading the audio.

States: "listening" (no music; the light is left as it was), "playing", and
"settling" (fading back to how the light was before music mode started).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
import os
import signal
import sys
from typing import Any

from ..backends import current_light
from ..config import CONFIG_DIR, _write_private
from ..desktop import DesktopError
from ..light import LightBase, LightState
from ..link import LinkError
from .analysis import Analyzer, Features, merge_events
from .applemusic import Track, anchor_for, now_playing
from .capture import CaptureError, Source, open_source
from .moods import MOODS, Mood
from .render import Output, Renderer

log = logging.getLogger(__name__)

STATUS_PATH = CONFIG_DIR / "music.json"
CONTROL_PATH = CONFIG_DIR / "music_control.json"

DESKTOP_HZ = 8.0
BLUETOOTH_HZ = 3.0
FADE_IN_SECONDS = 2.0
ZERO_SECONDS = 10.0
STATUS_EVERY = 5.0
CONTROL_EVERY = 1.0
PERMISSION_MESSAGE = (
    "Music mode hears only silence while the Music app is playing. Allow Terminal in "
    "System Settings, Privacy and Security, Screen and System Audio Recording, then "
    "start music mode again."
)


@dataclass
class RunnerConfig:
    mood: str = "ethereal"
    silence_exit_seconds: float = 15 * 60
    settle_seconds: float = 5.0
    music_poll_seconds: float = 3.0


class _Writer:
    """Sends light commands one at a time, newest first, dropping stale ones."""

    def __init__(self, light: Callable[[], LightBase], on_error: Callable[[str], None]) -> None:
        self._light = light
        self._on_error = on_error
        self._pending: tuple | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task = asyncio.get_running_loop().create_task(self._loop())

    def post(self, command: tuple) -> None:
        self._pending = command
        self._idle.clear()
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            command, self._pending = self._pending, None
            if command is not None:
                await self._send(command)
            if self._pending is None:
                self._idle.set()

    async def _send(self, command: tuple) -> None:
        kind, *args = command
        try:
            lamp = self._light()
            if kind == "color":
                await lamp.set_color(*args)
            elif kind == "brightness":
                await lamp.set_brightness(*args)
            elif kind == "apply":
                await lamp.apply(*args)
        except (DesktopError, LinkError, ValueError, OSError) as err:
            log.warning("light command failed: %s", err)
            self._on_error(f"light command failed: {err}")

    async def drain(self) -> None:
        await self._idle.wait()

    async def close(self) -> None:
        await self.drain()
        self._task.cancel()


def _read_offset() -> float:
    try:
        return float(json.loads(CONTROL_PATH.read_text()).get("brightness_offset", 0.0))
    except (FileNotFoundError, ValueError, AttributeError):
        return 0.0


class _Run:
    def __init__(self, config: RunnerConfig, source: Source, light: Callable[[], LightBase],
                 music: Callable[[], Track | None]) -> None:
        if config.mood not in MOODS:
            raise ValueError(f"Unknown mood {config.mood!r}. Moods: {', '.join(MOODS)}")
        self.config = config
        self.mood: Mood = MOODS[config.mood]
        self.source = source
        self.light = light
        self.music = music
        self.analyzer = Analyzer(source.rate)
        self.renderer = Renderer(self.mood, brightness_offset=_read_offset())
        self.state = "listening"
        self.track: Track | None = None
        self.message: str | None = None
        self.start_state = LightState()
        self.now = 0.0  # audio seconds
        self.stop = asyncio.Event()
        self._fade: list[tuple] = []  # commands still to send for a fade, one per tick
        self._after_fade: str | None = None
        self._listening_since = 0.0
        self._zero_since: float | None = None
        self._next = {"tick": 0.0, "music": 0.0, "control": 0.0, "status": 0.0}
        self._pending: Features | None = None  # features since the last tick, events merged
        self.writer: _Writer | None = None

    # ── Status ───────────────────────────────────────────────────────────────

    def write_status(self) -> None:
        lamp_backend = getattr(self.light(), "backend", None)
        _write_private(STATUS_PATH, {
            "pid": os.getpid(),
            "mood": self.mood.name,
            "state": self.state,
            "track": {"name": self.track.name, "artist": self.track.artist} if self.track else None,
            "source": self.source.name,
            "backend": lamp_backend,
            "started_at": self.started_at,
            "message": self.message,
        })

    def note_error(self, message: str) -> None:
        self.message = message

    # ── Fades ────────────────────────────────────────────────────────────────

    def _tick_seconds(self) -> float:
        backend = getattr(self.light(), "backend", "")
        return 1 / (DESKTOP_HZ if backend == "amaran Desktop" else BLUETOOTH_HZ)

    def _plan_fade_in(self) -> None:
        """Dim the current look to the mood's floor, then start desaturated."""

        floor = self.renderer._range()[0]
        start = self.start_state
        steps = max(1, round(FADE_IN_SECONDS / self._tick_seconds()))
        hue = self.mood.anchors[0][0]
        if start.power is not False and start.mode == "color" and start.hue is not None:
            self.renderer.begin(Output(start.hue, start.saturation or 0, start.brightness or floor))
            self._fade = []
        else:
            if start.power is not False and start.brightness is not None and start.brightness > floor:
                self._fade = [("brightness", start.brightness + (floor - start.brightness) * (i + 1) / steps)
                              for i in range(steps)]
            else:
                self._fade = []
            self.renderer.begin(Output(hue, 0.0, floor))
        self._after_fade = "playing"
        self.state = "playing"

    def _plan_settle(self, current: Output) -> None:
        """Fade from the music's colour back toward the start state, then restore it."""

        start = self.start_state
        steps = max(1, round(self.config.settle_seconds / self._tick_seconds()))
        if start.power is not False and start.mode == "color" and start.hue is not None:
            end = Output(start.hue, start.saturation or 0, start.brightness or current.brightness)
        else:
            # Towards white at the start brightness (or the floor when it was off).
            level = start.brightness if start.power is not False and start.brightness else current.brightness
            end = Output(current.hue, 0.0, level)
        turn = (end.hue - current.hue + 180) % 360 - 180
        self._fade = [
            ("color", (current.hue + turn * f) % 360,
             current.saturation + (end.saturation - current.saturation) * f,
             current.brightness + (end.brightness - current.brightness) * f)
            for f in ((i + 1) / steps for i in range(steps))
        ]
        if start.mode is not None or start.power is False:
            self._fade.append(("apply", replace(start)))
        self._after_fade = "listening"
        self.state = "settling"

    # ── Loop ─────────────────────────────────────────────────────────────────

    async def poll_music(self) -> None:
        try:
            track = await asyncio.to_thread(self.music)
        except Exception as err:  # the Music app is a bonus; never fatal
            log.warning("could not read the Music app: %s", err)
            return
        if track and (self.track is None or (track.name, track.artist) != (self.track.name, self.track.artist)):
            self.renderer.jump_to_anchor(anchor_for(track, len(self.mood.anchors)))
            self.track = track
            self.write_status()
        elif track is None and self.track is not None:
            self.track = None
            self.write_status()
        elif track:
            self.track = track

    def tick(self, f: Features, dt: float) -> None:
        if self._fade:
            self.writer.post(self._fade.pop(0))
            if not self._fade and self._after_fade:
                self.state, self._after_fade = self._after_fade, None
                if self.state == "listening":
                    self._listening_since = self.now
                self.write_status()
            return
        if self.state == "playing":
            if f.section == "silence" or self.stop.is_set():
                self._plan_settle(self.renderer._out)
                self.write_status()
            elif f.is_music:
                out = self.renderer.step(f, dt)
                self.writer.post(("color", out.hue, out.saturation, out.brightness))
        elif self.state == "listening" and f.is_music and f.section != "silence" and not self.stop.is_set():
            self._plan_fade_in()
            self.write_status()

    async def main(self) -> int:
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.writer = _Writer(self.light, self.note_error)
        try:
            lamp = self.light()
            await lamp.read_status()
            self.start_state = replace(lamp.state)
        except (DesktopError, LinkError, ValueError, OSError) as err:
            self.note_error(f"could not read the light's starting state: {err}")
        self.write_status()

        try:
            async for block in self.source.blocks():
                f = self.analyzer.feed(block)
                self._pending = f if self._pending is None else merge_events(self._pending, f)
                self.now += len(block) / self.source.rate

                if self.now >= self._next["music"]:
                    self._next["music"] = self.now + self.config.music_poll_seconds
                    await self.poll_music()
                if self.now >= self._next["control"]:
                    self._next["control"] = self.now + CONTROL_EVERY
                    self.renderer.brightness_offset = _read_offset()

                # Exact zeros while Music says it is playing: capture is refused.
                if f.digital_zero:
                    self._zero_since = self.now if self._zero_since is None else self._zero_since
                    if self.now - self._zero_since >= ZERO_SECONDS and self.track and self.track.playing:
                        self.message = PERMISSION_MESSAGE
                        print(PERMISSION_MESSAGE, file=sys.stderr)
                        self.write_status()
                        return 2
                else:
                    self._zero_since = None

                if self.now >= self._next["tick"]:
                    dt = self._tick_seconds()
                    self._next["tick"] = self.now + dt
                    self.tick(self._pending, dt)
                    self._pending = None

                if self.state == "listening" and not self._fade:
                    if self.stop.is_set():
                        break
                    if self.now - self._listening_since >= self.config.silence_exit_seconds:
                        break
                if self.now >= self._next["status"]:
                    self._next["status"] = self.now + STATUS_EVERY
                    self.write_status()
        finally:
            # Source ended mid-song (or an error): finish the fade back.
            if self.state == "playing" and not self._fade:
                self._plan_settle(self.renderer._out)
            while self._fade:
                self.writer.post(self._fade.pop(0))
                await self.writer.drain()
            await self.writer.close()
            if self.state == "settling":
                self.state = "listening"
            self.write_status()
            await self.source.close()
        return 0


async def run(config: RunnerConfig, source: Source | None = None,
              light: Callable[[], LightBase] = current_light,
              music: Callable[[], Track | None] = now_playing) -> int:
    """Run music mode until silence times out or SIGTERM. Returns the exit code."""

    try:
        source = source or await open_source()
    except CaptureError as err:
        print(f"music mode cannot listen: {err}", file=sys.stderr)
        return 1
    loop_run = _Run(config, source, light, music)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, loop_run.stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        return await loop_run.main()
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass


def status_of(path: Any = None) -> dict | None:
    """The last status the runner wrote, if any."""

    try:
        return json.loads((path or STATUS_PATH).read_text())
    except (FileNotFoundError, ValueError):
        return None
