"""The background listener: records listening sessions for training, never touches the light.

Started at login by a launch agent (`amaran-mcp listen`). It captures system
audio like music mode, and while music is playing writes a session (FLAC,
a feature log once a second, and the owner's marks) under
CONFIG_DIR/recordings. A session closes after a stretch of quiet; the
seconds just before music was recognised are kept too. The light only ever
changes in music mode, when the owner asks for it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
import signal
import time

import numpy as np

from ..config import CONFIG_DIR, _write_private, is_ours
from .analysis import Analyzer
from .capture import CaptureError, Source, SystemAudio
from . import recorder as recordings  # its RECORDINGS_DIR read at use, so tests can move it
from .recorder import Recorder

log = logging.getLogger(__name__)

STATUS_PATH = CONFIG_DIR / "listener.json"
PREROLL_SECONDS = 10.0  # music is recognised a few seconds in; keep what came just before
REASK_SECONDS = 10.0  # how often a refused session asks again whether it may open


def status() -> dict | None:
    """The listener's status if it is running."""

    try:
        data = json.loads(STATUS_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return None
    try:
        pid = int(data.get("pid", -1))
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError, OverflowError):
        return None
    except PermissionError:
        pass
    return data if is_ours(pid, "listen") else None


class _Session:
    def __init__(self, marks_seen: int | None = None) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = recordings.RECORDINGS_DIR / stamp
        for n in range(2, 100):  # a session that closes and reopens within a second
            if not self.directory.exists():
                break
            self.directory = recordings.RECORDINGS_DIR / f"{stamp}-{n}"
        self.recorder = Recorder(self.directory)
        self.recorder._make_directory()
        self.features = open(self.directory / "features.jsonl", "a")
        self.marks_seen = _marks_count() if marks_seen is None else marks_seen
        self._second = 0.0

    def write(self, block: np.ndarray, rate: int, features) -> None:
        self.recorder.write(block, rate)
        if self.recorder.seconds >= self._second + 1.0:
            self._second = self.recorder.seconds
            row = {"t": round(self.recorder.seconds, 2), **asdict(features)}
            self.features.write(json.dumps(row, default=str) + "\n")

    def take_marks(self) -> None:
        from .runner import MARKS_PATH

        if not MARKS_PATH.exists():
            return
        lines = MARKS_PATH.read_text().splitlines()
        for line in lines[self.marks_seen :]:
            try:
                entry = json.loads(line)
                ago = max(0.0, time.time() - float(entry["wall"]))
                self.recorder.mark(entry["kind"], entry.get("note"), seconds_ago=ago)
            except (ValueError, KeyError, RuntimeError) as err:
                log.warning("could not place a mark: %s", err)
        self.marks_seen = len(lines)

    def close(self) -> None:
        self.take_marks()
        self.recorder.close()
        self.features.close()


def _marks_count() -> int:
    from .runner import MARKS_PATH

    return len(MARKS_PATH.read_text().splitlines()) if MARKS_PATH.exists() else 0


class Sessions:
    """Record while music plays: a session opens when music is recognised (keeping the
    seconds just before it) and closes after `quiet_close_seconds` without music, so
    calls, videos and silence around the music are not kept. The listener and music
    mode both record through this, under one rule."""

    def __init__(self, quiet_close_seconds: float = 120.0, may_open=lambda: True,
                 first_marks: int | None = None) -> None:
        self.quiet_close_seconds = quiet_close_seconds
        self.session: _Session | None = None
        self._may_open = may_open
        self._first_marks = first_marks  # music mode: marks since it started belong to its first session
        self._quiet = 0.0
        self._ask_in = 0.0  # seconds until may_open is asked again after a no
        self._preroll: deque = deque()

    @property
    def recording(self) -> str | None:
        session = self.session
        return str(session.directory) if session and not session.recorder.stopped else None

    @property
    def stopped(self) -> str | None:
        return self.session.recorder.stopped if self.session else None

    def feed(self, block: np.ndarray, rate: int, features) -> bool:
        """Take one block; True when a session opened or closed."""

        dt = len(block) / rate
        changed = False
        self._ask_in -= dt
        if features.is_music:
            self._quiet = 0.0
            if self.session is None and self._ask_in <= 0 and not self._may_open():
                self._ask_in = REASK_SECONDS
            elif self.session is None and self._ask_in <= 0:
                self.session = _Session(self._first_marks)
                self._first_marks = None
                for old, old_features in self._preroll:
                    self.session.write(old, rate, old_features)
                self._preroll.clear()
                changed = True
        elif self.session is not None:
            self._quiet += dt
            if self._quiet >= self.quiet_close_seconds:
                self.close()
                changed = True
        if self.session is not None:
            self.session.write(block, rate, features)
        else:
            self._preroll.append((block, features))
            while len(self._preroll) * dt > PREROLL_SECONDS:
                self._preroll.popleft()
        return changed

    def take_marks(self) -> None:
        if self.session is not None:
            self.session.take_marks()

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


def _music_mode_records() -> bool:
    """Music mode recording on its own (the listener was down when it started)."""

    from . import control

    return (control.status() or {}).get("recorded_by") == "music mode"


async def listen(source: Source | None = None, quiet_close_seconds: float = 120.0) -> int:
    try:
        source = source or SystemAudio()
        blocks = source.blocks()
    except CaptureError as err:
        log.error("cannot listen: %s", err)
        return 1
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass
    analyzer: Analyzer | None = None
    sessions = Sessions(quiet_close_seconds, may_open=lambda: not _music_mode_records())
    now = 0.0
    next_status = 0.0

    def write_status() -> None:
        _write_private(STATUS_PATH, {
            "pid": os.getpid(),
            "recording": sessions.recording,
            "recording_stopped": sessions.stopped,
            "state": "music" if sessions.session else "quiet",
        })

    try:
        async for block in blocks:
            if stop.is_set():
                break
            rate = source.rate
            if analyzer is None:
                analyzer = Analyzer(rate)
            now += len(block) / rate
            classes = source.latest_classes() if hasattr(source, "latest_classes") else None
            f = analyzer.feed(block, classes)
            if sessions.feed(block, rate, f):
                write_status()
            if now >= next_status:
                next_status = now + 5.0
                sessions.take_marks()
                write_status()
    finally:
        sessions.close()
        await source.close()
        write_status()
    return 0
