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

from ..config import CONFIG_DIR, _write_private
from .analysis import Analyzer
from .capture import CaptureError, Source, SystemAudio
from .recorder import RECORDINGS_DIR, Recorder

log = logging.getLogger(__name__)

STATUS_PATH = CONFIG_DIR / "listener.json"
PREROLL_SECONDS = 10.0  # music is recognised a few seconds in; keep what came just before


def status() -> dict | None:
    """The listener's status if it is running."""

    try:
        data = json.loads(STATUS_PATH.read_text())
    except (FileNotFoundError, ValueError):
        return None
    try:
        os.kill(int(data.get("pid", -1)), 0)
    except (ProcessLookupError, ValueError, OverflowError):
        return None
    except PermissionError:
        pass
    return data


class _Session:
    def __init__(self) -> None:
        self.directory = RECORDINGS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.recorder = Recorder(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.features = open(self.directory / "features.jsonl", "a")
        self.marks_seen = _marks_count()
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
    preroll: deque = deque()
    session: _Session | None = None
    quiet = 0.0
    now = 0.0
    next_status = 0.0

    def write_status() -> None:
        _write_private(STATUS_PATH, {
            "pid": os.getpid(),
            "recording": str(session.directory) if session and not session.recorder.stopped else None,
            "recording_stopped": session.recorder.stopped if session else None,
            "state": "music" if session else "quiet",
        })

    try:
        async for block in blocks:
            if stop.is_set():
                break
            rate = source.rate
            if analyzer is None:
                analyzer = Analyzer(rate)
            dt = len(block) / rate
            now += dt
            classes = source.latest_classes() if hasattr(source, "latest_classes") else None
            f = analyzer.feed(block, classes)
            if f.is_music:
                quiet = 0.0
                if session is None:
                    session = _Session()
                    for old, old_features in preroll:
                        session.write(old, rate, old_features)
                    preroll.clear()
                    write_status()
            elif session is not None:
                quiet += dt
                if quiet >= quiet_close_seconds:
                    session.close()
                    session = None
                    write_status()
            if session is not None:
                session.write(block, rate, f)
            else:
                preroll.append((block, f))
                while len(preroll) * dt > PREROLL_SECONDS:
                    preroll.popleft()
            if now >= next_status:
                next_status = now + 5.0
                if session:
                    session.take_marks()
                write_status()
    finally:
        if session is not None:
            session.close()
        await source.close()
        session = None
        write_status()
    return 0
