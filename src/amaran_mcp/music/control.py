"""Start, stop and steer music mode from another process (the MCP server, the CLI).

Music mode runs as its own detached process so it outlives a Claude session.
It is found through a PID file; it reports through runner.STATUS_PATH and
reads brightness changes from runner.CONTROL_PATH.
"""

from __future__ import annotations

from collections.abc import Callable
import json
import os
import signal
import subprocess
import sys
import time

from ..config import CONFIG_DIR, _write_private
from . import runner
from .moods import MOODS

PID_PATH = CONFIG_DIR / "music.pid"
LOG_PATH = CONFIG_DIR / "music.log"


def _alive(pid: int) -> bool:
    try:
        # Reap it first if it is our own finished child, or it lingers as a zombie.
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid() -> int | None:
    try:
        return int(PID_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def status() -> dict | None:
    """What music mode is doing, or None when it is not running."""

    pid = _pid()
    if pid is None:
        return None
    if not _alive(pid):
        PID_PATH.unlink(missing_ok=True)
        return None
    return {"running": True, **(runner.status_of() or {}), "pid": pid}


def _spawn(argv: list[str]) -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "ab") as log:
        process = subprocess.Popen(argv, start_new_session=True, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
    return process.pid


def start(mood: str = "ethereal", spawn: Callable[[list[str]], int] = _spawn, record: bool = True) -> dict:
    if mood not in MOODS:
        raise ValueError(f"Unknown mood {mood!r}. Moods: {', '.join(MOODS)}")
    if status():
        stop()
    from . import listener

    runner.CONTROL_PATH.unlink(missing_ok=True)  # a new session starts at the mood's own range
    if listener.status():
        record = False  # the background listener records; never twice
    else:
        runner.MARKS_PATH.unlink(missing_ok=True)
    argv = [sys.executable, "-m", "amaran_mcp", "music", "--mood", mood]
    pid = spawn(argv if record else argv + ["--no-record"])
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(f"{pid}\n")
    return {"running": True, "pid": pid, "mood": mood, "recording": record}


def mark(kind: str, note: str | None = None) -> dict:
    """Flag a moment in the session being recorded (music mode places it in time)."""

    from . import listener
    from .recorder import MARK_KINDS

    current = status()
    background = listener.status()
    recording = (current or {}).get("recording") or (background or {}).get("recording")
    if not recording:
        raise RuntimeError("nothing is recording right now (play some music first)")
    if kind not in MARK_KINDS:
        raise ValueError(f"Mark kinds are: {', '.join(MARK_KINDS)}")
    entry = {"wall": time.time(), "kind": kind, "note": note}
    runner.MARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(runner.MARKS_PATH, "a") as marks:
        marks.write(json.dumps(entry) + "\n")
    return entry


def stop(timeout: float = 8.0) -> bool:
    """Ask music mode to settle the light back and exit. True if it was running."""

    pid = _pid()
    if pid is None or not _alive(pid):
        PID_PATH.unlink(missing_ok=True)
        return False
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)
    PID_PATH.unlink(missing_ok=True)
    return True


def brightness_range(mood: str, offset: float) -> tuple[float, float]:
    low, high = MOODS[mood].brightness
    return low + offset, high + offset


def set_brightness(percent: float) -> float:
    """Centre music mode's brightness range on percent. Returns the offset."""

    current = status()
    mood = (current or {}).get("mood") or "ethereal"
    low, high = MOODS[mood].brightness
    offset = percent - (low + high) / 2
    _write_private(runner.CONTROL_PATH, {"brightness_offset": offset})
    return offset
