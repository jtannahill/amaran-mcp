"""Record listening sessions locally, for training music mode on the owner's music.

A session is a directory under CONFIG_DIR/recordings with FLAC segments
(24 kHz mono, a new file every 30 minutes), the runner's feature log, and
marks.json: moments the owner flagged ("lift", "vocal", ...) in session time.
Nothing is ever deleted automatically; at the size cap recording just stops.
The recordings folder is readable by this user only: it holds what the Mac played.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from ..config import CONFIG_DIR

log = logging.getLogger(__name__)

RECORDINGS_DIR = CONFIG_DIR / "recordings"
OUTPUT_RATE = 24000
MARK_KINDS = ("breakdown", "lift", "vocal", "build", "drop", "other")
CAP_CHECK_SECONDS = 60.0  # how often a session re-checks the size cap
# A launch agent's PATH has no Homebrew folders (seen live: the listener crashed).
FFMPEG_CANDIDATES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")


def ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in FFMPEG_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return candidate
    return "ffmpeg"


def recordings_size() -> int:
    if not RECORDINGS_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in RECORDINGS_DIR.rglob("*") if p.is_file())


class Recorder:
    def __init__(self, directory: Path, cap_bytes: int = 30 * 2**30, segment_seconds: float = 1800.0) -> None:
        self.directory = directory
        self.cap_bytes = cap_bytes
        self.segment_seconds = segment_seconds
        self.stopped: str | None = None  # why recording stopped, if it did
        self.seconds = 0.0  # audio recorded so far
        self._segment = 0
        self._segment_seconds = 0.0
        self._encoder: subprocess.Popen | None = None
        self._marks: list[dict[str, Any]] = []
        self._base_size = recordings_size()
        self._checked_at = 0.0
        if self._base_size >= cap_bytes:
            self.stopped = self._cap_message()
            log.warning(self.stopped)

    def _cap_message(self) -> str:
        return f"recording stopped at the {self.cap_bytes / 2**30:.0f} GB cap"

    def _make_directory(self) -> None:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        RECORDINGS_DIR.chmod(0o700)  # also closes up folders made before this rule
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _open_segment(self, rate: int) -> None:
        self._make_directory()
        self._segment += 1
        path = self.directory / f"audio-{self._segment:03d}.flac"
        try:
            self._encoder = subprocess.Popen(
                [ffmpeg(), "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
                 "-ar", str(OUTPUT_RATE), "-c:a", "flac", "-y", str(path)],
                stdin=subprocess.PIPE)
        except OSError as err:
            self._encoder = None
            self.stopped = f"recording stopped: ffmpeg could not start ({err})"
            log.error(self.stopped)
        self._segment_seconds = 0.0

    def _close_segment(self) -> None:
        encoder, self._encoder = self._encoder, None
        if encoder:
            try:
                encoder.stdin.close()
            except OSError:
                pass
            try:
                encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log.error("ffmpeg did not finish a segment in 30 s; stopping it")
                encoder.kill()
                encoder.wait()

    def _written(self) -> int:
        if not self.directory.exists():
            return self._base_size
        return self._base_size + sum(p.stat().st_size for p in self.directory.glob("audio-*.flac"))

    def write(self, block: np.ndarray, rate: int) -> None:
        if self.stopped:
            return
        if self._encoder is None:
            self._open_segment(rate)
            if self._encoder is None:
                return
        try:
            self._encoder.stdin.write(np.asarray(block, dtype="<f4").tobytes())
        except (BrokenPipeError, OSError) as err:
            encoder, self._encoder = self._encoder, None
            encoder.kill()
            encoder.wait()  # reap it rather than leave a zombie
            self.stopped = f"recording stopped: ffmpeg failed ({err})"
            log.error(self.stopped)
            return
        dt = len(block) / rate
        self.seconds += dt
        self._segment_seconds += dt
        if self._segment_seconds >= self.segment_seconds:
            self._close_segment()
        if self.seconds - self._checked_at >= CAP_CHECK_SECONDS or self._encoder is None:
            self._checked_at = self.seconds
            if self._written() >= self.cap_bytes:
                self._close_segment()
                self.stopped = self._cap_message()
                log.warning(self.stopped)

    def mark(self, kind: str, note: str | None = None, seconds_ago: float = 0.0) -> dict[str, Any]:
        """Flag a moment, `seconds_ago` before the current end of the recording."""

        if self.seconds == 0:
            raise RuntimeError("nothing recorded yet")
        if kind not in MARK_KINDS:
            raise ValueError(f"Mark kinds are: {', '.join(MARK_KINDS)}")
        entry = {"t": round(max(0.0, self.seconds - seconds_ago), 2), "kind": kind, "note": note}
        self._marks.append(entry)
        self._make_directory()
        (self.directory / "marks.json").write_text(json.dumps(self._marks, indent=1))
        return entry

    def close(self) -> None:
        self._close_segment()
        if self._written() >= self.cap_bytes and not self.stopped:
            self.stopped = self._cap_message()
