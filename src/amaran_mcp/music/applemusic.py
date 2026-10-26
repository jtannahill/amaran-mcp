"""What the Music app on this Mac is playing, if it is running.

Read through osascript. The script checks that Music is already running
before talking to it, so asking never launches the app. Without Automation
permission (or with Music closed) there is simply no track.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import logging
import subprocess

log = logging.getLogger(__name__)

SCRIPT = """
if application "Music" is running then
    tell application "Music"
        if player state is stopped then return ""
        set t to current track
        return (player state as text) & tab & (name of t) & tab & (artist of t)
    end tell
end if
return ""
"""

_warned = False


@dataclass(frozen=True)
class Track:
    name: str
    artist: str
    playing: bool


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=3)


def now_playing(run: Callable[[list[str]], subprocess.CompletedProcess] = _run) -> Track | None:
    global _warned
    try:
        result = run(["osascript", "-e", SCRIPT])
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        if "-1743" in (result.stderr or "") and not _warned:
            _warned = True
            log.warning("To show track names, allow Terminal to control Music in "
                        "System Settings, Privacy and Security, Automation.")
        return None
    parts = result.stdout.rstrip("\n").split("\t")
    if len(parts) != 3:
        return None
    state, name, artist = parts
    return Track(name, artist, state == "playing")


def anchor_for(track: Track, anchors: int) -> int:
    """The palette stop a track starts on: the same song, the same colour."""

    digest = hashlib.sha256(f"{track.name}|{track.artist}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % anchors
