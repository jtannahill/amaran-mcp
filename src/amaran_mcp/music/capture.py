"""Where music mode's audio comes from.

The main source is the Mac's own output, captured by a small Swift helper
(systemtap.swift) that is compiled on first use. It never opens a microphone,
so AirPods stay in their normal high-quality mode.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from ..config import CONFIG_DIR

HERE = Path(__file__).parent
SWIFT_SOURCE = HERE / "systemtap.swift"
INFO_PLIST = HERE / "Info.plist"


class CaptureError(RuntimeError):
    """No usable audio source."""


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (SWIFT_SOURCE, INFO_PLIST):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def helper_path() -> Path:
    """The compiled helper, building it when missing or out of date."""

    bin_dir = CONFIG_DIR / "bin"
    binary = bin_dir / "amaran-systemtap"
    stamp = bin_dir / "amaran-systemtap.sha256"
    wanted = _source_hash()
    if binary.exists() and stamp.exists() and stamp.read_text().strip() == wanted:
        return binary
    bin_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "swiftc", "-O", "-swift-version", "5", str(SWIFT_SOURCE), "-o", str(binary),
        "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist",
        "-Xlinker", str(INFO_PLIST),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as err:
        raise CaptureError("swiftc not found: install Xcode or its command line tools") from err
    if result.returncode != 0:
        raise CaptureError(f"could not build the audio helper:\n{result.stderr.strip()}")
    stamp.write_text(wanted + "\n")
    return binary
