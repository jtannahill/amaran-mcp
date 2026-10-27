"""Record one app's audio to a WAV file, optionally without it being heard.

Usage: uv run python tools/capture_app.py OUT.wav --seconds 320 [--bundle com.google.Chrome] [--mute]

Runs the music-mode audio helper with a tap on only the apps whose bundle ID
starts with --bundle. With --mute those apps go silent on the speakers while
they are recorded, so a track playing in a browser tab can be captured for
the demo page without anyone hearing it. The app must already be playing (or
paused mid-track) when this starts, because Core Audio only lists apps that
have opened audio. Recordings are local; do not commit them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import BinaryIO
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def command(helper: str, bundle: str, mute: bool) -> list[str]:
    return [helper, "--only-bundle", bundle] + (["--mute"] if mute else [])


def read_stream(stream: BinaryIO, seconds: float) -> tuple[int, np.ndarray]:
    """The helper's output: a "RATE <hz>" line, then float32 mono samples."""

    line = stream.readline()
    if not line:
        raise ValueError("the helper stopped before sending audio (is the app playing? see its message above)")
    header = line.decode().split()
    if len(header) != 2 or header[0] != "RATE":
        raise ValueError(f"unexpected helper output: {' '.join(header)!r}")
    rate = int(header[1])
    wanted = int(seconds * rate) * 4
    chunks, total = [], 0
    try:
        while total < wanted:
            chunk = stream.read(min(65536, wanted - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except KeyboardInterrupt:  # stopped early: keep what was read
        print("stopped early; keeping what was captured", file=sys.stderr)
    data = b"".join(chunks)
    return rate, np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").copy()


def write_wav(path: Path, rate: int, samples: np.ndarray) -> None:
    """16-bit mono; a capture louder than full scale (a stereo app's mono mixdown
    reaches about 2) is scaled down rather than clipped."""

    peak = float(np.abs(samples).max(initial=0))
    if peak == 0:
        raise ValueError("the capture is silent: the app played nothing the tap could hear")
    if peak > 1:
        samples = samples / peak
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    from amaran_mcp.music.capture import helper_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--bundle", default="com.google.Chrome")
    parser.add_argument("--mute", action="store_true")
    args = parser.parse_args()
    process = subprocess.Popen(command(str(helper_path()), args.bundle, args.mute), stdout=subprocess.PIPE)
    rate, samples = 0, np.zeros(0, dtype=np.float32)
    try:
        rate, samples = read_stream(process.stdout, args.seconds)
    except ValueError as err:
        sys.exit(f"capture failed: {err}")
    finally:
        # Whatever happened, the helper must go: while it runs, a muted app stays muted.
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if not rate:
        sys.exit("capture failed: nothing was read")
    try:
        write_wav(args.out, rate, samples)
    except ValueError as err:
        sys.exit(f"capture failed: {err}")
    print(f"{len(samples) / rate:.1f} s at {rate} Hz, peak {float(np.abs(samples).max(initial=0)):.2f}")


if __name__ == "__main__":
    main()
