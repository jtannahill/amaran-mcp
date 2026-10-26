"""Where music mode's audio comes from.

The main source is the Mac's own output, captured by a small Swift helper
(systemtap.swift) that is compiled on first use. It never opens a microphone,
so AirPods stay in their normal high-quality mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import hashlib
import logging
from pathlib import Path
import subprocess
from typing import Protocol

import numpy as np

from ..config import CONFIG_DIR

log = logging.getLogger(__name__)

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


class Source(Protocol):
    rate: int
    name: str

    def blocks(self) -> AsyncIterator[np.ndarray]: ...  # mono float32
    async def close(self) -> None: ...


def _block_size(rate: int) -> int:
    return max(512, rate // 24)  # about 40 ms


class SystemAudio:
    """What the Mac is playing, read from the helper's stdout."""

    name = "system audio"

    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command
        self._process: asyncio.subprocess.Process | None = None
        self.rate = 0

    async def start(self) -> None:
        if self._process:
            return
        command = self._command or [str(helper_path())]
        self._process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        header = (await asyncio.wait_for(self._process.stdout.readline(), 15)).decode().split()
        if len(header) != 2 or header[0] != "RATE":
            raise CaptureError(await self._failure("the audio helper did not start"))
        self.rate = int(header[1])

    async def _failure(self, fallback: str) -> str:
        await self._process.wait()
        message = (await self._process.stderr.read()).decode().strip()
        return message or f"{fallback} (exit code {self._process.returncode})"

    async def blocks(self) -> AsyncIterator[np.ndarray]:
        await self.start()
        size = _block_size(self.rate) * 4
        while True:
            try:
                data = await self._process.stdout.readexactly(size)
            except asyncio.IncompleteReadError as err:
                if err.partial:
                    yield np.frombuffer(err.partial[: len(err.partial) // 4 * 4], dtype="<f4").astype(np.float32)
                code = await self._process.wait()
                if code not in (0, -15):
                    raise CaptureError(await self._failure("the audio helper stopped")) from None
                return
            yield np.frombuffer(data, dtype="<f4").astype(np.float32)

    async def close(self) -> None:
        process, self._process = self._process, None
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()


BLUETOOTH_NAMES = ("airpods", "beats", "bluetooth", "hands-free", "headset")


def pick_microphone(devices: list[dict]) -> int | None:
    """The built-in microphone, never a Bluetooth headset (AirPods would drop
    to call quality the moment their microphone opens)."""

    for index, device in enumerate(devices):
        name = str(device.get("name", "")).lower()
        if device.get("max_input_channels", 0) < 1 or any(b in name for b in BLUETOOTH_NAMES):
            continue
        if ("macbook" in name or "built-in" in name) and "microphone" in name:
            return index
    return None


class MicAudio:
    """Fallback: the built-in microphone, for music played on speakers."""

    def __init__(self, index: int, name: str, rate: int) -> None:
        self._index, self.name, self.rate = index, name, rate
        self._stream = None

    async def blocks(self) -> AsyncIterator[np.ndarray]:
        import sounddevice

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)

        def callback(indata, _frames, _time, _status) -> None:
            block = indata[:, 0].astype(np.float32).copy()
            loop.call_soon_threadsafe(lambda: queue.full() or queue.put_nowait(block))

        self._stream = sounddevice.InputStream(
            device=self._index, channels=1, samplerate=self.rate,
            blocksize=_block_size(self.rate), dtype="float32", callback=callback)
        self._stream.start()
        while True:
            yield await queue.get()

    async def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream:
            stream.stop()
            stream.close()


async def open_source() -> Source:
    """System audio when the tap works, else the built-in microphone."""

    system = SystemAudio()
    try:
        await system.start()
        return system
    except (CaptureError, OSError, TimeoutError) as err:
        await system.close()
        log.warning("system audio unavailable (%s); trying the built-in microphone", err)
    import sounddevice

    devices = list(sounddevice.query_devices())
    index = pick_microphone(devices)
    if index is None:
        raise CaptureError("no system audio and no built-in microphone to listen with")
    rate = int(devices[index].get("default_samplerate") or 48000)
    return MicAudio(index, str(devices[index]["name"]), rate)
