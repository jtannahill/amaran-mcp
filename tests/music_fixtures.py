"""Synthetic audio for music-mode tests: kicks, pads, speech-like bursts, silence."""

from __future__ import annotations

import numpy as np

RNG_SEED = 7


def _pads(rate: int, seconds: float) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) / 3
    swell = 1 + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    return (0.15 * chord * swell).astype(np.float32)


def kick_track(rate: int, seconds: float, bpm: float = 125, pads: bool = True) -> np.ndarray:
    """Four-on-the-floor kick with off-beat hats, over pads."""

    rng = np.random.default_rng(RNG_SEED)
    n = int(rate * seconds)
    out = _pads(rate, seconds) if pads else np.zeros(n, np.float32)
    beat = int(rate * 60 / bpm)
    kick_len = int(rate * 0.25)
    k = np.arange(kick_len) / rate
    kick = 0.8 * np.sin(2 * np.pi * (50 + 60 * np.exp(-k * 30)) * k) * np.exp(-k * 12)
    hat_len = int(rate * 0.04)
    hat = 0.08 * rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / rate * 80)
    for start in range(0, n, beat):
        end = min(start + kick_len, n)
        out[start:end] += kick[: end - start]
        off = start + beat // 2
        if off < n:
            hend = min(off + hat_len, n)
            out[off:hend] += hat[: hend - off]
    return out.astype(np.float32)


def pads_only(rate: int, seconds: float) -> np.ndarray:
    return _pads(rate, seconds)


def speech_like(rate: int, seconds: float) -> np.ndarray:
    """Syllable-length bursts of 300-3000 Hz noise with gaps; no bass."""

    rng = np.random.default_rng(RNG_SEED)
    n = int(rate * seconds)
    out = np.zeros(n, np.float32)
    pos = 0
    while pos < n:
        burst = int(rate * rng.uniform(0.15, 0.3))
        noise = rng.standard_normal(burst)
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(burst, 1 / rate)
        spectrum[(freqs < 300) | (freqs > 3000)] = 0
        shaped = np.fft.irfft(spectrum, burst) * np.hanning(burst)
        end = min(pos + burst, n)
        out[pos:end] = 0.3 * shaped[: end - pos] / (np.abs(shaped).max() + 1e-9)
        pos = end + int(rate * rng.uniform(0.1, 0.4))
    return out


def silence(rate: int, seconds: float) -> np.ndarray:
    return np.zeros(int(rate * seconds), np.float32)


def feed_in_blocks(analyzer, samples: np.ndarray, block: int = 1024) -> list:
    return [analyzer.feed(samples[i : i + block]) for i in range(0, len(samples) - block + 1, block)]


class FakeSource:
    """A capture source that plays fixture audio as fast as it is read."""

    name = "fake"

    def __init__(self, parts, block=1024, rate=24000):
        import numpy as _np
        self._samples = _np.concatenate(parts)
        self._block = block
        self.rate = rate

    async def blocks(self):
        import asyncio as _asyncio
        for i in range(0, len(self._samples) - self._block + 1, self._block):
            yield self._samples[i : i + self._block]
            await _asyncio.sleep(0)

    async def close(self):
        pass


def write_wav(directory, samples, rate, name="clip.wav"):
    import wave
    path = directory / name
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path
