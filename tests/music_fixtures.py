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


NOTES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def chord(rate: int, seconds: float, root: str, minor: bool, octave: int = 4) -> np.ndarray:
    """A sustained triad (with a bass root an octave down), soft attack and release."""

    n = int(rate * seconds)
    t = np.arange(n) / rate
    base = 12 * (octave + 1) + NOTES[root]  # MIDI number
    out = np.zeros(n)
    for midi, gain in ((base - 12, 0.6), (base, 1.0), (base + (3 if minor else 4), 0.9), (base + 7, 0.8)):
        f = 440.0 * 2 ** ((midi - 69) / 12)
        out += gain * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))
    fade = np.minimum(1, np.minimum(t / 0.2, (seconds - t) / 0.2))
    return (0.05 * out * fade).astype(np.float32)


def progression(rate: int, chords: list[tuple[str, bool]], seconds_each: float) -> np.ndarray:
    return np.concatenate([chord(rate, seconds_each, root, minor) for root, minor in chords])


def riser(rate: int, seconds: float) -> np.ndarray:
    """A build: noise sweeping upward plus snare hits that speed up."""

    rng = np.random.default_rng(RNG_SEED)
    n = int(rate * seconds)
    out = np.zeros(n)
    seg = int(rate * 0.25)
    for start in range(0, n, seg):
        end = min(start + seg, n)
        noise = rng.standard_normal(end - start)
        spectrum = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(end - start, 1 / rate)
        cutoff = 200 * (40 ** (start / n))  # 200 Hz rising to about 8 kHz
        spectrum[freqs < cutoff] = 0
        out[start:end] = np.fft.irfft(spectrum, end - start) * (0.05 + 0.25 * start / n)
    pos, gap = 0.0, 0.45
    hit = int(rate * 0.06)
    while pos < seconds - 0.1:
        i = int(pos * rate)
        burst = rng.standard_normal(min(hit, n - i)) * np.exp(-np.arange(min(hit, n - i)) / rate * 50)
        out[i : i + len(burst)] += 0.3 * burst
        pos += gap
        gap = max(0.06, gap * 0.93)
    return out.astype(np.float32)
