"""Log-mel spectrogram at 10 frames a second: the spectrogram model's input.

The same code builds training data from files (mel_of) and streams live
audio (MelFrontend), so both see identical numbers. Each frame is one 100 ms
chunk, Hann-windowed, in 64 triangular mel bands from 30 Hz to 11 kHz.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

FRAMES_PER_SECOND = 10
MELS = 64
FMIN, FMAX = 30.0, 11000.0


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def _setup(rate: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    hop = rate // FRAMES_PER_SECOND
    n_fft = 1 << int(np.ceil(np.log2(hop)))
    freqs = np.fft.rfftfreq(n_fft, 1 / rate)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array(FMIN)), _hz_to_mel(np.array(min(FMAX, rate / 2))), MELS + 2))
    bank = np.zeros((MELS, len(freqs)))
    for i in range(MELS):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (freqs - lo) / (mid - lo)
        falling = (hi - freqs) / (hi - mid)
        bank[i] = np.clip(np.minimum(rising, falling), 0, None)
    bank /= bank.sum(axis=1, keepdims=True) + 1e-12
    return hop, n_fft, np.hanning(hop), bank


def _frames(chunks: np.ndarray, rate: int) -> np.ndarray:
    hop, n_fft, window, bank = _setup(rate)
    power = np.abs(np.fft.rfft(chunks * window, n=n_fft, axis=-1)) ** 2
    return np.log10(power @ bank.T + 1e-10).astype(np.float32)


def mel_of(samples: np.ndarray, rate: int) -> np.ndarray:
    """A whole recording as (frames, MELS)."""

    hop = _setup(rate)[0]
    n = len(samples) // hop
    if n == 0:
        return np.zeros((0, MELS), np.float32)
    return _frames(samples[: n * hop].astype(np.float64).reshape(n, hop), rate)


class MelFrontend:
    """Streaming version of mel_of: feed blocks of any size, get finished frames."""

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.hop = _setup(rate)[0]
        self._buffer = np.zeros(0, np.float64)

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float64)])
        n = len(self._buffer) // self.hop
        if n == 0:
            return []
        chunks = self._buffer[: n * self.hop].reshape(n, self.hop)
        self._buffer = self._buffer[n * self.hop :]
        return list(_frames(chunks, self.rate))
