"""Bring live audio to the rate music mode was trained at.

Every training feature and spectrogram was computed from audio decoded to
24 kHz, but the Mac plays at its output device's rate, usually 48 kHz. The
analyser reads levels and spectra whose numbers depend on the rate (the
spectrogram's band power, the spectral centroid), so live audio goes through
this first: a windowed-sinc low-pass below the new Nyquist, then sampling at
the new rate, keeping state across blocks so a stream comes out continuous.
The low-pass copies the gentle roll-off of ffmpeg's default resampler, which
decoded every training track (0.82 at 11 kHz, 0.58 at 11.5 kHz): a sharper
filter keeps more of the top band than the model ever saw, and on the live
check that cost more agreement than the resampling saved.
"""

from __future__ import annotations

import numpy as np

ANALYSIS_RATE = 24000
TAPS = 65  # odd, with a delay that halves to whole samples
DELAY = (TAPS - 1) // 2  # input samples the output lags by (0.7 ms at 48 kHz)
PASSBAND = 0.965  # of the output Nyquist; with BETA, fitted to ffmpeg's response
BETA = 6.0  # Kaiser window shape


class Resampler:
    """Streaming resampler from `rate` to `out_rate`; any block sizes in, continuous out."""

    def __init__(self, rate: int, out_rate: int = ANALYSIS_RATE) -> None:
        self.rate, self.out_rate = rate, out_rate
        self._step = rate / out_rate  # input samples per output sample
        cutoff = 0.5 * PASSBAND * min(1.0, out_rate / rate)  # cycles per input sample
        n = np.arange(TAPS) - (TAPS - 1) / 2
        taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(TAPS, BETA)
        self._taps = (taps / taps.sum()).astype(np.float64)
        self._history = np.zeros(TAPS - 1)
        self._last = 0.0  # the last filtered sample, for interpolating across blocks
        self._pos = 1.0  # next output position, counting the carried-over sample as 0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        if self.rate == self.out_rate:
            return block
        x = np.concatenate([self._history, np.asarray(block, dtype=np.float64)])
        self._history = x[-(TAPS - 1):]
        filtered = np.concatenate([[self._last], np.convolve(x, self._taps, mode="valid")])
        self._last = filtered[-1]
        positions = np.arange(self._pos, len(filtered) - 1, self._step)
        self._pos = (positions[-1] + self._step if len(positions) else self._pos) - (len(filtered) - 1)
        left = positions.astype(np.int64)
        frac = positions - left
        return (filtered[left] * (1 - frac) + filtered[np.minimum(left + 1, len(filtered) - 1)] * frac).astype(np.float32)
