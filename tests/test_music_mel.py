"""The log-mel front end shared by spectrogram training and runtime."""

import numpy as np

from amaran_mcp.music.mel import FRAMES_PER_SECOND, MELS, MelFrontend

RATE = 24000


def frames_of(samples, block=1024):
    m = MelFrontend(RATE)
    out = []
    for i in range(0, len(samples) - block + 1, block):
        out.extend(m.feed(samples[i : i + block]))
    return np.array(out)


def test_ten_frames_per_second_of_64_mels():
    frames = frames_of(np.zeros(RATE * 5, np.float32) + 1e-3)
    assert frames.shape[1] == MELS == 64
    assert abs(len(frames) - 5 * FRAMES_PER_SECOND) <= 1


def test_a_tone_lands_in_the_right_mel_band():
    t = np.arange(RATE * 2) / RATE
    low = frames_of((0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)).mean(axis=0)
    high = frames_of((0.3 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)).mean(axis=0)
    assert np.argmax(low) < np.argmax(high)
    assert np.argmax(low) < MELS // 3 and np.argmax(high) > MELS // 2


def test_whole_file_and_streaming_agree():
    from amaran_mcp.music.mel import mel_of
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=RATE * 3).astype(np.float32)
    streamed = frames_of(x)
    whole = mel_of(x, RATE)
    n = min(len(streamed), len(whole))
    assert np.allclose(streamed[:n], whole[:n], atol=1e-4)
