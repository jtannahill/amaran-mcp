"""Live audio reaches the analysis at the rate music mode was trained at."""

import numpy as np
from scipy.signal import resample_poly

from amaran_mcp.music.analysis import Analyzer
from amaran_mcp.music.resample import ANALYSIS_RATE, DELAY, Resampler
from amaran_mcp.music.runner import load_model
from music_fixtures import feed_in_blocks, kick_track


def tone(rate: int, hz: float, seconds: float = 1.0) -> np.ndarray:
    return np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate).astype(np.float32)


def peak_hz(x: np.ndarray, rate: int) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(mag)])


def in_blocks(r: Resampler, x: np.ndarray, sizes=(2000, 1837, 7, 512, 1)) -> np.ndarray:
    out, i, k = [], 0, 0
    while i < len(x):
        out.append(r(x[i : i + sizes[k % len(sizes)]]))
        i += sizes[k % len(sizes)]
        k += 1
    return np.concatenate(out)


def test_rates_in_come_out_at_the_analysis_rate_with_the_pitch_kept():
    for rate in (48000, 44100, 96000):
        out = in_blocks(Resampler(rate), tone(rate, 1000, 2.0))
        assert abs(len(out) - 2 * ANALYSIS_RATE) <= 2
        assert abs(peak_hz(out[2400:], ANALYSIS_RATE) - 1000) < 2


def test_block_sizes_do_not_change_the_stream():
    x = tone(48000, 440, 1.0) + 0.3 * tone(48000, 5000, 1.0)
    whole = Resampler(48000)(x)
    pieces = in_blocks(Resampler(48000), x)
    assert len(whole) == len(pieces) and np.allclose(whole, pieces, atol=1e-6)


def test_highs_above_the_new_nyquist_do_not_fold_back():
    # 15 kHz at 48 kHz would alias to 9 kHz at 24 kHz without the low-pass.
    out = Resampler(48000)(tone(48000, 15000, 1.0))[200:]
    assert np.sqrt(np.mean(out ** 2)) < 0.01


def test_the_top_rolls_off_like_the_decoder_the_training_audio_came_from():
    # ffmpeg's default resampler, measured: 1.00 at 10 kHz, 0.82 at 11 kHz, 0.58 at 11.5 kHz.
    for hz, gain in ((10000, 0.995), (11000, 0.821), (11500, 0.579)):
        out = Resampler(48000)(tone(48000, hz, 1.0))[200:]
        assert abs(np.sqrt(2 * np.mean(out ** 2)) - gain) < 0.05


def test_the_training_rate_passes_through():
    x = tone(ANALYSIS_RATE, 440)
    assert Resampler(ANALYSIS_RATE)(x) is x


def test_48k_audio_reads_like_the_same_audio_at_24k():
    """What the model sees live matches what it saw in training (decoded to 24 kHz)."""

    model = load_model()
    track = kick_track(48000, 40)
    live = Analyzer(48000, model=model)
    trained = Analyzer(ANALYSIS_RATE, model=model)
    # Blocks that land on the same samples, and the reference shifted by the filter's 1 ms delay.
    decoded = np.concatenate([np.zeros(DELAY // 2, np.float32), resample_poly(track, 1, 2).astype(np.float32)])
    live_fs = feed_in_blocks(live, track, block=2048)
    trained_fs = feed_in_blocks(trained, decoded, block=1024)
    # The per-second rows the model reads (section features, then spectral bands), and the spectrogram.
    assert np.abs(np.array(live._history) - np.array(trained._history)).max() < 0.05
    mel = np.abs(np.array(live._mel_frames) - np.array(trained._mel_frames))
    assert mel.mean() < 0.01 and mel.max() < 0.3
    assert live_fs[-1].section == trained_fs[-1].section
    assert abs(live_fs[-1].energy - trained_fs[-1].energy) < 0.05


def test_a_bad_sample_or_an_empty_block_does_not_break_the_analysis():
    a = Analyzer(48000)
    feed_in_blocks(a, kick_track(48000, 10), block=2000)
    poisoned = kick_track(48000, 1)[:2000].copy()
    poisoned[100] = np.nan
    a.feed(poisoned)
    assert a.feed(np.zeros(0, dtype=np.float32)) is not None
    after = feed_in_blocks(a, kick_track(48000, 20), block=2000)
    assert all(np.isfinite(f.energy) for f in after) and after[-1].is_music
