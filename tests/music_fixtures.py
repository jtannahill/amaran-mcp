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


def sung(rate: int, seconds: float, notes: list[float]) -> np.ndarray:
    """A sung "ah": harmonics shaped by vowel formants, 5.5 Hz vibrato, sustained
    notes with short glides between them (female range in the tests)."""

    n = int(rate * seconds)
    t = np.arange(n) / rate
    each = seconds / len(notes)
    index = np.minimum((t // each).astype(int), len(notes) - 1)
    target = np.log(np.array(notes))[index]
    glide = int(rate * 0.12)
    log_f0 = np.convolve(target, np.ones(glide) / glide, mode="same")  # short glides
    log_f0[: glide // 2] = target[: glide // 2]
    f0 = np.exp(log_f0) * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * t))  # vibrato, about 50 cents
    phase = 2 * np.pi * np.cumsum(f0) / rate
    formants = ((800, 1.0, 120), (1200, 0.6, 150), (2500, 0.3, 250))
    out = np.zeros(n)
    for h in range(1, 13):
        freq = h * f0
        gain = sum(g * np.exp(-0.5 * ((freq - c) / w) ** 2) for c, g, w in formants) + 0.02
        out += gain * np.sin(h * phase) / h ** 0.5
    note_pos = (t % each) / each
    shape = np.minimum(1, np.minimum(note_pos / 0.08, (1 - note_pos) / 0.08))
    return (0.05 * out * shape).astype(np.float32)


def speech_over(rate: int, seconds: float, bed: np.ndarray) -> np.ndarray:
    voice = speech_like(rate, seconds)
    n = min(len(voice), len(bed))
    return (bed[:n] + voice[:n]).astype(np.float32)


def hats_and_pads(rate: int, seconds: float, bpm: float = 125) -> np.ndarray:
    """An intro: off-beat hats over pads, no kick yet."""

    rng = np.random.default_rng(RNG_SEED)
    out = _pads(rate, seconds).astype(np.float64)
    beat = int(rate * 60 / bpm)
    hat_len = int(rate * 0.04)
    hat = 0.08 * rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / rate * 80)
    for start in range(beat // 2, len(out), beat):
        end = min(start + hat_len, len(out))
        out[start:end] += hat[: end - start]
    return out.astype(np.float32)


def arrangement(rate: int) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """A trance track in miniature, with where each section is."""

    parts = [
        ("silence", silence(rate, 2)),
        ("intro", hats_and_pads(rate, 16)),
        ("groove", kick_track(rate, 30)),
        ("breakdown", progression(rate, [("A", True), ("F", False), ("C", False), ("G", False)], 10.0)),
        ("build", riser(rate, 16) + progression(rate, [("A", True)], 16.0)),
        ("climax", kick_track(rate, 30) * 1.3),
        ("groove2", kick_track(rate, 10)),
    ]
    marks, t = {}, 0.0
    for name, samples in parts:
        marks[name] = (t, t + len(samples) / rate)
        t += len(samples) / rate
    return np.concatenate([p for _, p in parts]).astype(np.float32), marks


def make_features(**fields):
    """Features for renderer and runner tests, with quiet defaults."""

    from amaran_mcp.music.analysis import Features
    from amaran_mcp.music.harmony import Harmony
    from amaran_mcp.music.tempo import Beat
    from amaran_mcp.music.timbre import Timbre

    beat = Beat(fields.pop("bpm", None), fields.pop("confidence", 0.0), fields.pop("bar", False),
                fields.pop("phrase", False), fields.pop("seconds_to_bar", None),
                fields.pop("seconds_to_phrase", None))
    root = fields.pop("chord_root", None)
    major = fields.pop("major", None if root is None else fields.get("balance", 0) >= 0)
    harmony = Harmony(None if root is None else "x", root, major, 0.9 if root is not None else 0.0,
                      fields.pop("balance", 0.0), fields.pop("change", False), fields.pop("lift", False))
    timbre = Timbre(fields.pop("vocal", 0.0), fields.pop("tension", 0.0), fields.pop("openness", 0.5),
                    fields.pop("level_db", -20.0))
    base = dict(energy=0.5, bass=0.5, is_music=True, digital_zero=False, section="groove",
                section_seconds=10.0, release=False, bloom=False)
    base.update(fields)
    return Features(beat=beat, harmony=harmony, timbre=timbre, **base)


def sub_bass_pads(rate: int, seconds: float) -> np.ndarray:
    """A breakdown that keeps its sub-bass: pads over a held 55 Hz sine, no kick."""

    t = np.arange(int(rate * seconds)) / rate
    fade = np.minimum(1, t / 0.5)
    return (_pads(rate, seconds) + 0.35 * np.sin(2 * np.pi * 55 * t) * fade).astype(np.float32)


def bass_swell(rate: int, seconds: float) -> np.ndarray:
    """Seen on a real episode: inside a breakdown the sub-bass swells and pumps
    gently (sidechain), without a kick."""

    t = np.arange(int(rate * seconds)) / rate
    level = np.where(t < seconds / 2, 0.2, 0.45)  # an abrupt swell halfway
    pump = 1 - 0.35 * (0.5 + 0.5 * np.cos(2 * np.pi * 2.3 * t))
    return (_pads(rate, seconds) + level * pump * np.sin(2 * np.pi * 55 * t)).astype(np.float32)
