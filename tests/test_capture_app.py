"""tools/capture_app.py: reading the helper's stream and the command it runs."""

import io
import sys
from pathlib import Path
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import capture_app  # noqa: E402


def stream(rate, samples):
    return io.BytesIO(f"RATE {rate}\n".encode() + np.asarray(samples, dtype="<f4").tobytes())


def test_reads_the_rate_header_and_stops_at_the_length_asked():
    rate, samples = capture_app.read_stream(stream(1000, np.linspace(-1, 1, 5000)), seconds=2.0)
    assert rate == 1000
    assert len(samples) == 2000
    assert samples[0] == -1.0


def test_a_short_stream_returns_what_there_is():
    rate, samples = capture_app.read_stream(stream(1000, np.zeros(300)), seconds=2.0)
    assert len(samples) == 300


def test_wav_round_trip(tmp_path):
    path = tmp_path / "out.wav"
    capture_app.write_wav(path, 1000, np.array([0.0, 0.5, -0.5], dtype=np.float32))
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (1000, 1, 2)
        data = np.frombuffer(w.readframes(3), "<i2")
    assert list(data) == [0, 16383, -16383]


def test_silent_capture_command_taps_only_the_app_and_mutes_it():
    cmd = capture_app.command("/bin/helper", "com.google.Chrome", mute=True)
    assert cmd == ["/bin/helper", "--only-bundle", "com.google.Chrome", "--mute"]


def test_a_capture_louder_than_full_scale_is_scaled_down_not_clipped(tmp_path):
    # The tap's mono mixdown of a stereo app can reach twice full scale.
    path = tmp_path / "loud.wav"
    capture_app.write_wav(path, 1000, np.array([0.0, 1.0, -2.0], dtype=np.float32))
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(3), "<i2")
    assert list(data) == [0, 16383, -32767]
