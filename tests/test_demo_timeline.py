"""tools/demo_timeline.py: a session's marks in song time."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import demo_timeline  # noqa: E402


def test_the_same_moment_called_twice_counts_once_even_with_another_mark_between(tmp_path):
    marks = tmp_path / "marks.json"
    marks.write_text(json.dumps([{"t": 10.0, "kind": "lift", "note": None},
                                 {"t": 11.0, "kind": "build", "note": None},
                                 {"t": 12.5, "kind": "lift", "note": None},
                                 {"t": 30.0, "kind": "lift", "note": None}]))
    assert demo_timeline.marks(marks, start=0.0, duration=60.0) == [[10.0, "Lift"], [11.0, "Build"], [30.0, "Lift"]]
