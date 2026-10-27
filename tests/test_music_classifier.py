"""Sound classes from the capture helper, on their own file descriptor."""

import asyncio
import sys

from amaran_mcp.music.capture import SystemAudio

FAKE_WITH_CLASSES = """
import json, os, struct, sys, time
fd = int(sys.argv[sys.argv.index('--classify-fd') + 1])
sys.stdout.buffer.write(b'RATE 24000\\n')
for i in range(2):
    os.write(fd, (json.dumps({"t": i * 0.5, "singing": 0.2 + 0.6 * i, "speech": 0.1}) + "\\n").encode())
sys.stdout.buffer.write(struct.pack('<f', 0.1) * 24000)
sys.stdout.buffer.flush()
time.sleep(0.3)
"""

FAKE_WITHOUT_FLAG = """
import struct, sys
if '--classify-fd' in sys.argv:
    print('amaran-systemtap: unknown option --classify-fd', file=sys.stderr)
    sys.exit(2)
sys.stdout.buffer.write(b'RATE 24000\\n')
sys.stdout.buffer.write(struct.pack('<f', 0.1) * 24000)
sys.stdout.buffer.flush()
"""


async def drain(src):
    try:
        async for _ in src.blocks():
            await asyncio.sleep(0.01)
    finally:
        await src.close()


def test_latest_classes_come_from_the_side_channel(tmp_path):
    fake = tmp_path / "helper.py"
    fake.write_text(FAKE_WITH_CLASSES)
    src = SystemAudio(command=[sys.executable, "-I", str(fake)])
    asyncio.run(drain(src))
    assert src.latest_classes() == {"t": 0.5, "singing": 0.8, "speech": 0.1}


def test_helper_without_classifier_still_streams(tmp_path):
    fake = tmp_path / "old_helper.py"
    fake.write_text(FAKE_WITHOUT_FLAG)
    src = SystemAudio(command=[sys.executable, "-I", str(fake)])
    asyncio.run(drain(src))
    assert src.rate == 24000 and src.latest_classes() is None
