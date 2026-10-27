"""The light link, the shared mesh counter and stopping music mode, when things go wrong."""

import asyncio
import importlib
import os
import signal
import subprocess
import sys
import time

import pytest
from bleak.exc import BleakError

NET_KEY = "00112233445566778899aabbccddeeff"
APP_KEY = "ffeeddccbbaa99887766554433221100"


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config as module
    importlib.reload(module)
    return module


def make_link(config):
    from amaran_mcp import link
    importlib.reload(link)
    light = config.Light(name="Tube", model="amaran PT2c", code="400H5", node_address=2,
                         mac="A4:C1:38:00:00:01", net_key=NET_KEY, app_key=APP_KEY, source_address=0x7123)
    return link, link.MeshLink(light, config.StateStore(), idle_timeout=60)


class Broken:
    """A connected client whose every write fails."""

    is_connected = True
    mtu_size = 185

    async def write_gatt_char(self, *_, **__):
        raise BleakError("write failed")

    async def disconnect(self):
        pass


# ── The mesh counter is shared by every process that sends ─────────────────────

def test_two_processes_never_reuse_a_sequence_number(config):
    server, music = config.StateStore(), config.StateStore()  # each read the file once
    taken = []
    for _ in range(50):
        taken += list(server.reserve(1)) + list(music.reserve(3))
    assert len(taken) == len(set(taken))
    assert config.StateStore().next_seq == max(taken) + 1


def test_saving_other_state_never_lowers_the_counter(config):
    behind, ahead = config.StateStore(), config.StateStore()
    ahead.reserve(100)
    behind.remember_device("UUID")
    assert config.StateStore().next_seq == 101


# ── Bluetooth errors reach callers as LinkError, never raw ─────────────────────

def test_a_write_that_fails_twice_is_a_link_error(config):
    link, mesh = make_link(config)

    async def go():
        async def ensure():
            return Broken()
        mesh._ensure = ensure
        with pytest.raises(link.LinkError):
            await mesh.send([b"\x00" * 10])
    asyncio.run(go())


def test_a_query_that_fails_twice_is_a_link_error(config):
    link, mesh = make_link(config)

    async def go():
        async def ensure():
            return Broken()
        mesh._ensure = ensure
        with pytest.raises(link.LinkError):
            await mesh.query(b"\x00" * 10, lambda m: m, timeout=0.1)
    asyncio.run(go())


def test_a_failed_scan_is_a_link_error(config, monkeypatch):
    link, mesh = make_link(config)

    async def refuse(*_, **__):
        raise BleakError("Bluetooth is turned off")
    monkeypatch.setattr(link.BleakScanner, "find_device_by_filter", refuse)
    with pytest.raises(link.LinkError, match="Bluetooth is on"):
        asyncio.run(mesh._find())


def test_a_neighbours_light_on_another_network_is_never_used(config, monkeypatch):
    link, mesh = make_link(config)

    class Device:
        address, name = "NEIGHBOUR", "light"

    class Advert:
        service_data = {link.PROXY_SERVICE: b"\x00" + b"\x99" * 8}  # a different network ID

    async def scan(filt, timeout):
        assert not filt(Device(), Advert())
        return None
    monkeypatch.setattr(link.BleakScanner, "find_device_by_filter", scan)
    with pytest.raises(link.LinkError, match="not advertising"):
        asyncio.run(mesh._find())


# ── Music mode's light writer outlives any one failed command ──────────────────

def test_the_writer_survives_an_unexpected_error(config):
    from amaran_mcp.music import runner
    importlib.reload(runner)
    seen, errors = [], []

    class Flaky:
        async def set_color(self, *args):
            if not seen:
                seen.append("boom")
                raise RuntimeError("unexpected")
            seen.append(args)

    async def go():
        writer = runner._Writer(lambda: Flaky(), errors.append)
        writer.post(("color", 1, 2, 3))
        assert await writer.drain(1.0)
        writer.post(("color", 4, 5, 6))
        assert await writer.drain(1.0)
        await writer.close()
    asyncio.run(go())
    assert seen == ["boom", (4, 5, 6)] and errors


# ── Stopping music mode ─────────────────────────────────────────────────────────

@pytest.fixture
def control(config):
    from amaran_mcp.music import control as module, runner
    importlib.reload(runner)
    importlib.reload(module)
    return module


def spawn(code: str) -> int:
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True).pid


def test_a_stale_pid_that_is_not_music_mode_is_left_alone(control, monkeypatch):
    other = spawn("import time; time.sleep(30)")
    try:
        control.PID_PATH.parent.mkdir(parents=True, exist_ok=True)
        control.PID_PATH.write_text(f"{other}\n")
        assert control.status() is None and control.stop() is None
        assert control._alive(other)  # never signalled
    finally:
        os.kill(other, signal.SIGKILL)


def test_stop_says_settled_when_music_mode_exits(control, monkeypatch):
    pid = spawn("import time; time.sleep(30)")  # exits on SIGTERM, as music mode does after its fade
    control.PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    control.PID_PATH.write_text(f"{pid}\n")
    monkeypatch.setattr(control, "_ours", lambda p: True)
    assert control.stop(timeout=5) == "settled"


def test_stop_says_killed_and_takes_the_whole_group_when_it_hangs(control, monkeypatch):
    code = ("import signal, subprocess, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)")
    pid = spawn(code)
    time.sleep(0.5)
    child = int(subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()[0])
    control.PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    control.PID_PATH.write_text(f"{pid}\n")
    monkeypatch.setattr(control, "_ours", lambda p: True)
    assert control.stop(timeout=1) == "killed"
    time.sleep(0.3)
    assert subprocess.run(["ps", "-p", str(child)], capture_output=True).returncode != 0  # the helper went too


# ── The server hands the light to music mode ───────────────────────────────────

def test_start_music_mode_releases_the_light_first(config, monkeypatch):
    from amaran_mcp import server
    order = []

    async def close_all():
        order.append("close")
    monkeypatch.setattr(server, "close_all", close_all)
    monkeypatch.setattr(server.control, "start", lambda mood, record=True: order.append("start") or {"running": True})
    asyncio.run(server.start_music_mode("ethereal"))
    assert order == ["close", "start"]
