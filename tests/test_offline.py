"""Checks that need no light: payload bytes, mesh round trip, import, state."""

import asyncio
import importlib
import json
import sqlite3

import pytest

from amaran_mcp.sidus import protocol

NET_KEY = "00112233445566778899aabbccddeeff"
APP_KEY = "ffeeddccbbaa99887766554433221100"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("AMARAN_MCP_HOME", str(tmp_path / "cfg"))
    from amaran_mcp import config
    importlib.reload(config)
    return tmp_path, config


def make_light(config, code="400H5"):
    return config.Light(name="Tube", model="amaran PT2c", code=code, node_address=2,
                        mac="A4:C1:38:00:00:01", net_key=NET_KEY, app_key=APP_KEY,
                        source_address=0x7123)


def test_payloads_match_known_samples():
    # Samples from amaran-hacs docs/protocol.md, verified on real lights.
    assert protocol.power_payload(True) == bytes.fromhex("8d 00 00 00 00 00 00 00 01 8c")
    assert protocol.power_payload(False) == bytes.fromhex("8c 00 00 00 00 00 00 00 00 8c")
    assert protocol.brightness_payload(500) == bytes.fromhex("0c 00 00 00 00 00 00 00 7d 8f")
    assert protocol.cct_payload(intensity=300, kelvin=5600) == bytes.fromhex("31 00 00 00 00 40 01 23 4b 82")
    assert protocol.hsi_payload(hue=45, saturation=60, intensity=800) == bytes.fromhex("fd 00 00 00 00 00 af 05 c8 81")


def test_effects_and_stop():
    from amaran_mcp.light import EFFECT_NAMES, _effect_payloads
    pt2c = ["cop_car", "faulty_bulb", "fire", "fireworks", "lightning",
            "paparazzi", "party_lights", "pulsing", "tv"]
    for flag in pt2c:
        packets = _effect_payloads(EFFECT_NAMES[flag], 40)
        assert len(packets) == 1 and packets[0][9] == 0x87
        assert packets[0][0] == sum(packets[0][1:]) & 0xFF
    # Documented first-generation stop packet.
    assert _effect_payloads("off") == [bytes.fromhex("96000000000000000f87")]


def test_capabilities_for_pt2c(home):
    _, config = home
    caps = config.capabilities(make_light(config))
    assert (caps.kelvin_min, caps.kelvin_max, caps.tint, caps.hsi) == (2700, 10000, True, True)
    assert "fire" in caps.effects and len(caps.effects) == 9


def test_mesh_round_trip_through_link(home):
    """Encode as the link does, then decode as a light would see it."""
    _, config = home
    light = make_light(config)
    store = config.StateStore()
    from amaran_mcp.link import MeshLink

    sent = []

    class FakeClient:
        mtu_size = 23  # forces proxy SAR segmentation
        async def write_gatt_char(self, _uuid, data, response):
            sent.append(bytes(data))

    link = MeshLink(light, store)
    payload = protocol.cct_payload(intensity=300, kelvin=5600)
    asyncio.run(link._send_access(FakeClient(), [payload]))

    # Reassemble the SAR segments.
    assert sent[0][0] >> 6 == 0b01 and sent[-1][0] >> 6 == 0b11
    pdu = bytes([sent[0][0] & 0x3F]) + b"".join(s[1:] for s in sent)
    msg = protocol.decode_mesh_proxy_access(net_key=bytes.fromhex(NET_KEY), app_key=bytes.fromhex(APP_KEY),
                                            iv_index=0, proxy_pdu=pdu)
    assert msg.source_address == 0x7123 and msg.destination_address == 2
    assert msg.sidus_payload == payload
    assert config.StateStore().next_seq == 2  # persisted before use


def test_reply_updates_state(home):
    """A status report from the light, fed through the notify handler."""
    _, config = home
    light = make_light(config)
    from amaran_mcp.light import AmaranLight

    lamp = AmaranLight(light, config.StateStore())
    # The light reports CCT 3200K, 22 %, tint +5 (report = setter with bit 79 clear).
    report = bytearray(protocol.cct_payload(intensity=220, kelvin=3200, gm=5))
    report[9] &= 0x7F
    report[1] |= 0x01  # power on
    report[0] = sum(report[1:]) & 0xFF
    reply = protocol.build_mesh_proxy_pdu(net_key=bytes.fromhex(NET_KEY), app_key=bytes.fromhex(APP_KEY),
                                          src=2, dst=1, seq=500, iv_index=0, sidus_payload=bytes(report))

    async def go():
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        lamp.link._waiters.append((lambda m: m if m.sidus_status else None, fut))
        lamp.link._on_notify(None, bytearray(reply))
        msg = await asyncio.wait_for(fut, 1)
        lamp._apply_report(msg.sidus_status, msg.sidus_payload)

    asyncio.run(go())
    s = lamp.state
    assert (s.power, s.mode, s.kelvin, s.brightness, s.tint) == (True, "white", 3200, 22, 5)


def test_import_from_desktop_db(home, tmp_path):
    _, config = home
    db = tmp_path / "amaran.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table mesh (uuid text, net_key text, app_key text, update_time int)")
        conn.execute("create table fixtures (uuid text, mesh_uuid text, mac_address text, code text, name text, node_address int, product_id int)")
        conn.execute("insert into mesh values ('m1', ?, ?, 1)", (NET_KEY.upper(), APP_KEY.upper()))
        conn.execute("insert into fixtures values ('f1', 'm1', 'A4C138000001', '', 'Desk tube', 2, 102)")
    fixtures = config.list_desktop_fixtures(db)
    assert fixtures[0]["code"] == "400H5" and fixtures[0]["model"] == "amaran PT2c"
    light = config.import_light(fixtures[0])
    assert light.mac == "A4:C1:38:00:00:01" and 0x7000 <= light.source_address < 0x7FFF
    assert (config.LIGHT_PATH.stat().st_mode & 0o777) == 0o600
    # Re-importing the same network keeps our address (and so our counter).
    assert config.import_light(fixtures[0]).source_address == light.source_address


def test_desktop_backend_against_fake_desktop(home, monkeypatch):
    """Message shape, node matching and tint scale, against a stand-in Desktop."""
    _, config = home
    import websockets
    from amaran_mcp import desktop

    seen = []

    async def fake_desktop(ws):
        async for raw in ws:
            msg = json.loads(raw)
            seen.append(msg)
            assert msg["version"] == 2 and msg["token"]
            data = {
                "get_fixture_list": [{"node_id": "OTHER-000000", "name": "Other"},
                                     {"node_id": "400H5-000001", "name": "Tube"}],
                "get_node_config": {"cct_support": True, "cct_min": 2700, "cct_max": 10000,
                                    "gm_support": True, "hsi_support": True},
                "get_system_effect_list": ["fire", "fault_bulb", "rainbow"],
                "get_sleep": False,
                "get_intensity": 300,
                "get_cct": {"cct": 3200, "intensity": 300, "gm": 150},
            }.get(msg["action"])
            await ws.send(json.dumps({"code": 0, "type": "response", "request_id": msg["request_id"],
                                      "action": msg["action"], "data": data}))

    async def go():
        async with websockets.serve(fake_desktop, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            monkeypatch.setattr(desktop, "URL", f"ws://127.0.0.1:{port}/ws")
            lamp = desktop.DesktopLight(make_light(config))
            status = await lamp.read_status()
            await lamp.set_white(5600, 40, tint=-3)
            await lamp.set_effect("faulty_bulb", 20)  # Bluetooth's spelling still works
            await lamp.close()
            return status

    status = asyncio.run(go())
    assert (status["kelvin"], status["tint"], status["brightness"], status["power"]) == (3200, 5, 30, True)
    set_cct = next(m for m in seen if m["action"] == "set_cct")
    assert set_cct["node_id"] == "400H5-000001"
    assert set_cct["args"] == {"cct": 5600, "intensity": 400, "gm": 70}
    effect = next(m for m in seen if m["action"] == "set_system_effect")
    assert effect["args"] == {"effect_type": "fault_bulb", "intensity": 200}
