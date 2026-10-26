"""Control through amaran Desktop's local OpenAPI WebSocket.

When amaran Desktop is running it holds the light's one Bluetooth connection,
so we ask it to send commands instead. The API (ws://127.0.0.1:33782/ws) is
documented at tools.sidus.link/openapi/docs; companion-module-amaran-desktop
(Bitfocus, MIT) showed the message and token format.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import os
import time
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import websockets

from .config import Capabilities, Light
from .light import LightBase, LightState

log = logging.getLogger(__name__)

HOST, PORT = "127.0.0.1", 33782
URL = f"ws://{HOST}:{PORT}/ws"

# Desktop's tint scale is 0..200 with 100 neutral; ours is -10..+10.
GM_STEP = 10


class DesktopError(RuntimeError):
    """amaran Desktop is not reachable or refused a request."""


def desktop_running(timeout: float = 0.3) -> bool:
    """True if amaran Desktop is listening for API connections."""

    import socket

    try:
        with socket.create_connection((HOST, PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _token() -> str:
    """The request token: an AES-256-GCM encrypted timestamp, base64 of iv|tag|ct.

    With AMARAN_OPENAPI_KEY (the key from an approved OpenAPI application, as
    hex or base64) the timestamp is encrypted with it. Without one, a random key
    is used, as the Companion module does; current Desktop builds accept that.
    """

    configured = os.environ.get("AMARAN_OPENAPI_KEY")
    if configured:
        try:
            key = bytes.fromhex(configured)
        except ValueError:
            key = base64.b64decode(configured)
    else:
        key = os.urandom(32)
    iv = os.urandom(12)
    sealed = AESGCM(key).encrypt(iv, str(int(time.time())).encode(), None)
    return base64.b64encode(iv + sealed[-16:] + sealed[:-16]).decode()


class DesktopClient:
    """One WebSocket to amaran Desktop, opened on first use."""

    def __init__(self) -> None:
        self._ws: Any = None
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def request(self, action: str, node_id: str | None = None, **args: Any) -> Any:
        async with self._lock:
            for attempt in range(2):
                try:
                    return await self._request(action, node_id, args)
                except (OSError, websockets.ConnectionClosed) as err:
                    await self.close()
                    if attempt:
                        raise DesktopError(f"amaran Desktop is not reachable ({err})") from err

    async def _request(self, action: str, node_id: str | None, args: dict[str, Any]) -> Any:
        if self._ws is None:
            self._ws = await websockets.connect(URL, open_timeout=3)
        request_id = f"mcp-{next(self._ids)}"
        message = {
            "version": 2,
            "type": "request",
            "client_id": "amaran-mcp",
            "request_id": request_id,
            "action": action,
            "args": args,
            "token": _token(),
        }
        if node_id:
            message["node_id"] = node_id
        await self._ws.send(json.dumps(message))
        deadline = time.monotonic() + 5
        while (left := deadline - time.monotonic()) > 0:
            try:
                reply = json.loads(await asyncio.wait_for(self._ws.recv(), left))
            except TimeoutError:
                break
            # Errors echo the request instead of carrying request_id.
            if reply.get("request_id") != request_id and (reply.get("request") or {}).get("request_id") != request_id:
                continue  # an event, or a reply to someone else
            if reply.get("code") != 0:
                raise ValueError(f"amaran Desktop refused {action}: {reply.get('message')}")
            return reply.get("data")
        raise DesktopError(f"amaran Desktop did not answer {action}")

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()


class DesktopLight(LightBase):
    """The same tools as AmaranLight, sent through amaran Desktop."""

    backend = "amaran Desktop"

    def __init__(self, light: Light | None = None, client: DesktopClient | None = None) -> None:
        self.light = light
        self.client = client or DesktopClient()
        self.node_id: str | None = None
        self.name = light.name if light else "amaran light"
        self.model = light.model if light else "amaran light"
        self.caps = Capabilities()
        self.state = LightState()

    async def _node(self) -> str:
        """Find our light among Desktop's fixtures and learn what it supports."""

        if self.node_id:
            return self.node_id
        fixtures = await self.client.request("get_fixture_list") or []
        if not fixtures:
            raise DesktopError("amaran Desktop has no lights paired")
        match = fixtures[0]
        if self.light:
            # Desktop names a node by product code plus the end of the MAC.
            suffix = self.light.mac.replace(":", "")[-6:].upper()
            expected = f"{self.light.code}-{suffix}".upper()
            match = next((f for f in fixtures if str(f.get("node_id", "")).upper() == expected),
                         next((f for f in fixtures if f.get("name") == self.light.name), None))
            if match is None:
                names = ", ".join(str(f.get("name")) for f in fixtures)
                raise DesktopError(f"amaran Desktop does not have {self.light.name}; it has {names}")
        elif len(fixtures) > 1:
            names = ", ".join(str(f.get("name")) for f in fixtures)
            raise DesktopError(f"amaran Desktop has several lights ({names}); run amaran-mcp import --name")
        node_id = str(match["node_id"])
        config = await self.client.request("get_node_config", node_id) or {}
        effects = await self.client.request("get_system_effect_list", node_id) or []
        self.caps = Capabilities(
            hsi=bool(config.get("hsi_support")),
            cct=bool(config.get("cct_support")),
            kelvin_min=int(config.get("cct_min") or 2700),
            kelvin_max=int(config.get("cct_max") or 6500),
            tint=bool(config.get("gm_support")),
            effects=list(effects),
        )
        self.name = str(match.get("name") or self.name)
        self.node_id = node_id
        return node_id

    async def _call(self, action: str, **args: Any) -> Any:
        return await self.client.request(action, await self._node(), **args)

    # ── Reading ──────────────────────────────────────────────────────────────

    async def read_status(self) -> dict[str, Any]:
        """Power and brightness from Desktop; the mode is the last one we set.

        The API reports the stored white and colour values but not which mode
        is active, so a change made in the amaran apps shows only its power and
        brightness here.
        """

        s = self.state
        s.power = not await self._call("get_sleep")
        s.brightness = round((await self._call("get_intensity")) / 10)
        if s.mode in (None, "white"):
            cct = await self._call("get_cct")
            s.kelvin = cct.get("cct")
            s.tint = round((cct.get("gm", 100) - 100) / GM_STEP)
            s.mode = s.mode or "white"
        elif s.mode == "color":
            hsi = await self._call("get_hsi")
            s.hue, s.saturation = hsi.get("hue"), hsi.get("sat")
        return {
            "light": self.name,
            "model": self.model,
            "backend": self.backend,
            "reported_by": "amaran Desktop",
            **self._status(),
        }

    # ── Writing ──────────────────────────────────────────────────────────────

    async def _wake(self) -> None:
        if self.state.power is not True:
            await self._call("set_sleep", sleep=False)
            self.state.power = True

    async def set_power(self, on: bool) -> None:
        await self._call("set_sleep", sleep=not on)
        self.state.power = on

    async def set_brightness(self, percent: float) -> None:
        await self._wake()
        if self.state.mode == "effect" and self.state.effect:
            await self._call("set_system_effect", effect_type=self.state.effect, intensity=round(percent * 10))
        else:
            await self._call("set_intensity", intensity=round(percent * 10))
        self.state.brightness = round(percent)

    async def set_white(self, kelvin: int, brightness: float | None = None, tint: int = 0) -> None:
        await self._node()
        self._check_white(kelvin, tint)
        percent = self._brightness(brightness)
        await self._wake()
        args: dict[str, Any] = {"cct": kelvin, "intensity": round(percent * 10)}
        if self.caps.tint:
            args["gm"] = 100 + tint * GM_STEP
        await self._call("set_cct", **args)
        self.state = LightState(True, "white", round(percent), kelvin, tint)

    async def set_color(self, hue: float, saturation: float, brightness: float | None = None) -> None:
        await self._node()
        self._check_color()
        percent = self._brightness(brightness)
        hue, saturation = round(hue) % 360, round(max(0.0, min(100.0, saturation)))
        await self._wake()
        await self._call("set_hsi", hue=hue, sat=saturation, intensity=round(percent * 10))
        self.state = LightState(True, "color", round(percent), hue=hue, saturation=saturation)

    async def set_effect(self, effect: str, brightness: float | None = None) -> None:
        await self._node()
        effect = self._check_effect(effect)
        percent = self._brightness(brightness)
        await self._wake()
        await self._call("set_system_effect", effect_type=effect, intensity=round(percent * 10))
        self.state.mode, self.state.effect, self.state.brightness = "effect", effect, round(percent)

    async def stop_effect(self) -> None:
        """Desktop has no stop action: setting a steady white or colour ends it."""

        previous = self.state
        if previous.kelvin is None and previous.hue is None:
            cct = await self._call("get_cct")
            previous.kelvin = cct.get("cct")
        await self._restore_steady(previous)

    async def close(self) -> None:
        await self.client.close()
