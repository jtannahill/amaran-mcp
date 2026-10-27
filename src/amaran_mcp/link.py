"""One Bluetooth Mesh proxy connection to the light, opened on demand.

The light is the proxy: we connect to it over GATT, write encrypted mesh
messages to Proxy Data In, and read its replies from Proxy Data Out. The link
closes after a short idle period so the amaran apps can reach the light too.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import TypeVar

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.exc import BleakError

from .config import Light, StateStore
from .sidus.protocol import (
    DecodedAccessMessage,
    build_mesh_proxy_pdu,
    build_proxy_filter_pdu,
    decode_mesh_proxy_access,
    decode_secure_network_beacon,
    mesh_network_id,
)

log = logging.getLogger(__name__)

PROXY_SERVICE = "00001828-0000-1000-8000-00805f9b34fb"
PROXY_IN = "00002add-0000-1000-8000-00805f9b34fb"
PROXY_OUT = "00002ade-0000-1000-8000-00805f9b34fb"

PROXY_TYPE_NETWORK = 0x00
PROXY_TYPE_BEACON = 0x01
PROXY_TYPE_CONFIG = 0x02

# amaran-hacs found 50 ms between packets reliable on real lights.
PACKET_GAP = 0.05

T = TypeVar("T")


class LinkError(RuntimeError):
    """The light could not be reached."""


class MeshLink:
    def __init__(self, light: Light, store: StateStore, idle_timeout: float = 30.0) -> None:
        self.light = light
        self.store = store
        self.idle_timeout = idle_timeout
        self._net_key = bytes.fromhex(light.net_key)
        self._app_key = bytes.fromhex(light.app_key)
        self._client: BleakClient | None = None
        self._lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._beacon = asyncio.Event()
        self._filter_status = asyncio.Event()
        self._rx = bytearray()
        self._waiters: list[tuple[Callable[[DecodedAccessMessage], object], asyncio.Future]] = []

    # ── Public API ───────────────────────────────────────────────────────────

    async def send(self, payloads: list[bytes]) -> None:
        """Send 10-byte Sidus payloads to the light, in order."""

        async with self._lock:
            try:
                await self._send_access(await self._ensure(), payloads)
            except BleakError as err:
                log.info("write failed (%s); reconnecting once", err)
                await self._drop()
                try:
                    await self._send_access(await self._ensure(), payloads)
                except BleakError as again:
                    await self._drop()
                    raise LinkError(f"Lost the Bluetooth link to the light ({again}).") from again
            self._arm_idle()

    async def query(
        self,
        payload: bytes,
        pick: Callable[[DecodedAccessMessage], T | None],
        timeout: float = 2.5,
    ) -> T | None:
        """Send a request and return the first reply that pick() accepts."""

        async with self._lock:
            try:
                try:
                    return await self._query(payload, pick, timeout)
                except BleakError as err:
                    log.info("query failed (%s); reconnecting once", err)
                    await self._drop()
                    try:
                        return await self._query(payload, pick, timeout)
                    except BleakError as again:
                        await self._drop()
                        raise LinkError(f"Lost the Bluetooth link to the light ({again}).") from again
            finally:
                self._arm_idle()

    async def _query(self, payload: bytes, pick: Callable[[DecodedAccessMessage], T | None],
                     timeout: float) -> T | None:
        client = await self._ensure()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        waiter = (pick, future)
        self._waiters.append(waiter)
        try:
            for _ in range(2):
                await self._send_access(client, [payload])
                try:
                    return await asyncio.wait_for(asyncio.shield(future), timeout)
                except TimeoutError:
                    continue
            return None
        finally:
            self._waiters.remove(waiter)

    async def close(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        async with self._lock:
            await self._drop()

    # ── Connection ───────────────────────────────────────────────────────────

    async def _ensure(self) -> BleakClient:
        if self._client and self._client.is_connected:
            return self._client
        device = await self._find()
        log.info("connecting to %s (%s)", device.name or "light", device.address)
        client = BleakClient(device, disconnected_callback=self._on_disconnect)
        try:
            await client.connect(timeout=15.0)
        except (BleakError, TimeoutError) as err:
            raise LinkError(
                f"Found the light but could not connect ({err}). If the amaran app "
                "is open and connected to it, close the app and try again."
            ) from err
        self._client = client
        try:
            self._beacon.clear()
            self._filter_status.clear()
            self._rx.clear()
            await client.start_notify(PROXY_OUT, self._on_notify)
            # The proxy sends a Secure Network Beacon on connect; it carries the
            # current IV index and proves this light is on our network.
            try:
                await asyncio.wait_for(self._beacon.wait(), 3.0)
            except TimeoutError:
                log.info("no network beacon; using stored IV index %s", self.store.iv_index)
            seq = self.store.reserve(1)[0]
            await self._write(client, build_proxy_filter_pdu(
                net_key=self._net_key,
                app_key=self._app_key,
                src=self.light.source_address,
                seq=seq,
                iv_index=self.store.iv_index,
            ))
            try:
                await asyncio.wait_for(self._filter_status.wait(), 2.0)
            except TimeoutError:
                log.info("no proxy filter status; replies may not arrive")
        except BleakError as err:
            await self._drop()
            raise LinkError(f"Connected to the light, but it stopped answering ({err}).") from err
        except BaseException:
            await self._drop()
            raise
        if self._beacon.is_set():
            self.store.remember_device(device.address)
        return client

    async def _find(self) -> BLEDevice:
        # One scan that stops at the first sign of our light: the device that
        # last passed the beacon check, or a proxy advertising our network ID.
        # The light can go quiet for several seconds after a disconnect.
        network_id = mesh_network_id(self._net_key)
        remembered = self.store.ble_address
        others: dict[str, BLEDevice] = {}

        def ours(device: BLEDevice, adv: AdvertisementData) -> bool:
            if device.address == remembered:
                return True
            data = adv.service_data.get(PROXY_SERVICE)
            if data is None:
                return False
            if len(data) >= 9 and data[0] == 0x00 and data[1:9] == network_id:
                return True
            if data[:1] == b"\x01":
                # Node Identity: it hides its network, so it may be ours. A proxy
                # naming a different network is a neighbour's light, never a candidate.
                others[device.address] = device
            return False

        try:
            device = await BleakScanner.find_device_by_filter(ours, timeout=20.0)
        except BleakError as err:
            raise LinkError(
                f"Could not scan for the light ({err}). Check that Bluetooth is on and that "
                "this app may use it (System Settings, Privacy and Security, Bluetooth)."
            ) from err
        if device:
            return device
        if len(others) == 1:
            # Some lights advertise Node Identity instead of the network ID; with
            # one candidate in range, try it. The beacon check confirms it.
            return next(iter(others.values()))
        if not others:
            raise LinkError(
                "The light is not advertising over Bluetooth. It may be off or out of "
                "range, or connected to another app (amaran Desktop, or the amaran or "
                "Sidus Link app on a phone): it takes one connection at a time."
            )
        raise LinkError(f"{len(others)} Bluetooth Mesh lights in range and none matched this network")

    def _on_disconnect(self, _client: BleakClient) -> None:
        log.info("light disconnected")

    async def _drop(self) -> None:
        client, self._client = self._client, None
        for _, future in self._waiters:
            if not future.done():
                future.cancel()
        if client:
            try:
                await client.disconnect()
            except BleakError:
                pass

    def _arm_idle(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        self._idle_task = asyncio.get_running_loop().create_task(self._close_when_idle())

    async def _close_when_idle(self) -> None:
        await asyncio.sleep(self.idle_timeout)
        async with self._lock:
            await self._drop()

    # ── Wire ─────────────────────────────────────────────────────────────────

    async def _send_access(self, client: BleakClient, payloads: list[bytes]) -> None:
        sequences = self.store.reserve(len(payloads))
        for index, (seq, payload) in enumerate(zip(sequences, payloads)):
            if index:
                await asyncio.sleep(PACKET_GAP)
            await self._write(client, build_mesh_proxy_pdu(
                net_key=self._net_key,
                app_key=self._app_key,
                src=self.light.source_address,
                dst=self.light.node_address,
                seq=seq,
                iv_index=self.store.iv_index,
                sidus_payload=payload,
            ))

    async def _write(self, client: BleakClient, pdu: bytes) -> None:
        """Write a proxy PDU, splitting it with proxy SAR if the MTU is small."""

        room = max(20, client.mtu_size - 3)
        if len(pdu) <= room:
            await client.write_gatt_char(PROXY_IN, pdu, response=False)
            return
        kind, body, step = pdu[0] & 0x3F, pdu[1:], room - 1
        chunks = [body[i : i + step] for i in range(0, len(body), step)]
        for index, chunk in enumerate(chunks):
            sar = 0b01 if index == 0 else 0b11 if index == len(chunks) - 1 else 0b10
            await client.write_gatt_char(PROXY_IN, bytes([sar << 6 | kind]) + chunk, response=False)

    def _on_notify(self, _char: object, data: bytearray) -> None:
        sar, kind = data[0] >> 6, data[0] & 0x3F
        if sar == 0b01:
            self._rx = bytearray([kind]) + data[1:]
            return
        if sar == 0b10:
            self._rx += data[1:]
            return
        if sar == 0b11:
            pdu = bytes(self._rx + data[1:])
            self._rx.clear()
        else:
            pdu = bytes(data)

        if kind == PROXY_TYPE_BEACON:
            beacon = decode_secure_network_beacon(net_key=self._net_key, proxy_pdu=pdu)
            if beacon:
                self.store.set_iv_index(beacon.iv_index)
                self._beacon.set()
            return
        if kind == PROXY_TYPE_CONFIG:
            self._filter_status.set()
            return
        if kind != PROXY_TYPE_NETWORK:
            return
        message = decode_mesh_proxy_access(
            net_key=self._net_key,
            app_key=self._app_key,
            iv_index=self.store.iv_index,
            proxy_pdu=pdu,
        )
        if message is None or message.source_address != self.light.node_address:
            return
        log.debug("reply %s", message.sidus_payload.hex(" ") if message.sidus_payload else "-")
        for pick, future in self._waiters:
            if not future.done():
                value = pick(message)
                if value is not None:
                    future.set_result(value)
