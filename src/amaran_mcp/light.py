"""What the light can do, in percent, kelvin and degrees.

Remembers the last state it set or read, so a colour change keeps the current
brightness and a command to a light that is off wakes it first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any

from .config import Capabilities, Light, StateStore, capabilities, load_scenes, save_scenes
from .link import MeshLink
from .sidus import protocol
from .sidus.effects import EFFECTS, effect_values

log = logging.getLogger(__name__)

# Desktop flag ("cop_car") -> amaran-hacs preset name ("Cop car").
EFFECT_NAMES = {flag: name for name, (flag, *_rest) in EFFECTS.items()}

DEFAULT_BRIGHTNESS = 50


@dataclass
class LightState:
    power: bool | None = None
    mode: str | None = None  # "white", "color" or "effect"
    brightness: int | None = None  # percent
    kelvin: int | None = None
    tint: int | None = None
    hue: int | None = None
    saturation: int | None = None
    effect: str | None = None  # Desktop flag, e.g. "fire"


def _intensity(percent: float) -> int:
    return round(max(0.0, min(100.0, percent)) * 10)


def _percent(ha_brightness: int | None) -> int | None:
    return None if ha_brightness is None else round(ha_brightness / 255 * 100)


def _effect_payloads(name: str, percent: float = 0) -> list[bytes]:
    """Sidus payloads for an amaran-hacs preset name, or "off" to stop."""

    values = effect_values(name, _intensity(percent))
    return [protocol._finalize_sidus(bytearray(v.to_bytes(10, "little"))) for v in values]


def _cct_tint(payload: bytes) -> int:
    """Green/magenta offset from a CCT report; protocol.py does not decode it."""

    low = int.from_bytes(payload[:8], "little")
    return max(-10, min(10, ((low >> 45) & 0x7F) - 10))


class AmaranLight:
    def __init__(self, light: Light, store: StateStore | None = None) -> None:
        self.light = light
        self.caps: Capabilities = capabilities(light)
        self.link = MeshLink(light, store or StateStore())
        self.state = LightState()

    # ── Reading ──────────────────────────────────────────────────────────────

    async def read_status(self) -> dict[str, Any]:
        """Ask the light for its state and battery; fall back to what we last set."""

        report = await self.link.query(
            protocol.status_request_payload(),
            lambda m: m if m.sidus_status else None,
        )
        confirmed = report is not None
        if report:
            self._apply_report(report.sidus_status, report.sidus_payload)
        power = await self.link.query(
            protocol.power_status_request_payload(),
            lambda m: m.sidus_power_info,
        )
        # Only the fields for the current mode; the others are kept for stop_effect.
        hidden = {"white": {"hue", "saturation", "effect"},
                  "color": {"kelvin", "tint", "effect"},
                  "effect": {"kelvin", "tint", "hue", "saturation"}}.get(self.state.mode or "", set())
        result: dict[str, Any] = {
            "light": self.light.name,
            "model": self.light.model,
            "confirmed_by_light": confirmed,
            **{k: v for k, v in asdict(self.state).items() if v is not None and k not in hidden},
        }
        if power:
            result["battery_percent"] = power.battery_percentage
            result["plugged_in"] = power.power_supply_mode == "ac"
            if power.power_supply_mode == "battery" and power.battery_time_minutes:
                result["battery_minutes_left"] = power.battery_time_minutes
        return result

    def _apply_report(self, status: protocol.SidusStatus, payload: bytes | None) -> None:
        s = self.state
        if status.color_mode == "effect_off":
            s.effect = None
            return
        if status.power is not None:
            s.power = status.power
        if status.brightness is not None:
            s.brightness = _percent(status.brightness)
        if status.effect:
            s.mode = "effect"
            s.effect = next((f for f, n in EFFECT_NAMES.items() if n == status.effect), None)
        elif status.color_mode == "color_temp":
            s.mode, s.effect = "white", None
            s.kelvin = status.color_temp_kelvin
            if payload:
                s.tint = _cct_tint(payload)
        elif status.color_mode == "hs" and status.hs_color:
            s.mode, s.effect = "color", None
            s.hue, s.saturation = (round(v) for v in status.hs_color)

    # ── Writing ──────────────────────────────────────────────────────────────

    async def _send(self, payloads: list[bytes]) -> None:
        if self.state.power is not True:
            payloads = [protocol.power_payload(True), *payloads]
        await self.link.send(payloads)
        self.state.power = True

    async def set_power(self, on: bool) -> None:
        await self.link.send([protocol.power_payload(on)])
        self.state.power = on

    async def set_brightness(self, percent: float) -> None:
        if self.state.mode == "effect" and self.state.effect:
            # An effect carries its own intensity; brightness re-applies it.
            await self._send(_effect_payloads(EFFECT_NAMES[self.state.effect], percent))
        else:
            await self._send([protocol.brightness_payload(_intensity(percent))])
        self.state.brightness = round(percent)

    async def set_white(self, kelvin: int, brightness: float | None = None, tint: int = 0) -> None:
        if not self.caps.cct:
            raise ValueError(f"{self.light.model} has no white (colour temperature) mode")
        if not self.caps.kelvin_min <= kelvin <= self.caps.kelvin_max:
            raise ValueError(
                f"{self.light.model} does colour temperature from {self.caps.kelvin_min}K "
                f"to {self.caps.kelvin_max}K, not {kelvin}K"
            )
        if tint and not self.caps.tint:
            raise ValueError(f"{self.light.model} has no green/magenta tint control")
        percent = self._brightness(brightness)
        await self._send([protocol.cct_payload(intensity=_intensity(percent), kelvin=kelvin, gm=tint)])
        self.state = LightState(True, "white", round(percent), kelvin, tint)

    async def set_color(self, hue: float, saturation: float, brightness: float | None = None) -> None:
        if not self.caps.hsi:
            raise ValueError(f"{self.light.model} has no colour mode")
        percent = self._brightness(brightness)
        hue, saturation = round(hue) % 360, round(max(0.0, min(100.0, saturation)))
        await self._send([protocol.hsi_payload(hue=hue, saturation=saturation, intensity=_intensity(percent))])
        self.state = LightState(True, "color", round(percent), hue=hue, saturation=saturation)

    async def set_effect(self, effect: str, brightness: float | None = None) -> None:
        if effect not in self.caps.effects:
            raise ValueError(
                f"{self.light.model} effects are: {', '.join(self.caps.effects) or 'none'}"
            )
        percent = self._brightness(brightness)
        await self._send(_effect_payloads(EFFECT_NAMES[effect], percent))
        self.state.mode, self.state.effect, self.state.brightness = "effect", effect, round(percent)

    async def stop_effect(self) -> None:
        """Stop the effect and go back to the last steady white or colour."""

        previous = self.state
        if previous.power is not False:
            await self.link.send(_effect_payloads("off"))
        self.state = LightState(power=previous.power, brightness=previous.brightness)
        if previous.kelvin is not None:
            await self.set_white(previous.kelvin, previous.brightness, previous.tint or 0)
        elif previous.hue is not None:
            await self.set_color(previous.hue, previous.saturation or 100, previous.brightness)

    def _brightness(self, requested: float | None) -> float:
        if requested is not None:
            return max(0.0, min(100.0, requested))
        return self.state.brightness if self.state.brightness is not None else DEFAULT_BRIGHTNESS

    # ── Scenes ───────────────────────────────────────────────────────────────

    def save_scene(self, name: str) -> dict[str, Any]:
        if self.state.mode is None:
            raise ValueError("Nothing to save yet: set the light or read its status first")
        scenes = load_scenes()
        scenes[name] = {k: v for k, v in asdict(self.state).items() if v is not None}
        save_scenes(scenes)
        return scenes[name]

    async def recall_scene(self, name: str) -> dict[str, Any]:
        scenes = load_scenes()
        if name not in scenes:
            raise ValueError(f"No scene called {name!r}. Saved scenes: {', '.join(scenes) or 'none'}")
        scene = LightState(**scenes[name])
        if scene.power is False:
            await self.set_power(False)
        elif scene.mode == "white" and scene.kelvin is not None:
            await self.set_white(scene.kelvin, scene.brightness, scene.tint or 0)
        elif scene.mode == "color" and scene.hue is not None:
            await self.set_color(scene.hue, scene.saturation or 100, scene.brightness)
        elif scene.mode == "effect" and scene.effect:
            await self.set_effect(scene.effect, scene.brightness)
        return scenes[name]

    async def close(self) -> None:
        await self.link.close()
