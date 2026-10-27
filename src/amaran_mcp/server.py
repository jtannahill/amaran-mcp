"""MCP tools for one amaran light, including music mode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import colorsys
import functools
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .backends import close_all, current_light as light
from .config import ConfigError, load_scenes
from .desktop import DesktopError
from .link import LinkError
from .music import control
from .music.moods import MOODS


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await close_all()


mcp = MCPServer(
    "amaran",
    instructions=(
        "Controls the user's amaran light from this Mac, through amaran Desktop when "
        "it is running, otherwise over Bluetooth directly. "
        "Brightness is a percentage. If brightness is left out, the light keeps "
        "its current brightness. Commands wake the light if it is off. "
        "Music mode (start_music_mode) makes the light follow what this Mac is "
        "playing; other light commands stop it, except set_brightness, which "
        "moves its brightness range."
    ),
    lifespan=lifespan,
)

def tool(**options: Any):
    """Register a tool whose expected failures reach the model as readable errors."""

    def register(fn):
        @functools.wraps(fn)
        async def run(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except (ConfigError, DesktopError, LinkError, ValueError) as err:
                raise ToolError(str(err)) from err

        return mcp.tool(**options)(run)

    return register


async def take_over() -> None:
    """A direct light command ends music mode first, so the two never fight."""

    if control.status():
        await asyncio.to_thread(control.stop)


Percent = Annotated[float, Field(ge=0, le=100, description="Brightness, 0-100 %")]
CHANGES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


@tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_light() -> dict[str, Any]:
    """Read the light's power, mode, brightness, colour, effect and battery.

    Also lists what this model supports (colour temperature range, effects)."""

    music = control.status() or {"running": False}
    lamp = light()
    if music["running"] and lamp.backend == "bluetooth":
        # Music mode holds the light's one Bluetooth connection.
        return {"light": lamp.name, "backend": lamp.backend, "music_mode": music}
    status = await lamp.read_status()
    status["music_mode"] = music
    status["supports"] = {
        "kelvin_range": [lamp.caps.kelvin_min, lamp.caps.kelvin_max] if lamp.caps.cct else None,
        "tint": lamp.caps.tint,
        "color": lamp.caps.hsi,
        "effects": lamp.caps.effects,
    }
    status["scenes"] = sorted(load_scenes())
    return status


@tool(annotations=CHANGES)
async def set_power(on: bool) -> str:
    """Turn the light on or off. On restores what it was doing before."""

    await take_over()
    await light().set_power(on)
    return "on" if on else "off"


@tool(annotations=CHANGES)
async def set_brightness(brightness: Percent) -> str:
    """Change brightness only, keeping the current white, colour or effect.

    While music mode runs, this moves music mode's brightness range instead."""

    music = control.status()
    if music:
        offset = control.set_brightness(brightness)
        low, high = control.brightness_range(music.get("mood") or "ethereal", offset)
        return f"music mode brightness range moved to {low:g}..{high:g}%"
    await light().set_brightness(brightness)
    return f"brightness {brightness:g}%"


@tool(annotations=CHANGES)
async def set_white(
    kelvin: Annotated[int, Field(description="Colour temperature, e.g. 3200 tungsten, 5600 daylight")],
    brightness: Percent | None = None,
    tint: Annotated[int, Field(ge=-10, le=10, description="Green/magenta offset, 0 neutral")] = 0,
) -> str:
    """Set white light by colour temperature. get_light shows this model's range."""

    await take_over()
    lamp = light()
    await lamp.set_white(kelvin, brightness, tint)
    return f"white {kelvin}K at {lamp.state.brightness}%" + (f", tint {tint:+d}" if tint else "")


@tool(annotations=CHANGES)
async def set_color(
    hue: Annotated[float | None, Field(ge=0, le=360, description="Hue in degrees: 0 red, 120 green, 240 blue")] = None,
    saturation: Annotated[float, Field(ge=0, le=100, description="Saturation, 0-100 %")] = 100,
    hex: Annotated[str | None, Field(description="Colour as #RRGGBB, instead of hue and saturation")] = None,
    brightness: Percent | None = None,
) -> str:
    """Set a coloured light, by hue and saturation or by a #RRGGBB hex colour."""

    if hex is not None:
        value = hex.lstrip("#")
        if len(value) != 6:
            raise ValueError("hex must look like #FF8800")
        r, g, b = (int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))
        h, s, _v = colorsys.rgb_to_hsv(r, g, b)
        hue, saturation = h * 360, s * 100
    if hue is None:
        raise ValueError("Give a hue or a hex colour")
    await take_over()
    lamp = light()
    await lamp.set_color(hue, saturation, brightness)
    return f"colour hue {lamp.state.hue}, saturation {lamp.state.saturation}% at {lamp.state.brightness}%"


@tool(annotations=CHANGES)
async def set_effect(
    effect: Annotated[str, Field(description="An effect from get_light's supports.effects, e.g. fire, tv, lightning")],
    brightness: Percent | None = None,
) -> str:
    """Start one of the light's built-in effects."""

    await take_over()
    lamp = light()
    await lamp.set_effect(effect, brightness)
    return f"effect {effect} at {lamp.state.brightness}%"


@tool(annotations=CHANGES)
async def stop_effect() -> str:
    """Stop the running effect and return to the last steady white or colour."""

    await take_over()
    await light().stop_effect()
    return "effect stopped"


@tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def save_scene(name: str) -> dict[str, Any]:
    """Save the light's current look under a name, replacing any scene with that name."""

    lamp = light()
    await lamp.read_status()
    return lamp.save_scene(name)


@tool(annotations=CHANGES)
async def recall_scene(name: str) -> dict[str, Any]:
    """Put the light back into a saved scene. get_light lists the saved scenes."""

    await take_over()
    return await light().recall_scene(name)


@tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def start_music_mode(
    mood: Annotated[str, Field(description="ethereal (default), deep or dawn")] = "ethereal",
    record: Annotated[bool, Field(description="save the session on this Mac for training (default on)")] = True,
) -> dict[str, Any]:
    """Start music mode: the light follows what this Mac is playing (including AirPlay from an iPhone) with slow, ethereal colour. Moods: ethereal, deep, dawn."""

    if mood not in MOODS:
        raise ValueError(f"Unknown mood {mood!r}. Moods: {', '.join(MOODS)}")
    # The light takes one Bluetooth connection; music mode needs it, not this server.
    await close_all()
    return await asyncio.to_thread(control.start, mood, record=record)


@tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def stop_music_mode() -> str:
    """Stop music mode and return the light to how it was."""

    result = await asyncio.to_thread(control.stop)
    if result == "settled":
        return "music mode stopped; the light is back to how it was"
    if result == "killed":
        return ("music mode stopped, but it did not finish fading back in time, "
                "so the light may still show its last colour")
    return "music mode was not running"


@tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def mark_music_moment(
    kind: Annotated[str, Field(description="breakdown, lift, vocal, build, drop or other")],
    note: str | None = None,
) -> str:
    """Flag this moment in the music mode recording ("the chords just lifted"), for training. Call it the moment the user says so."""

    try:
        control.mark(kind, note)
    except RuntimeError as err:
        return str(err)
    return f"marked {kind}"
