"""MCP tools for one amaran light over Bluetooth."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import colorsys
import functools
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .config import ConfigError, load_light, load_scenes
from .light import AmaranLight
from .link import LinkError

_light: AmaranLight | None = None


def light() -> AmaranLight:
    global _light
    if _light is None:
        _light = AmaranLight(load_light())
    return _light


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if _light:
            await _light.close()


mcp = MCPServer(
    "amaran",
    instructions=(
        "Controls the user's amaran light over Bluetooth from this Mac. Each call "
        "takes about a second, longer if the light has to be found first. "
        "Brightness is a percentage. If brightness is left out, the light keeps "
        "its current brightness. Commands wake the light if it is off."
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
            except (ConfigError, LinkError, ValueError) as err:
                raise ToolError(str(err)) from err

        return mcp.tool(**options)(run)

    return register


Percent = Annotated[float, Field(ge=0, le=100, description="Brightness, 0-100 %")]
CHANGES = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


@tool(annotations=ToolAnnotations(readOnlyHint=True))
async def get_light() -> dict[str, Any]:
    """Read the light's power, mode, brightness, colour, effect and battery.

    Also lists what this model supports (colour temperature range, effects)."""

    lamp = light()
    status = await lamp.read_status()
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

    await light().set_power(on)
    return "on" if on else "off"


@tool(annotations=CHANGES)
async def set_brightness(brightness: Percent) -> str:
    """Change brightness only, keeping the current white, colour or effect."""

    await light().set_brightness(brightness)
    return f"brightness {brightness:g}%"


@tool(annotations=CHANGES)
async def set_white(
    kelvin: Annotated[int, Field(description="Colour temperature, e.g. 3200 tungsten, 5600 daylight")],
    brightness: Percent | None = None,
    tint: Annotated[int, Field(ge=-10, le=10, description="Green/magenta offset, 0 neutral")] = 0,
) -> str:
    """Set white light by colour temperature. get_light shows this model's range."""

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
    lamp = light()
    await lamp.set_color(hue, saturation, brightness)
    return f"colour hue {lamp.state.hue}, saturation {lamp.state.saturation}% at {lamp.state.brightness}%"


@tool(annotations=CHANGES)
async def set_effect(
    effect: Annotated[str, Field(description="An effect from get_light's supports.effects, e.g. fire, tv, lightning")],
    brightness: Percent | None = None,
) -> str:
    """Start one of the light's built-in effects."""

    lamp = light()
    await lamp.set_effect(effect, brightness)
    return f"effect {effect} at {lamp.state.brightness}%"


@tool(annotations=CHANGES)
async def stop_effect() -> str:
    """Stop the running effect and return to the last steady white or colour."""

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

    return await light().recall_scene(name)
