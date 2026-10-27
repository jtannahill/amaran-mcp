"""Pick how to reach the light: through amaran Desktop, or Bluetooth directly.

AMARAN_BACKEND=auto (the default) checks whether amaran Desktop is running, at
most every few seconds: when it is, it holds the light's one Bluetooth
connection, so commands go through it; otherwise the Mac connects to the light
itself. "desktop" or "bluetooth" forces one.
"""

from __future__ import annotations

import os
import time

from .config import ConfigError, LIGHT_PATH, load_light
from .desktop import DesktopLight, desktop_running
from .light import AmaranLight, LightBase

_lights: dict[str, LightBase] = {}
PROBE_SECONDS = 5.0  # music mode asks many times a second; Desktop is not opened that often
_probe: tuple[float, bool] | None = None  # (when, whether Desktop was running)


def _desktop_running() -> bool:
    global _probe
    now = time.monotonic()
    if _probe is None or now - _probe[0] >= PROBE_SECONDS:
        _probe = (now, desktop_running())
    return _probe[1]


def choose() -> str:
    backend = os.environ.get("AMARAN_BACKEND", "auto").lower()
    if backend not in ("auto", "desktop", "bluetooth"):
        raise ConfigError("AMARAN_BACKEND must be auto, desktop or bluetooth")
    if backend == "auto":
        return "desktop" if _desktop_running() else "bluetooth"
    return backend


def current_light() -> LightBase:
    backend = choose()
    if backend not in _lights:
        if backend == "desktop":
            # Desktop knows its lights; the imported config only picks one of several.
            _lights[backend] = DesktopLight(load_light() if LIGHT_PATH.exists() else None)
        else:
            _lights[backend] = AmaranLight(load_light())
    return _lights[backend]


async def close_all() -> None:
    for light in _lights.values():
        await light.close()
    _lights.clear()
