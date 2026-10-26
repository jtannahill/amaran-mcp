"""amaran-mcp: control an amaran light over Bluetooth Mesh, as an MCP server or CLI.

  amaran-mcp import [--db PATH] [--name NAME]   copy the light and keys from amaran Desktop
  amaran-mcp import --account EMAIL [--country US]   ...or from the amaran account
  amaran-mcp serve                              run the MCP server on stdio
  amaran-mcp status | on | off
  amaran-mcp brightness PCT
  amaran-mcp white KELVIN [PCT] [TINT]
  amaran-mcp color HUE [SAT] [PCT]
  amaran-mcp effect NAME [PCT] | effect off
  amaran-mcp music-probe [SECONDS]              print the level of what this Mac is playing
  amaran-mcp music [--mood NAME]                run music mode in the foreground
  amaran-mcp music-stop                         stop a background music mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys


def _import(args: argparse.Namespace) -> None:
    from .config import LIGHT_PATH, find_desktop_db, import_light, list_cloud_fixtures, list_desktop_fixtures

    if args.account:
        import getpass
        password = getpass.getpass(f"amaran password for {args.account}: ")
        fixtures = list_cloud_fixtures(args.account, password, args.country)
    else:
        fixtures = list_desktop_fixtures(args.db or find_desktop_db())
    if args.name:
        fixtures = [f for f in fixtures if f["name"].lower() == args.name.lower()]
        if not fixtures:
            raise SystemExit(f"No light named {args.name!r}")
    if len(fixtures) > 1:
        names = ", ".join(repr(f["name"]) for f in fixtures)
        raise SystemExit(f"Found {len(fixtures)} lights ({names}); pick one with --name")
    light = import_light(fixtures[0])
    print(f"Imported {light.name} ({light.model}, mesh address {light.node_address}) to {LIGHT_PATH}")


def _music_probe(seconds: float) -> None:
    """Print one line per second: level of the system audio the helper hears."""

    from array import array
    import math
    import subprocess

    from .music.capture import CaptureError, helper_path

    helper = subprocess.Popen([str(helper_path())], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        header = helper.stdout.readline().decode().split()
        if len(header) != 2 or header[0] != "RATE":
            raise CaptureError(helper.stderr.read().decode().strip() or "the audio helper did not start")
        rate = int(header[1])
        print(f"listening at {rate} Hz")
        for second in range(int(seconds)):
            samples = array("f")
            samples.frombytes(helper.stdout.read(rate * 4))
            peak = max((abs(v) for v in samples), default=0.0)
            rms = math.sqrt(sum(v * v for v in samples) / max(len(samples), 1))
            level = 20 * math.log10(rms) if rms > 0 else float("-inf")
            print(f"{second + 1:3d}s  rms {level:7.1f} dBFS  peak {peak:.3f}")
    finally:
        helper.terminate()
        helper.wait(timeout=5)


async def _run(args: argparse.Namespace) -> None:
    from .backends import current_light

    lamp = current_light()
    rest = [float(v) for v in args.values] if args.command != "effect" else []
    try:
        match args.command:
            case "status":
                print(json.dumps(await lamp.read_status(), indent=2))
                return
            case "on" | "off":
                await lamp.set_power(args.command == "on")
            case "brightness":
                await lamp.set_brightness(*rest)
            case "white":
                kelvin, *more = rest
                await lamp.set_white(int(kelvin), *more[:1], *(int(t) for t in more[1:2]))
            case "color":
                await lamp.set_color(*rest)
            case "effect":
                name, *more = args.values
                if name == "off":
                    await lamp.stop_effect()
                else:
                    await lamp.set_effect(name, *(float(v) for v in more))
        print("ok", json.dumps({k: v for k, v in vars(lamp.state).items() if v is not None}))
    finally:
        await lamp.close()


def main() -> None:
    # stdout carries the MCP protocol when serving, so all logging goes to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="amaran-mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["import", "serve", "status", "on", "off",
                                            "brightness", "white", "color", "effect",
                                            "music-probe", "music", "music-stop"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--db", type=Path, help="amaran Desktop amaran.db (import)")
    parser.add_argument("--account", help="amaran account email or phone (import from the account)")
    parser.add_argument("--country", default="US", help="account country code (import --account)")
    parser.add_argument("--name", help="which light to import, if there are several")
    parser.add_argument("--mood", default="ethereal", help="music mode mood: ethereal, deep or dawn")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from .config import ConfigError
    from .desktop import DesktopError
    from .link import LinkError
    from .music.capture import CaptureError

    try:
        if args.command == "import":
            _import(args)
        elif args.command == "music":
            from .music.runner import RunnerConfig, run
            raise SystemExit(asyncio.run(run(RunnerConfig(mood=args.mood))))
        elif args.command == "music-stop":
            from .music import control
            print("stopped" if control.stop() else "music mode was not running")
        elif args.command == "music-probe":
            _music_probe(float(args.values[0]) if args.values else 10)
        elif args.command == "serve":
            from .server import mcp
            mcp.run()
        else:
            asyncio.run(_run(args))
    except (CaptureError, ConfigError, DesktopError, LinkError, ValueError) as err:
        raise SystemExit(f"error: {err}") from None
