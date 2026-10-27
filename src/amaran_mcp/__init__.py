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
  amaran-mcp music [--mood NAME] [--log-features PATH]   run music mode in the foreground
  amaran-mcp music-stop                         stop a background music mode
  amaran-mcp listen                             record listening sessions (no light); run at login
  amaran-mcp mark KIND [NOTE]                   flag this moment: breakdown, lift, vocal, build, drop
  amaran-mcp install-listener | uninstall-listener   the launch agent that runs `listen`
  amaran-mcp music-record SECONDS NAME          record a tuning clip of what this Mac plays
  amaran-mcp music-replay PATH                  replay a clip through the analysis
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

    import os
    import threading

    read_fd, write_fd = os.pipe()
    helper = subprocess.Popen([str(helper_path()), "--classify-fd", str(write_fd)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, pass_fds=(write_fd,))
    os.close(write_fd)
    latest: dict = {}

    def read_classes() -> None:
        for line in os.fdopen(read_fd, "r"):
            try:
                latest.update(json.loads(line))
            except ValueError:
                pass

    threading.Thread(target=read_classes, daemon=True).start()
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
            top = sorted(((v, k) for k, v in latest.items() if k != "t"), reverse=True)[:3]
            classes = "  ".join(f"{k} {v:.2f}" for v, k in top)
            print(f"{second + 1:3d}s  rms {level:7.1f} dBFS  peak {peak:.3f}  {classes}")
    finally:
        helper.terminate()
        helper.wait(timeout=5)


LAUNCH_AGENT = "com.jtannahill.amaran-listen"
SOURCE = "git+https://github.com/jtannahill/amaran-mcp"


def _listener_command(code: Path, which=None) -> list[str]:
    """What the login item runs: a clone runs its own code, a `uv tool install` its
    installed command; anything else (uvx, a Claude Code plugin) lives in a cache
    that can be cleared or replaced, so it runs the published repo through uvx."""

    import shutil

    which = which or shutil.which
    project = code.parents[2]
    if (project / ".git").exists() and (project / "pyproject.toml").exists() \
            and Path.home() / ".claude" not in project.parents:
        return [which("uv") or "uv", "--directory", str(project), "run", "amaran-mcp", "listen"]
    if installed := which("amaran-mcp"):
        return [installed, "listen"]
    return [which("uvx") or "uvx", "--from", SOURCE, "amaran-mcp", "listen"]


def _launch_agent(install: bool) -> None:
    """Start (or remove) the background listener at login, as this user."""

    import os
    import plistlib
    import subprocess

    plist = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT}.plist"
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCH_AGENT}"], capture_output=True)
    if not install:
        plist.unlink(missing_ok=True)
        print("listener removed")
        return
    from .config import CONFIG_DIR

    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_bytes(plistlib.dumps({
        "Label": LAUNCH_AGENT,
        "ProgramArguments": _listener_command(Path(__file__).resolve()),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(CONFIG_DIR / "listener.log"),
        "StandardErrorPath": str(CONFIG_DIR / "listener.log"),
        "ProcessType": "Background",
    }))
    subprocess.run(["launchctl", "bootstrap", domain, str(plist)], check=True)
    print(f"listener installed: {plist}")


def _music_replay(path: Path) -> None:
    """Print a clip's analysis as a compact timeline."""

    from .music import clips

    def short(row: dict, key: str, fmt: str = "{:.2f}") -> str:
        value = row
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return "-" if value is None else (fmt.format(value) if isinstance(value, float) else str(value))

    columns = [("t", "t", "{:.0f}"), ("section", "section", "{}"), ("energy", "energy", "{:.2f}"),
               ("bpm", "beat.bpm", "{:.0f}"), ("chord", "harmony.chord", "{}"),
               ("balance", "harmony.balance", "{:+.2f}"), ("vocal", "timbre.vocal", "{:.2f}"),
               ("tension", "timbre.tension", "{:.2f}"), ("open", "timbre.openness", "{:.2f}")]
    print("  ".join(f"{name:>8}" for name, _, _ in columns) + "  events")
    for row in clips.replay(path):
        events = [e for e in ("drop", "release", "bloom") if row.get(e)]
        print("  ".join(f"{short(row, key, fmt):>8}" for _, key, fmt in columns) + "  " + " ".join(events))


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
                                            "music-probe", "music", "music-stop",
                                            "music-record", "music-replay", "listen",
                                            "install-listener", "uninstall-listener", "mark"])
    parser.add_argument("values", nargs="*")
    parser.add_argument("--db", type=Path, help="amaran Desktop amaran.db (import)")
    parser.add_argument("--account", help="amaran account email or phone (import from the account)")
    parser.add_argument("--country", default="US", help="account country code (import --account)")
    parser.add_argument("--name", help="which light to import, if there are several")
    parser.add_argument("--mood", default="ethereal", help="music mode mood: ethereal, deep or dawn")
    parser.add_argument("--log-features", type=Path, help="music: write one JSON line per tick here")
    parser.add_argument("--no-record", action="store_true", help="music: do not save the session")
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
            raise SystemExit(asyncio.run(run(RunnerConfig(mood=args.mood, log_features=args.log_features,
                                                          record=not args.no_record))))
        elif args.command == "music-record":
            from .music import clips
            seconds, name = float(args.values[0]), (args.values[1] if len(args.values) > 1 else "clip")
            print(asyncio.run(clips.record(seconds, name)))
        elif args.command == "music-replay":
            _music_replay(Path(args.values[0]))
        elif args.command == "mark":
            from .music import control
            kind = {"climax": "drop", "vocals": "vocal"}.get(args.values[0], args.values[0]) if args.values else "other"
            try:
                control.mark(kind, " ".join(args.values[1:]) or None)
                print(f"marked {kind}")
            except RuntimeError as err:
                print(err)
        elif args.command == "listen":
            from .music.listener import listen
            raise SystemExit(asyncio.run(listen()))
        elif args.command in ("install-listener", "uninstall-listener"):
            _launch_agent(args.command == "install-listener")
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
