# amaran-mcp

An MCP server (and small CLI) that controls one amaran light from a Mac. It
goes through amaran Desktop's local API when Desktop is running, and talks
Bluetooth Mesh to the light directly when it is not. It was built for the PT2c
pixel tube; any Sidus Mesh light that amaran Desktop can pair with should work.

Tools: `get_light`, `set_power`, `set_brightness`, `set_white`, `set_color`,
`set_effect`, `stop_effect`, `save_scene`, `recall_scene`.

Not affiliated with or endorsed by Aputure or amaran. Tested on a PT2c with
macOS; status, brightness, white, tint, colour, the fire effect and power were
each confirmed by the light's own status reports.

## Backends

`AMARAN_BACKEND=auto` (the default) picks one on every call:

- **amaran Desktop** (when it is running): commands go over its local OpenAPI
  WebSocket at `ws://127.0.0.1:33782/ws`. Fast (tens of milliseconds), and it
  includes Desktop's full effect list, pixel effects too (rainbow, pixel fire,
  pixel chases, colour cycle). Status comes from Desktop's own record of the
  light, and battery level is not available.
- **Bluetooth** (when Desktop is not running): direct Bluetooth Mesh, below.
  Each call takes a second or more, but no app needs to be open, and battery
  level and the light's own status reports are available.

Set `AMARAN_BACKEND=desktop` or `bluetooth` to force one. The Desktop API wants
a token in each request: set `AMARAN_OPENAPI_KEY` to the key from an approved
amaran OpenAPI application (hex or base64). Without one, the server signs with
a random key, as the Bitfocus Companion module does, which current Desktop
builds accept. How an approved key is meant to be used is a guess until one is
issued.

## How the Bluetooth backend works

The light acts as a Bluetooth Mesh proxy. The server connects to it when a
tool runs, sends encrypted mesh messages (Sidus opcode `0x26`), reads the
light's status and battery replies, then disconnects after 30 seconds idle so
the amaran apps can reach the light again.

The mesh keys come from amaran Desktop's local database. The server sends from
its own mesh address and stores its message counter, which only goes up,
in `~/.config/amaran-mcp/state.json`. Lights ignore any message whose counter
is not higher than the last one they saw from that address, so the counter
must never go back.

## Setup

1. Install amaran Desktop and pair the light in it.
2. Quit amaran Desktop fully (it keeps running after its window closes, and
   holds the light), then clone this repo and import the light:

   ```sh
   git clone https://github.com/jtannahill/amaran-mcp && cd amaran-mcp
   uv run amaran-mcp import
   uv run amaran-mcp status
   ```

   If the light was paired with the amaran phone app, the keys are already
   synced to the amaran account and Desktop is not needed:
   `uv run amaran-mcp import --account you@example.com [--country US]`
   (the password is asked for and never stored). This path is untested.

   `import` writes `~/.config/amaran-mcp/light.json` (mode 600; it holds the
   mesh keys). The first Bluetooth use asks for permission for the app running
   the command (Terminal, iTerm, Claude).

3. Register with Claude Code:

   ```sh
   claude mcp add -s user amaran -- uv --directory /path/to/amaran-mcp run amaran-mcp serve
   ```

## CLI

```sh
uv run amaran-mcp on | off | status
uv run amaran-mcp brightness 40
uv run amaran-mcp white 3200 60 [tint]
uv run amaran-mcp color 30 100 50       # hue, saturation, brightness
uv run amaran-mcp effect fire 70 | effect off
```

## Limits

- One light. Custom per-pixel patterns are not supported; the built-in effects
  are, including the pixel effects when going through amaran Desktop.
- The light accepts one Bluetooth connection at a time. The auto backend avoids
  fighting amaran Desktop over it, but a phone app holding it still blocks the
  Bluetooth backend.
- Through amaran Desktop, the active mode (white, colour, effect) is the last
  one this server set; a change made in the amaran apps shows only its power
  and brightness.
- Re-pairing the light in amaran Desktop changes the keys: run `import` again.

## Credits

`src/amaran_mcp/sidus/` is vendored unmodified from
[neyako/amaran-hacs](https://github.com/neyako/amaran-hacs) (MIT), which builds
on [wesbos/amaran-BLE-control](https://github.com/wesbos/amaran-BLE-control).
The amaran Desktop message and token format follow
[bitfocus/companion-module-amaran-desktop](https://github.com/bitfocus/companion-module-amaran-desktop) (MIT)
and the [amaran OpenAPI docs](https://tools.sidus.link/openapi/docs/protocol).
