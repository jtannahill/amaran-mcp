# amaran-mcp

An MCP server (and small CLI) that controls one amaran light from this Mac over
Bluetooth Mesh, with no amaran app running. It was built for the PT2c pixel
tube, and any Sidus Mesh light that amaran Desktop can pair with should work.

Tools: `get_light`, `set_power`, `set_brightness`, `set_white`, `set_color`,
`set_effect`, `stop_effect`, `save_scene`, `recall_scene`.

Not affiliated with or endorsed by Aputure or amaran. Tested on a PT2c with
macOS; status, brightness, white, tint, colour, the fire effect and power were
each confirmed by the light's own status reports.

## How it works

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

- One light. Per-pixel effects are not decoded; the built-in effects are.
- The light accepts one Bluetooth connection at a time. If the amaran app holds
  it, commands fail until the app lets go.
- Re-pairing the light in amaran Desktop changes the keys: run `import` again.

## Credits

`src/amaran_mcp/sidus/` is vendored unmodified from
[neyako/amaran-hacs](https://github.com/neyako/amaran-hacs) (MIT), which builds
on [wesbos/amaran-BLE-control](https://github.com/wesbos/amaran-BLE-control).
