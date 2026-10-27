# amaran-mcp

Control an amaran light from a Mac, by asking Claude or from the terminal.
Two things it is for:

- **Music mode.** The light follows what the Mac plays: slow colour through
  the groove, pale when the kick drops out, and a bloom when a breakdown's
  chords lift and the vocal comes in.
- **Studio lighting from a prompt.** Describe the shot, a talking head or a
  corporate interview, or show Claude a photo of the set, and it sets the
  light to match.

<a href="https://jtannahill.github.io/amaran-mcp/">
<picture>
  <source media="(max-width: 600px)" srcset="docs/hero-narrow.svg">
  <img alt="A light tube changing colour through Craig Connelly and Nicholas Gunn's Miss You as music mode drives it: blues and violets through the drops, warmer through the breakdown, full at the final drop. Below it, the sections music mode heard." src="docs/hero-wide.svg">
</picture>
</a>

**[Play a breakdown](https://jtannahill.github.io/amaran-mcp/)**: press play on
a vocal trance track and watch the light do what music mode does, second by
second, in step with the music: Craig Connelly's "Miss You" next to what a
human expert marked while listening, nilsix and Clara Yates's "Through It
All", and Ferry Corsten's "Persiste" from A State of Trance 1300.

An MCP server for Claude Code (or any MCP client) and a small CLI, for the
[amaran PT2c](https://amarancreators.com/pages/amaran-pt2c) pixel tube and other Sidus Mesh lights. Not affiliated with or
endorsed by Aputure or amaran.

[Music mode](#music-mode) · [Studio lighting](#studio-lighting-from-a-prompt) ·
[Setup](#setup) · [Tools](#tools) ·
[What's next](#next) ·
[How it reaches the light](#how-it-reaches-the-light) ·
[Research](https://github.com/jtannahill/amaran-mcp/blob/main/docs/research.md)

## Music mode

```sh
uv run amaran-mcp music --mood ethereal     # or ask Claude: "start music mode"
```

The light follows whatever this Mac plays, AirPlay from an iPhone included.
It listens to the Mac's own audio output, not a microphone, so headphones are
fine. Everything runs on the Mac.

### Built for vocal trance

It was tuned on the euphoric, vocal end of trance, the sound of Armin van
Buuren's [A State of Trance](https://www.astateoftrance.com/),
[Find Your Harmony](https://andrewrayel.net/fyh-radioshow/) and
[Future Sound of Egypt](https://futuresoundofegypt.com/), and the genre's
habits are built in:

| The music | Music mode |
|---|---|
| 138 BPM, a bar every 1.74 s, phrases of 8 and 32 bars | Counts in bars at the tempo it hears; the model looks at about one 8-bar phrase at a time |
| A rolling offbeat bass that never stops | Hears the kick by its beat, not by the bass level |
| The long breakdown: pads, piano, a reverb-soaked vocal | Pale first, then the chords set the colour |
| Euphoria from minor-key progressions lifting onto major chords | The bloom is tied to that lift and nothing else |
| Snare rolls, risers and filter sweeps in the build | Tension from noise and density, so the colour deepens |
| The drop: the kick back under the supersaw anthem | Opens out to full on a real drop, not on a bass swell |
| A radio host talking over the mix | Apple's classifier tells speech from singing: talk never blooms |

Why this matters for the model, and what the training data lacks:
[Vocal trance at 138 BPM](https://github.com/jtannahill/amaran-mcp/blob/main/docs/research.md#vocal-trance-at-138-bpm).

### What listens

| Stage | What runs | Where |
|---|---|---|
| **Capture** | A Core Audio process tap on the system output | Swift helper, macOS 14.2+ |
| **Voices** | **Apple SoundAnalysis**, the built-in on-device sound classifier: singing, choir, speech, rapping, crowd | Apple's model, no network |
| **Tempo and chords** | Kick-band onset tracking for the beat; chroma matched to 24 chords for the harmony and the minor-to-major lift | numpy |
| **Sections** | **A machine-learned model** names intro, groove, breakdown, build or drop once a second | numpy, 13 ms a second |
| **Light** | A renderer turns sections, chords and voice into one slow colour | numpy |

Apple's classifier is how music mode tells a vocal from a DJ talking over the
mix: singing makes a bloom bigger, speech never blooms.

### The section model

Two models, blended 50/50:

- **Gradient-boosted trees** on a 16-second summary of what music mode hears:
  energy, bass, kick, tempo, harmony, tension, Apple's singing and speech
  readings, and 11 spectral readings.
- **A spectrogram CNN**, a small convolutional network that reads the last
  16 seconds as a log-mel spectrogram (64 bands, 10 frames a second).

Both learned from [SSME](https://osf.io/eydxk/): 402 Creative Commons EDM
tracks with drops, builds and breaks marked by experts. Training uses
scikit-learn and PyTorch; the shipped model is plain numpy arrays
(`src/amaran_mcp/music/sections.npz`), so music mode needs neither. Scored on
tracks each model had not seen:

| | Sections | Breakdowns | Builds | Drops |
|---|---|---|---|---|
| Rules | 0.36 | 0.02 | 0.00 | 0.10 |
| Trees | 0.48 | 0.05 | **0.08** | 0.19 |
| CNN | 0.44 | **0.11** | 0.07 | 0.15 |
| **Blend (shipped)** | **0.49** | 0.09 | 0.07 | **0.20** |

Sections is balanced accuracy, the share of seconds called right averaged
over the five sections; the others are F scores for finding a start within a
few seconds. Rules are the hand-written logic the model replaced.
Trance breakdowns are the weakest spot, which is why it also learns from your
own listening (below). The full research, every measurement and how to
retrain: [docs/research.md](https://github.com/jtannahill/amaran-mcp/blob/main/docs/research.md).

### What the light does

| Section | The light |
|---|---|
| Intro | Restrained, in the lower third of the brightness range |
| Groove | Drifts slowly through the mood's colours and breathes with the energy |
| Breakdown | Pale for its first bars, then each chord sets the colour: minor cool, major warm |
| Lift | The chords rise from minor to major: the light warms, swells and turns luminous, more with a vocal |
| Build | The drift stops, the colour deepens, brightness holds |
| Drop | Opens out to full |

Trance gets full colour: saturated blues, violets and orchids that only turn
paler at the start of a breakdown, and never grey. Hue turns at most 30
degrees a second, so it never flashes or pulses on the beat.

| Mood | Colours |
|---|---|
| `ethereal` | Deep blue, violet, teal, orchid |
| `deep` | Indigo, navy, magenta-violet |
| `dawn` | Lavender, peach, gold, rose |

Any other light command stops music mode, except brightness, which moves its
range. Needs Xcode's command line tools (the audio helper is compiled on first
run); macOS asks once for permission to record system audio.

### Teaching it your music

```sh
uv run amaran-mcp install-listener   # record listening sessions at login
uv run amaran-mcp mark lift          # or breakdown, vocal, build, drop
uv run amaran-mcp uninstall-listener
```

The listener records while music plays (FLAC, a feature log and your marks, in
`~/.config/amaran-mcp/recordings`), so a model can be trained on what you
actually listen to. It never touches the light, needs ffmpeg, stops at 30 GB,
and nothing leaves the Mac. Delete the folder to remove the recordings.

## Studio lighting from a prompt

Interviews and talking heads are mostly white light: the right colour
temperature, the right level, no cast on the face, and colour, if any, only
on the background. Say what the shot is and Claude sets the light with the
same tools:

> "Corporate interview, daylight from a window on the left, camera at 5600K.
> Soft key, a touch warm, not too bright."

> "Here's a photo of my set. Light my face for a webcam talk."

| What you need | How it is set |
|---|---|
| Match the camera's white balance | `set_white` in kelvin, such as 3200 tungsten or 5600 daylight (`get_light` shows the light's range) |
| Cancel a green or magenta cast from windows or office lights | `set_white` tint, -10 to +10 |
| Exposure for the face | `set_brightness` in percent |
| A background accent, when the look wants one | `set_color`, hue, saturation and brightness |
| The same look tomorrow, or per shot | `save_scene "interview A"`, then `recall_scene` |

Claude reads a photo of the set the way it reads any picture, so it can see
the window, the room's colour and where the light sits, and pick the white
balance and level from that. Today that is one light; a key, fill and back
light setup is planned as more amaran lights join.


You need a Mac, [uv](https://docs.astral.sh/uv/), and amaran Desktop with the
light paired in it.

1. Quit amaran Desktop fully (it keeps running after its window closes, and
   holds the light), then import the light:

   ```sh
   git clone https://github.com/jtannahill/amaran-mcp && cd amaran-mcp
   uv run amaran-mcp import
   uv run amaran-mcp status
   ```

2. Register it with Claude Code:

   ```sh
   claude mcp add -s user amaran -- \
     uv --directory "$PWD" run amaran-mcp serve
   ```

3. Ask for something: "warm white at 40 percent", "a slow violet", "start
   music mode, deep".

`import` writes `~/.config/amaran-mcp/light.json` (mode 600; it holds the mesh
keys). The first Bluetooth use asks for permission for the app running the
command. If the light was paired with the amaran phone app instead,
`uv run amaran-mcp import --account you@example.com` fetches the keys from the
amaran account (the password is asked for and never stored; untested).

## Tools

| Tool | Does |
|---|---|
| `get_light` | Power, brightness, mode and, over Bluetooth, battery |
| `set_power`, `set_brightness` | On or off; brightness in percent |
| `set_white` | Colour temperature in kelvin, with optional green or magenta tint |
| `set_color` | Hue, saturation and brightness |
| `set_effect`, `stop_effect` | Built-in effects such as fire, TV and lightning, and the pixel effects through Desktop |
| `save_scene`, `recall_scene` | Name the current look and bring it back |
| `start_music_mode`, `stop_music_mode` | Follow what the Mac plays, in a mood |
| `mark_music_moment` | Flag a lift, vocal, build or drop for training |

The same from the terminal:

```sh
uv run amaran-mcp on | off | status
uv run amaran-mcp brightness 40
uv run amaran-mcp white 3200 60      # kelvin, brightness, optional tint
uv run amaran-mcp color 30 100 50    # hue, saturation, brightness
uv run amaran-mcp effect fire 70
uv run amaran-mcp music --mood deep
```

## Next

Neither of these is built yet.

| Plan | What it would do |
|---|---|
| **The body as well as the music** | Layer health wearable data on top of music mode: heart rate from an Apple Watch, strain and recovery from WHOOP. The music still sets the sections and the colour; the body modulates them to some degree, for example a calmer, lower drift as your heart rate settles, or a fuller drop when you are moving |
| **More lights** | Several amaran lights at once: a key, fill and back light for interviews, set together from one prompt |
| **Keep the shot lit** | Watch the camera through a take and hold the face's exposure and skin tone steady, nudging brightness and colour temperature as daylight shifts or the speaker moves |

## How it reaches the light

`AMARAN_BACKEND=auto` (the default) picks one on every call:

| | amaran Desktop running | Desktop closed |
|---|---|---|
| Path | Desktop's local OpenAPI WebSocket, `ws://127.0.0.1:33782/ws` | Bluetooth Mesh, directly |
| Speed | Tens of milliseconds | A second or more |
| Effects | Desktop's full list, pixel effects included | The light's built-in effects |
| Battery | Not available | Available, with the light's own status reports |

Set `AMARAN_BACKEND=desktop` or `bluetooth` to force one.

<details>
<summary>The Desktop API key</summary>

The Desktop API wants a token in each request, signed with the key from an
approved amaran OpenAPI application (apply in amaran Desktop's User Center).
Put it in `~/.config/amaran-mcp/openapi_key` (mode 600) or
`AMARAN_OPENAPI_KEY`. Without one the server signs with a random key, as the
Bitfocus Companion module does. amaran Desktop 1.1 accepts any token, so
whether an approved key is used exactly as intended cannot be verified yet.

</details>

<details>
<summary>Bluetooth Mesh, directly</summary>

The light acts as a Bluetooth Mesh proxy. The server connects when a tool
runs, sends encrypted mesh messages (Sidus opcode `0x26`), reads the light's
status and battery replies, then disconnects after 30 seconds idle so the
amaran apps can reach the light again.

The mesh keys come from amaran Desktop's local database. The server sends from
its own mesh address and stores its message counter, which only goes up, in
`~/.config/amaran-mcp/state.json`. Lights ignore any message whose counter is
not higher than the last one they saw from that address, so the counter must
never go back.

</details>

Tested on a PT2c: status, brightness, white, tint, colour, the fire effect and
power were each confirmed by the light's own status reports.

## Limits

- One light. Custom per-pixel patterns are not supported; the built-in effects
  are, including the pixel effects through amaran Desktop.
- The light accepts one Bluetooth connection at a time. The auto backend
  avoids fighting amaran Desktop over it, but a phone app holding it still
  blocks the Bluetooth backend.
- Through amaran Desktop, the active mode is the last one this server set; a
  change made in the amaran apps shows only its power and brightness.
- Re-pairing the light in amaran Desktop changes the keys: run `import` again.

## Credits

`src/amaran_mcp/sidus/` is vendored unmodified from
[neyako/amaran-hacs](https://github.com/neyako/amaran-hacs) (MIT), which builds
on [wesbos/amaran-BLE-control](https://github.com/wesbos/amaran-BLE-control).
The amaran Desktop message and token format follow
[bitfocus/companion-module-amaran-desktop](https://github.com/bitfocus/companion-module-amaran-desktop) (MIT)
and the [amaran OpenAPI docs](https://tools.sidus.link/openapi/docs/protocol).
The section model learns from the [SSME dataset](https://osf.io/eydxk/)
(Yadati, Larson, Liem and Hanjalic). The demo tracks stream from SoundCloud.
