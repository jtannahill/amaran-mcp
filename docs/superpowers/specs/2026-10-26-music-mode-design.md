# Music mode: design

Date: 2026-10-26. Status: approved by the owner on 2026-10-26; revised the
same day after the first live run (see "Revision: musical understanding"),
revision approved by the owner in conversation.

## Intent

The owner plays house and trance at their desk, from the laptop or from their
iPhone, and wants the amaran PT2c tube to move with the music. It must feel
abstract and ethereal: the light breathes with the track and shifts colour
when the music changes, without ever reading as a beat-synced party light.
Their words: "not that amateur, make it more abstract", "ethereal".

Success: an hour of trance at the desk where the light clearly follows the
music, above all its breakdowns (the chords moving, the moment the chords
lift and the vocal comes in), and nothing ever flashes, cuts or becomes too
dim or too saturated to work under.

## Goals

- Follow the music the owner is actually hearing. They listen on AirPods, so
  the main input is the Mac's own sound output (system audio capture), not a
  microphone:
  - Laptop playback: capture the Mac's output on its way to the AirPods.
  - iPhone playback: the owner AirPlays from the iPhone to the Mac (macOS AirPlay
    Receiver), the Mac plays it to the AirPods, and the same capture hears it.
    The owner agreed to this.
  - Microphone only as a fallback, when the Mac's output is its own built-in
    speakers or another wired/non-Bluetooth output and system capture is
    unavailable.
- When Apple Music is playing on this Mac, use the current track as a bonus
  signal (see Apple Music below). Everything must work without it.
- Drive the tube with slow, smooth changes:
  - The tube shows one colour at a time and drifts slowly through a small
    palette (for example deep blues, violets, soft teals). A colour change
    takes several seconds.
  - Energy in the music shows as slow swells in brightness and saturation,
    smoothed over seconds. The track "breathes"; individual kicks are not
    visible.
  - Section by section behaviour, below in "Revision: musical understanding".
    (The first version's rule, "breakdowns go paler, a drop shifts the
    palette", was too simple: the owner's words were "Trance is not as simple
    as beats and drops".)
  - When the music stops, the tube settles back to the state it was in before
    music mode started.
- Start and stop from Claude ("start music mode", optionally naming a mood),
  and from the CLI.

## Anti-goals

Any of these makes the feature a failure even if it technically works:

- strobing, flashing, hard cuts, or a visible pulse on every beat
- brightness so low, or colour so saturated, that the desk stops being a
  usable place to work
- the built-in effect presets (party lights, cop car, and so on) standing in
  for the music response

## Non-goals

- Per-beat synchronisation.
- Per-pixel patterns along the tube. The APIs available drive the tube as one
  colour; the built-in pixel effects cannot follow audio.
- Camera input, the shedua.com remote endpoint, Spotify, an iPhone app. Each
  may come later as its own project.
- Track information for iPhone playback. AirPlay brings the sound, not the
  Music app's track details.

## Constraints

- macOS, this Mac, one light (the PT2c already configured in amaran-mcp).
- Commands go through amaran-mcp's existing backends. amaran Desktop's local
  API answers in 10 to 40 ms and is the expected path; direct Bluetooth works
  but each write is slower, so music mode must still look smooth at a lower
  update rate there.
- Through either API the tube shows one HSI colour or one white at a time.
- Never open a Bluetooth headset's microphone. Opening AirPods as an input
  switches them to the low-quality call codec and ruins the music; music mode
  must refuse any Bluetooth input device, even when it is the default input.
- macOS will ask for system audio capture permission (and for microphone
  permission only if the fallback is used) for the app that runs the process,
  and Automation permission to read the Music app. Each prompt happens once;
  the feature must give a clear message naming the System Settings pane when
  one is refused, and work without Automation.

## Decided

These were described to the owner, who said nothing was wrong or missing:

- The input is the Mac's own sound output, because the owner listens on
  AirPods; iPhone music reaches the Mac by AirPlay. Apple Music is a bonus that
  adds track information when it is playing on the laptop.
- Behaviour as listed under Goals: slow palette drift, energy as slow swells,
  paler breakdowns, palette shift on drops, settle back when music stops.
- Controlled by asking Claude to start or stop music mode, optionally with a
  mood. It runs as a background process on the Mac.
- The one-colour limit above.

## Builder's calls (made by Claude, not confirmed by the owner)

- **Process.** Music mode runs as its own background process
  (`amaran-mcp music [--mood NAME]`), not inside the MCP server, so it keeps
  running after a Claude session ends. A PID file under
  `~/.config/amaran-mcp/` lets the MCP tools find and stop it. One instance at
  a time; starting again with a different mood restarts it.
- **Ownership of the light.** While music mode runs, any other MCP command
  that changes the light (set_white, set_color, set_effect, recall_scene,
  set_power) stops music mode first, then applies. The exception is
  set_brightness: it shifts music mode's brightness range up or down, so
  "brighter" keeps the session going. That needs a small control channel into
  the running process (for example a file or local socket the process
  watches). get_light reports that music mode is running.
- **Silence and other sounds.** After about 10 s of silence the tube settles
  back to its starting state but music mode keeps listening, so pausing a
  track does not end the session; it picks up again when music returns. It
  exits on its own after 15 minutes of continuous silence. get_light reports
  "listening" versus "playing". Sound without sustained bass energy (speech,
  a video call) does not count as music and does not move the light.
- **Permissions and launch.** The owner runs Claude Code in Terminal, so
  macOS asks Terminal for system audio capture (and Automation) access. The MCP tool
  starts music mode as a detached child of its own process (new session,
  output to a log file), so the permission stays attributed to Terminal and
  the process outlives the Claude session. Not launchd. If capture is refused
  or delivers only zeros while something is playing, music mode exits with a
  message naming the System Settings, Privacy and Security pane to fix.
- **Tools.** `start_music_mode(mood?)`, `stop_music_mode()`, and a
  `music_mode` field in `get_light` (running or not, mood, current track if
  known).
- **Moods.** Three, each a palette plus brightness range: `ethereal`
  (default: deep blue, violet, soft teal, misty white), `deep` (indigo, navy,
  magenta-violet, low and slow), `dawn` (lavender, peach, pale gold, warmer
  whites). Exact colours are tuning work.
- **Audio capture.** A small Swift helper, built from source in the repo by
  the setup step, captures system output with a Core Audio process tap
  (macOS 14.2 and later) and writes mono float PCM, about 22 kHz, to stdout;
  the Python process reads it. If the tap cannot be created, the helper falls
  back to ScreenCaptureKit audio-only capture. The microphone fallback uses
  `sounddevice` on the built-in microphone by name and skips any Bluetooth
  device.
  - Risk to check first, before the rest is built: that a system-wide tap
    hears AirPlay Receiver audio (it is played by a system process, not an
    app). If it does not, say so to the owner before going further.
- **Audio analysis.** Short blocks analysed with numpy: overall loudness,
  bass / mid / high band energy, and spectral flux. No heavy analysis
  libraries. Capture is post-volume or pre-volume depending on the tap; the
  analysis normalises to a long-term average either way, so volume changes do
  not swing the light.
- **Shaping.** Slow envelope followers turn the features into a smooth
  "energy" (rise over about 1.5 s, fall over about 4 s). A long-term average
  gives the reference level, so it adapts to how loud the music is.
  - Breakdown: bass energy well below its recent average for a few seconds.
  - Drop: bass comes back strongly after a breakdown (or a sharp sustained
    jump in energy).
  - Silence: input near the noise floor for about 10 s.
- **Rendering.**
  - Hue drifts along the mood's palette on a slow loop (about 45 to 90 s per
    cycle), slightly faster with more energy.
  - Brightness moves between a floor and a ceiling (about 25% to 70%) with
    energy.
  - Saturation rises with energy and drops toward 20 to 30% in breakdowns.
  - A drop advances to the next palette anchor with a 2 to 3 s crossfade.
  - Every output is slew-limited: a maximum change per second in hue,
    saturation and brightness. The anti-goals are enforced here, whatever the
    analysis does.
  - Update rate about 8 per second through amaran Desktop, about 3 per
    second over Bluetooth.
- **Settling.** At start, record the light's current state. After silence,
  or on stop, fade back to it over about 5 s.
- **Apple Music.** Every few seconds, ask the Music app (via `osascript`) for
  its player state, track name and artist, only if Music is already running,
  so the check never launches it. On a track change, choose that track's
  starting palette anchor from a hash of the track name, so the same song
  tends to start in the same colour. The track name and artist appear in
  `get_light`.
- **Tests.** Offline tests feed synthetic audio (silence, steady tone, a
  4/4 kick pattern, a breakdown into a drop) through the analysis, and check
  the renderer's slew limits and brightness bounds. No real audio capture in
  tests. A live tuning pass with real music follows.

## Revision: musical understanding

After the first live run on a real track the owner said: "Trance is not as
simple as beats and drops...look it up". Research summary, which this
revision rests on:

- A trance track is a long journey, not a loop: intro, theme, a long soft
  breakdown with the beats gone, a rebuild, the anthem section, outro
  (Sound on Sound; Wikipedia, Trance music).
- The breakdown is the emotional peak, not a quiet gap: pad chords, delay
  tails and the vocal come forward and carry the track (Sound on Sound;
  Wikipedia, Uplifting trance).
- Harmony carries the feeling: long chord progressions, and the balance of
  major and minor sets euphoric versus melancholic (Wikipedia, Uplifting
  trance).
- A build is accumulating tension: risers, white-noise sweeps, snare rolls,
  filters opening, sometimes a key change up (production course outlines,
  Ableton forum).
- Measurable: harmonic change from chroma; tension tracks roughness and
  noisiness better than brightness alone (Farbood and Price 2017).

### The moment that matters most (decided by the owner)

Asked which moment in a trance track they most want to see in the light, the
owner said "the breakdown", and within it "when the chords lift and the vocal
comes in". Offered "the light blooms: the colour warms with the lift, and as
the voice enters the light slowly swells and turns luminous, then eases back
when the vocal phrase ends", the owner said "yes, bloom like that".

### Behaviour by section (described to the owner, who said nothing was wrong)

- **Breakdown (the centrepiece).** When the kick drops out the light goes
  quieter and more intimate and the harmony drives it. Each chord change is a
  slow wash to a new colour: minor chords pull deep and cool (blue, violet),
  major chords open warmer (lavender, rose, soft gold). Changes land on the
  music's phrases, never on beats.
- **The lift and the vocal: bloom.** When the chords lift and a voice enters,
  the light warms, slowly swells and turns luminous with the voice, then eases
  back when the vocal phrase ends.
- **Build.** The light gathers: its drift slows, the colour deepens and
  concentrates as risers and noise pile up, and it holds.
- **Release.** It opens out, fuller and brighter, into the anthem section.
  (This replaces "a drop moves to a new palette".)
- **Intro, groove, climax** (Claude's guess, accepted with the rest): the
  intro (first music after silence) is restrained, the groove breathes and
  drifts as in the first version, the climax is the fullest the mood allows.
  The outro section was dropped once the owner said they listen to continuous
  mixes.
  House mostly gets the groove behaviour.
- Unchanged: the anti-goals, the moods, start and stop from Claude, system
  audio capture including AirPlay.

### How the owner listens (decided by the owner)

Continuous mixes and radio shows: A State of Trance episodes, Andrew Rayel's
Find Your Harmony, Future Sound of Egypt. Consequences:

- Tracks blend into each other with no gap, so there is no track-level intro
  or outro to detect. Sections become: groove (including DJ blends between
  tracks), breakdown, build, climax, and silence. "Intro" only means the first
  stretch of music after silence; there is no outro section.
- The Music app's track name is of little use: an episode is one long item.
  It stays a bonus for status display; the first version's "same track, same
  starting colour" still applies per episode and is otherwise harmless.
- The hosts talk over the music (voice-overs, show IDs, jingles). Speech
  over music must not bloom.

### Bloom trigger and tuning clips (decided by the owner)

Asked, given that hosts talk over the music: the bloom needs a chord lift
during a breakdown; a sung vocal at that moment makes the bloom bigger. A
voice outside a breakdown does not bloom. The owner answered "ok".

Also answered "ok": short clips of episodes may be recorded to a local file
on this Mac for tuning, so breakdowns can be replayed offline. Clips stay on
this Mac, outside the repository, and are never uploaded.

### Risks the owner was told about

- Detecting a voice inside a full mix is the hardest part. Start with a
  lightweight method tuned on the owner's tracks; it may need a small machine
  learning model later.
- Chord detection is reliable on sustained chords (breakdowns), rougher under
  a full beat.

### Builder's calls for the revision (Claude, not confirmed by the owner)

- **Harmony.** 12-bin chroma from the FFT (numpy), smoothed over about 1 s;
  match against the 24 major and minor triads; report the current chord, its
  quality (major or minor) and a confidence. A "lift" is a change to a major
  chord, or a rise in the major/minor balance over about 2 s, with confidence
  above a threshold. Chord colour: each mood gains a cool set of stops (used
  for minor) and a warm set (used for major); the chord root picks the stop
  within the set, so the same progression repeats the same colours.
- **Vocal presence.** A 0..1 smoothed estimate from harmonic energy in the
  vocal band (about 200 Hz to 4 kHz) that is not in the sustained pad: pitch
  movement (vibrato around 4 to 7 Hz, glides) and harmonic peaks that come
  and go, against the slow-changing accompaniment. Tuned on the owner's
  tracks. An ML model is not part of this revision. Because the bloom is
  gated by a chord lift in a breakdown, the vocal estimate only scales the
  bloom's size; a lead synth mistaken for a voice makes a bigger bloom, never
  a spurious one, and host speech outside a lift does nothing.
- **Tension.** From spectral flatness (noise), the rising share of high
  frequencies (filters opening), and increasing onset density (snare rolls),
  each relative to the recent past.
- **Openness.** Spectral centroid relative to the track's own range: more
  open moves the light toward luminous and airy.
- **Tempo and phrases.** Estimate BPM (100 to 150) from the onset envelope's
  autocorrelation, keep a beat phase, count bars, and treat every 8 bars as a
  phrase boundary. Colour moves that come from structure (chord washes,
  section changes) start on the next phrase or bar boundary when tempo is
  known (bar boundary if the next phrase is more than about 8 s away), and
  immediately when it is not. The bloom follows the vocal directly; it is
  already slow.
- **Sections.** "intro", "groove", "breakdown", "build", "climax", plus
  "silence" (no outro; see How the owner listens). Intro: the first stretch
  of music after silence, until the full kick and bass arrive. Breakdown:
  kick gone for several bars after a groove. Build: tension rising over
  several bars (inside or after a breakdown). Climax: full energy after a
  build, held until energy falls back to groove level.
- **Feature log.** `amaran-mcp music --log-features PATH` writes one JSON line
  per tick (features and output) so tuning on real tracks can be done offline
  afterwards.
- **Pale breakdowns, superseded.** The first version's "breakdowns go paler"
  (and its 25% saturation cap) would hide the chord colours. A breakdown
  starts pale and intimate for its first bars, then colour builds as the
  harmony takes over; the cap is removed.
- **Chord pace.** Chords in a breakdown can change every 1 to 2 bars (1.7 to
  3.5 s at 138 BPM). The light answers each change, starting on the bar. The
  stops within a cool or warm set sit close together in hue (about 20 to 40
  degrees apart) so a wash completes in about a bar within the existing slew
  limits; the major/minor balance over the last 8 to 16 bars chooses between
  the cool and the warm set, so the overall run of the harmony sets the
  colour family and each chord moves within it.
- **What stays.** Capture, the runner's writer and fades, settling, control,
  the MCP tools and the slew limits are kept. The renderer and analyzer are
  extended; the first version's breakdown and drop tests are replaced by
  section tests.

## Left to the builder

Exact colours, time constants and thresholds (tuning), internal module
layout, how the PID file and logging work, and what the CLI prints.
