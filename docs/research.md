# Music mode: research, tuning and training

How music mode learned to follow trance: what the research says, what real
music taught us, the datasets used, and how every number here can be
reproduced. No audio is in this repository: public datasets are downloaded
to `~/datasets`, and the owner's own recordings stay in
`~/.config/amaran-mcp/clips` on their Mac.

## 1. What music mode is for

One amaran PT2c tube at a desk, following house and trance played on the Mac
or AirPlayed from an iPhone (A State of Trance, Find Your Harmony, Future
Sound of Egypt). It should feel abstract and ethereal: no flashing, no pulse
on every beat. The moment that matters most to its owner is the breakdown,
"when the chords lift and the vocal comes in", where the light should bloom.

## 2. How trance is built (the research)

- A trance track is a journey, not a loop: intro, theme, a long soft
  breakdown with the beats gone, a rebuild, the anthem, outro. Tracks run long
  so they can progress. ([Sound on Sound](https://www.soundonsound.com/techniques/secrets-house-trance))
- The breakdown is the emotional peak, not a gap: pad chords, delay tails and
  the vocal come forward. Uplifting trance stretches it and brings wash pads
  and chord layers up front. ([Sound on Sound](https://www.soundonsound.com/node/4914907),
  [Wikipedia: Uplifting trance](https://en.wikipedia.org/wiki/Uplifting_trance))
- Harmony carries the feeling: long progressions, with the balance of major
  and minor setting euphoric versus melancholic. ([Wikipedia](https://en.wikipedia.org/wiki/Uplifting_trance))
- A build piles up tension: risers, white-noise sweeps, snare rolls, filters
  opening, sometimes a key change up.
- Measurable cues: harmonic change from chroma; tension follows roughness and
  noisiness more than brightness ([Farbood and Price 2017](https://bpb-us-e1.wpmucdn.com/wp.nyu.edu/dist/f/11865/files/2020/08/Farbood_Price_2017.pdf));
  EDM-specific measures such as sub-bass ratio and sidechain pumping
  ([arXiv 2509.11474](https://arxiv.org/pdf/2509.11474)).

What music mode hears, built on that:

| Reading | How | Module |
| --- | --- | --- |
| Tempo, bars, 8-bar phrases | kick-band onset autocorrelation, locked beat grid | `music/tempo.py` |
| Kick in or out, right now | beat strength over the last 2 s at the remembered tempo | `music/tempo.py` |
| Chord, major/minor balance, lift | 12-bin chroma matched to 24 triads | `music/harmony.py` |
| Tension, openness | noisiness and high band, onset density, spectral centroid | `music/timbre.py` |
| Sections, bloom, release | state machine over the above (a trained model in progress) | `music/analysis.py` |

## 3. What the first real breakdown taught

A 3-minute clip of an ASOT breakdown (B flat minor, A flat, G flat, A flat),
recorded locally on 2026-10-26, replayed with `amaran-mcp music-replay`.

| Finding | Evidence | Change |
| --- | --- | --- |
| The breakdown kept its sub-bass, so "kick" by bass level never left | bass stayed at about 0.8 of its groove level through the breakdown | kick presence now from beat strength over the last 2 s at the remembered tempo: above 0.99 in the groove, below 0.12 in the breakdown, switching at 1:30 exactly as heard |
| Tempo confidence (8 s window) is too slow to time a release | it took 6 to 9 s to recover | a release is a sharp jump in low-band onsets (last 0.5 s against the 8 s before) with strong pumping, or a confirmed kick |
| A bass swell in the breakdown looked like a release | onset jump 2.4, pumping 0.31 | release needs a jump of 3 and pumping of 0.45 |
| Every chord change onto major counted as a lift | A flat to G flat would bloom on every chord | lifts are minor to major only |
| The heuristic vocal estimate read high through the groove | 0.3 to 0.9 everywhere | to be replaced by Apple's on-device sound classifier (section 5) |

Result after tuning: the breakdown is found at 1:36 (the kick left at about
1:30; the rule waits 4 bars), and the light blooms each time the progression
lifts from B flat minor to A flat, which the owner confirmed is right. One
false release at 1:48 remains open.

## 4. Benchmarks

### Tempo: GiantSteps tempo dataset

664 two-minute Beatport previews with crowd-corrected tempo labels
([GiantSteps](https://github.com/GiantSteps/giantsteps-tempo-dataset);
Knees et al., ISMIR 2015; Schreiber and Müller, ISMIR 2018). Scored the MIREX
way: Accuracy 1 within 4% of the label; Accuracy 2 also forgiving double,
half, triple and third tempo.

```sh
uv run python tools/benchmark_tempo.py ~/datasets/giantsteps-tempo-dataset
```

First look, 30 tracks between 100 and 150 BPM: whenever the tracker locked a
tempo it was right, but on mastered tracks its confidence (0.05 to 0.41) rarely
reached the threshold tuned on synthetic kicks (0.55): 10 of 30 locked. A
lower threshold locks 25 of 30 correctly. Full results: _in progress_.

### Key: Beatport EDM Key Dataset

1,486 two-minute excerpts across EDM genres (54 trance, 43 psy-trance, about
200 house variants) with single-key labels, CC BY-SA 4.0
([Zenodo 1101082](https://zenodo.org/record/1101082); Faraldo 2017). Music
mode follows chords, not keys, but both rest on the same chroma, so the key is
estimated from the harmony tracker's chroma and scored the MIREX way. The
mode (major or minor) alone is reported too: it decides music mode's cool or
warm colours.

```sh
uv run python tools/benchmark_key.py ~/datasets/beatport-edm-key
```

Results: _in progress_.

## 5. Machine learning

### Voice: Apple's on-device sound classifier

macOS SoundAnalysis ships a built-in classifier with 303 classes, including
speech, singing, choir_singing, rapping, synthesizer, bass_drum, crowd and
applause. It runs on the same captured audio, on-device. Singing makes the
bloom bigger; speech (a host talking over the music) never counts.

Status: _in progress_.

### Sections: a small trained model

Training data: the Socially Significant Music Events set
([osf.io/eydxk](http://osf.io/eydxk); Yadati, Larson, Liem and Hanjalic,
IEEE Transactions on Multimedia): 402 full Creative Commons SoundCloud EDM
tracks with expert-labelled drops, builds and breaks; plus the owner's own
labelled clips. The model ships only if it beats the hand-written rules on
held-out tracks.

Status: _in progress_.
