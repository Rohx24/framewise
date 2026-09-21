---
name: video-repro
description: Read a screen recording of a bug and turn it into a timestamped timeline plus key frames. Use whenever someone attaches or points at a screen recording, screencast, Loom, .mov/.mp4/.webm/.gif of a UI, or says things like "watch this video", "here's a repro video", "it breaks at 0:14", "see the recording", or when a bug report references a video instead of describing the steps. Also use to check a recording of your own UI work for stutters, hangs or broken transitions.
---

# video-repro

Coding agents are blind to screen recordings. The usual workaround — grab a
few screenshots at fixed intervals — misses exactly what a recording is for:
what happened *between* the frames, and how long it lasted.

This skill runs a continuous pass over every frame, segments the recording
into events, and emits a Markdown timeline plus a small set of key frames.
You read the frames; the timeline tells you when things moved and for how
long.

## Run it

```bash
python3 scripts/analyze.py RECORDING -o repro/
```

Then read `repro/timeline.md`. It embeds the key frames as relative image
links — open them to see the UI.

Requires `ffmpeg` and Python with `opencv-python` + `numpy`. Narration
transcription additionally needs `openai-whisper`; without it the rest still
works. Everything is local — no API key, no upload.

Useful flags:

| flag | why |
| --- | --- |
| `--frames N` | more or fewer key frames (default 12) |
| `--max-seconds N` | only analyse the first N seconds of a long recording |
| `--no-audio` | skip transcription (faster; avoids a model download) |
| `--whisper-model small` | better narration accuracy, slower |

## What the event kinds mean

The detector reports visual patterns. It does **not** know what any of them
mean — naming is descriptive on purpose.

| kind | the pixel pattern | what it usually is |
| --- | --- | --- |
| `busy-indicator` | one small area animates continuously while everything else holds still | spinner, progress bar, skeleton shimmer |
| `static` | not one pixel changed for over a second | frozen UI, or just nobody doing anything |
| `local-change` | a small area repainted once | click feedback, hover, a field updating |
| `region-change` | a large area repainted once | modal, drawer, panel, tab switch |
| `cut` | most of the screen repainted | navigation, page load, app switch |
| `scroll` | whole frame translated vertically | scrolling |
| `pointer` | a cursor-sized thing travelled across the screen | mouse movement |
| `flash` | global brightness stepped | theme flip, flash of unstyled content, white flash on load |

## Reading the output

1. **Start with "What stands out".** It flags hangs, unresolved spinners and
   dead clicks. It is hedged on purpose — confirm each one against frames.
2. **Use the timeline for timing, the frames for meaning.** The timeline is
   exact to about one frame. The frames are the only thing that tells you
   *what* the UI actually is.
3. **Anchor every claim to a timestamp.** "The spinner at 3.30s never
   resolves" is checkable. "The save is broken" is not.
4. **Then go to the code.** A `busy-indicator` that outlives the recording
   points at the request that started near its onset — the timeline names
   the preceding change, which is where the triggering interaction is.

## Do not over-read it

- The analyser sees pixels, never intent. It cannot see clicks, key presses,
  network activity or console errors. A `local-change` where the cursor is
  is *consistent with* a click — it is not proof of one.
- Region coordinates are percentages of the frame, not CSS selectors.
- Small text in key frames may be unreadable after downscaling. Re-extract a
  specific moment at full size if you need to read it:
  `ffmpeg -ss 3.30 -i RECORDING -frames:v 1 -q:v 2 out.jpg`
- If a recording is mostly video playback or continuous animation, the whole
  thing reads as busy and the timeline will be thin. Say so rather than
  inventing structure.

## Using it outside Claude Code

`repro/` is self-contained and model-agnostic. Upload `timeline.md` together
with the `frames/` images to any assistant with vision — the report carries
its own reading instructions. `events.json` holds the same data for scripts.
