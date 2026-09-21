# video-repro

**Make coding agents able to read bug reports that are screen recordings.**

Someone hands you a `.mov` and says "it breaks at 0:14". Your agent can't
watch it, so you transcribe it into prose by hand and lose half the detail.

`video-repro` turns the recording into a timestamped timeline plus a handful
of key frames — something any model with vision can actually read.

```
| `3.00s`          | local-change    | 0.82% of the screen changed (small region, middle-right)
| `3.30s - 11.97s` | busy-indicator  | something animated continuously for 8.7s while the
                                       rest of the screen was static
```

> **Likely hang.** Something animated continuously in a small region,
> middle-right from 3.30s and was *still going when the recording ended*
> 8.7s later, with the rest of the screen frozen. It started 0.1s after a
> change at 3.20s, which is where the triggering interaction most likely is.

## Why not just take screenshots

Because the answer is usually in the gaps. A screenshot every second tells
you a spinner exists; it can't tell you the spinner never stopped, which is
the entire bug. Duration *is* the finding.

Sampling also silently loses small things. Measuring change on a downscaled
frame averages a 20px spinner into nothing — the first version of this tool
confidently reported a hung UI as "completely static". Deltas are computed at
full resolution here and only summarised afterwards.

The same trap has a second form, and it is subtler. Learning a per-cell noise
floor over time seems obviously right, until you notice that a spinner
occupying its cell 72% of the time pushes its own percentile up into its own
signal and thresholds itself out of existence — reporting the hang, again, as
a frozen screen. The floor here is the *spatial* median within each frame: in
any single frame of a screen recording nearly every cell is static, so one
busy cell cannot move it. Both failures are in the test suite.

## How it works

Three layers, cheap to expensive:

1. **Signals** — decode every frame once, diff at full resolution, reduce to
   per-frame time series (changed pixels, where, translation, brightness).
   No ML. Pure numpy.
2. **Events** — segment those signals into bounded events. A spinner and a
   mouse cursor are both one small busy region; what separates them is that
   only one of them travels.
3. **Report** — findings, a key-frame budget, and a Markdown artifact.

The model only ever sees step 3. A 12-second recording becomes 5 events and
7 frames instead of 360 images.

Nothing in the pipeline interprets meaning — it reports what changed, where
and for how long. Calling it a failed save request is the reading model's
job, and keeping that boundary is what stops it inventing a story.

## Install

```bash
git clone https://github.com/YOURNAME/video-repro
pip install -r requirements.txt   # opencv-python, numpy
```

Needs `ffmpeg` on PATH. Optional: `pip install openai-whisper` to transcribe
narration — people say "and now it just hangs" out loud, and that is often
the most informative thing in the recording.

## Use

```bash
python3 scripts/analyze.py bug.mov -o repro/
```

Writes `repro/timeline.md`, `repro/frames/`, `repro/events.json`.

As a Claude Code skill, drop the repo in `~/.claude/skills/video-repro/` and
it triggers on its own whenever a recording comes up. It is plain Markdown
and images, so it works just as well pasted into any other assistant.

## Try it without a recording

```bash
python3 examples/make_sample.py examples/sample_bug.mp4
python3 scripts/analyze.py examples/sample_bug.mp4 -o /tmp/demo --no-audio
```

Renders a synthetic hung-save bug and analyses it. Sample output is in
[`examples/expected/`](examples/expected/timeline.md).

## Performance

A 2-minute 1080p recording analyses in about 20 seconds on a laptop — roughly
6× faster than watching it, and resolution barely matters because decoding
dominates. Transcription adds a few seconds.

Output is small on purpose: that recording yields ~12 key frames, not 3,600.

## Tests

```bash
python3 tests/test_detection.py
```

Every case in there is a bug that actually shipped during development: the
vanishing spinner, the click swallowed by pointer motion, a compressed
re-encode moving the reported hang five seconds late and repeating it four
times, and the self-suppressing noise floor. They run against generated
fixtures, so there is nothing to download.

## Limits

- Pixels only. No clicks, keystrokes, network or console — a change where the
  cursor is, is *consistent with* a click, not proof of one.
- Tuned for screen recordings. Camera footage and video playback read as
  continuously busy and produce a thin timeline.
- Key frames are downscaled to 1280px; re-extract full-size if you need to
  read small text.

## License

MIT
