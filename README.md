# framewise

Coding agents can't read video. These two skills let them.

| | |
| --- | --- |
| **`/watch`** | Find what broke in a screen recording, with timestamps. |
| **`/replicate`** | Take an interface apart so it can be rebuilt — layouts, plus motion measured frame by frame. |

Runs locally. No API key, no upload.

## Why

Sampling screenshots every second loses the thing you needed. A screenshot
shows a spinner exists; it can't show that it never stopped, which is the
bug. Duration is the finding.

Rebuilding has the same problem in reverse. A still frame can't contain
motion, so screenshots leave duration, distance and easing to guesswork.

## Install

```bash
git clone https://github.com/Rohx24/framewise
cd framewise && pip install -r requirements.txt && ./install.sh
```

Needs `ffmpeg` on PATH. Installs `/watch` and `/replicate`.

Try it without a recording:

```bash
python3 examples/make_sample.py examples/sample_bug.mp4
python3 scripts/analyze.py examples/sample_bug.mp4 -o /tmp/demo --no-audio
```

## /watch

360 frames become 5 events and 7 key frames:

```
| 3.00s          | local-change    | 0.82% of the screen changed (small region, middle-right)
| 3.30s - 11.97s | busy-indicator  | something animated continuously for 8.7s while the
                                     rest of the screen was static
```

> **Likely hang.** Something animated continuously from 3.30s and was still
> going when the recording ended 8.7s later, with the rest of the screen
> frozen. It started 0.1s after a change at 3.20s.

Event kinds are visual patterns, not diagnoses:

| kind | pattern | usually |
| --- | --- | --- |
| `busy-indicator` | one small area animates, all else still | spinner, progress bar |
| `transient` | a state appears for a few frames, then moves on | flash of an error or stale value |
| `jank` | motion advances unevenly | dropped frames |
| `static` | not one pixel changed for over a second | frozen UI |
| `scroll` / `pointer` | frame translation / cursor-sized thing moving | scrolling, mouse |
| `local-change` / `region-change` / `cut` | small / large / near-total repaint | click, modal, navigation |
| `flash` | global brightness step | theme flip, unstyled flash |

`transient` and `jank` cover the bugs nobody can report — a three-frame
flicker can't be screenshotted, and jank can only be described as "it feels
bad".

Key frames include a magnified crop of the changed region, cut from the
full-resolution frame before downscaling.

## /replicate

A recording is states and transitions, and they need different treatment.

- **States** — layouts it rests in. One full-resolution frame plus a hex
  palette.
- **Transitions** — sampled densely and measured: scale, centre,
  brightness, coverage, sharpness, edge colour separation per frame, with a
  CSS `cubic-bezier` fitted.

```bash
python3 scripts/replicate.py clip.mov -o rebuild/
python3 scripts/replicate.py clip.mov --from 6.3 --to 7.3 -o rebuild/
```

`examples/make_motion_sample.py` renders a circle scaling on a cubic
ease-out over 600ms. `replicate` isn't told:

```
### Transition 0 — 1.02s to 1.52s (500ms)
- Scale (subject area, 1.0 = largest seen) — 0.019 to 1
  - best easing `ease-out` = `cubic-bezier(0.0, 0.0, 0.58, 1.0)` (rmse 0.033, close fit)
```

Right family and magnitude, a neighbouring curve rather than the exact one.
The missing 100ms is real: the tail of an ease-out moves sub-pixel amounts.
Both facts are asserted in the tests, including that `linear` is rejected.

When no easing fits any channel, the report says so — that means staged
keyframes, a spring, or per-frame shader work, and no single tween will
match.

Edge colour separation indicates refraction. CSS doesn't split red from blue
at an edge; shaders do.

## How it works

```
  video
    │
    ├─ 1. signals ── decode once, diff at full resolution, reduce to
    │                per-frame series. numpy, no ML.
    │                changed pixels · where · translation · brightness
    │
    ├─ 2a. events ──── bounded events                     →  /watch
    │
    └─ 2b. measure ─── states + transitions, easing fit   →  /replicate
                              │
                              ▼
                    3. markdown + frames
                              │
                              ▼
                       4. the model reads it
```

Steps 1–3 are arithmetic, which is why the timestamps hold up. Step 4 is the
only place meaning is assigned. The pipeline reports what changed, where and
for how long; both skills carry a rule against naming a mechanism that
wasn't measured, since a scale curve looks the same from CSS, a spring or a
vertex shader.

## Three traps, all of which shipped here first

**Measure change before summarising.** Downscaling and then diffing averages
a 20px spinner to nothing. The first version reported a hung UI as
"completely static".

**A busy region can threshold itself out.** A per-cell noise floor learned
over time lets a spinner occupying its cell 72% of the time push its own
percentile into its own signal. The floor is the spatial median within each
frame instead — one busy cell can't move it.

**Screen recordings are variable frame rate.** Gaps in one file ranged from
7ms to 83ms, 148 of 412 more than 30% off the median. `index / fps` slides
every timestamp, and seeking that way extracts the wrong frames.

Colour matters too: red `(224,75,74)` and green `(32,160,110)` differ by
about 3 in luminance, so a grayscale pipeline can't see an error banner
become a success banner. Deltas compare colour channels.

## Output

Plain Markdown plus JPEGs, carrying their own reading instructions. Works in
any assistant with vision. `events.json` holds the same data for scripts.

A 2-minute 1080p recording takes about 20 seconds — roughly 6× faster than
watching it. Decoding dominates, so resolution barely matters.

## Tests

```bash
python3 tests/test_detection.py
```

40 checks against generated fixtures. Most are bugs that shipped: the
vanishing spinner, a click swallowed by pointer motion, a compressed
re-encode moving a hang five seconds late and reporting it four times,
phantom scrolls on a blank screen, forty false flashes during one scroll,
and an ease-out truncated until it read as linear.

## Limits

- Pixels only — no clicks, keystrokes, network or console. A change where
  the cursor is, is consistent with a click, not proof of one.
- A flash needs settled state either side. One mid-scroll can't be isolated,
  and back-to-back flickers report as one.
- Jank detection is whole-page; an inner scrolling panel isn't caught.
- Easing lands in the right family, not always the exact curve.
- Sub-pixel detail is gone in a compressed recording. Exact type sizes and
  1px borders are inference.
- Tuned for screen recordings. Camera footage and video playback read as
  continuously busy.

## License

MIT
