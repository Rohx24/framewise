---
name: replicate
description: Take a video of an interface apart so it can be rebuilt - resting states with full frames and palettes, and transitions measured frame by frame with easing curves fitted. Use whenever someone wants to recreate, clone, copy, port or match something they are showing in a video or screen recording - a page, a component, a layout, an interaction, an animation or a transition - or says things like "make it like this video", "rebuild this", "recreate this effect", "why does mine not look like theirs", or is comparing their build against a reference recording.
---

# replicate

Rebuilding an interface from a video fails in a specific way: the static
parts come out nearly right and the moving parts come out about half right.
That is not a looking-harder problem. A still frame cannot contain motion,
so any number of screenshots leaves you guessing at duration, distance and
easing — and guesses at those are what "50% there" looks like.

So this splits the recording in two and treats each half differently.

- **States** — the layouts it rests in. One clean full-resolution frame
  answers nearly everything, because type, spacing and colour are all
  readable straight off the pixels.
- **Transitions** — the motion between them. Sampled densely and
  *measured*: scale, position, brightness, coverage, blur and colour
  separation per frame, with an easing curve fitted.

## Run it

```bash
OUT="$(mktemp -d)/rebuild"
python3 "<this skill's directory>/scripts/replicate.py" VIDEO -o "$OUT"
```

Then read `$OUT/rebuild.md`. Narrow to one moment with `--from` / `--to`
(seconds) when the user points at a specific part; otherwise it finds the
states and the busiest transitions itself.

| flag | why |
| --- | --- |
| `--from 6.3 --to 7.3` | measure one specific moment |
| `--frames 30` | denser sampling of each transition |
| `--max-states`, `--max-transitions` | caps for a long recording |

Needs `ffmpeg`, `opencv-python`, `numpy`. Everything runs locally.

## Using what it gives you

**For a state**, work from the frame. It is full resolution — read the
type, the spacing, the borders, the alignment. The palette is listed as
hex with the share of pixels each covers, so you can tell a background
from an accent.

**For a transition**, work from the numbers:

- *Scale* is the subject's area relative to its largest, so `0.08 -> 1.0`
  is a thing growing from nearly nothing.
- *Centre X/Y* is position, as a fraction of the region. Flat means it did
  not move.
- *Brightness* and *coverage* together read as a fade.
- *Sharpness* rising means blur clearing.
- *Edge colour separation* is the giveaway for refraction or dispersion —
  CSS does not split red from blue at an edge, shaders do.

The fitted easing is a real CSS `cubic-bezier`, ready to paste. Check the
rmse: a close fit means that curve is the answer, and a poor fit across
every channel is itself the finding — it means staged keyframes, a spring,
or per-frame shader work, and no single tween will ever match it.

The contact sheet shows the whole motion at once. Read it before the
per-frame crops; it is usually the fastest way to see what is happening.

## Rules

- **Never name the technique as though it were measured.** A scale curve
  looks identical from CSS, from a spring, or from a vertex shader. State
  what the pixels did, then say what you are inferring and why.
- **Separate measured from inferred** in whatever you write. "Grows over
  1009ms" is measured. "Probably a WebGL refraction pass" is inference.
- **Don't invent a mechanism to explain something.** If the curves do not
  fit, say they do not fit.
- Sub-pixel detail is gone in a compressed recording. Exact type sizes and
  1px borders are inference, not measurement.
- For finding *bugs* in a recording rather than rebuilding it, use the
  `watch` skill instead. They share this codebase.
