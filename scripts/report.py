"""
Findings, keyframe selection, and the portable report.

The report is the actual deliverable. It is plain Markdown plus a folder of
JPEGs, so it can be read by any model with vision -- pasted into ChatGPT,
attached in Claude, committed to a repo, or pasted into a ticket. Nothing in
it depends on this pipeline being present to interpret it.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2

from events import Event, describe_region
from signals import Signals

# A busy indicator that never resolves is the classic hung-request symptom.
HANG_SECONDS = 2.0
# Stillness this long right after an interaction suggests a dropped click.
DEAD_SECONDS = 2.0
# Frames are written no wider than this. Big enough to read UI text, small
# enough that a dozen of them do not blow up a context window.
FRAME_WIDTH = 1280
FRAME_QUALITY = 82

# Close-ups of the region that changed. A downscaled full screenshot is
# fine for "a dialog opened" and useless for "the dialog is 7px off" --
# the defect is a handful of pixels wide by the time the frame is shrunk to
# fit a context window. Anything covering less than this much of the screen
# gets a magnified crop as well as the full frame.
CROP_MAX_AREA = 0.45
CROP_PAD = 0.35          # of the region's own size, added on every side
CROP_MIN_SPAN = 0.10     # never crop tighter than this much of the frame
CROP_TARGET_W = 560      # upscale small crops to at least this wide

INTERACTIONS = ("local-change", "region-change", "cut")


def findings(events: List[Event], sig: Signals) -> List[str]:
    """
    Surface the patterns that usually matter in a bug report.

    Deliberately conservative and hedged. This layer sees pixels, not
    intent -- it can say "a busy indicator never resolved", and it must not
    say "the save request failed".
    """
    out: List[str] = []
    end = sig.duration

    for ev in events:
        if ev.kind == "busy-indicator" and ev.duration >= HANG_SECONDS:
            unresolved = (end - ev.t1) < 0.4
            prior = [e for e in events
                     if e.kind in INTERACTIONS and e.t1 <= ev.t0 + 0.1]
            trigger = ""
            if prior and ev.t0 - prior[-1].t1 < 2.0:
                trigger = (f" It started {ev.t0 - prior[-1].t1:.1f}s after a "
                           f"change at {prior[-1].t0:.2f}s, which is where "
                           f"the triggering interaction most likely is.")
            if unresolved:
                out.append(
                    f"**Likely hang.** Something animated continuously in a "
                    f"{describe_region(ev.region)} from {ev.t0:.2f}s and was "
                    f"*still going when the recording ended* "
                    f"{ev.duration:.1f}s later, with the rest of the screen "
                    f"frozen.{trigger} Check the key frames to confirm it is "
                    f"a loading indicator."
                )
            else:
                out.append(
                    f"**Slow operation.** A busy indicator ran for "
                    f"{ev.duration:.1f}s ({ev.t0:.2f}s to {ev.t1:.2f}s) "
                    f"before the screen changed again.{trigger}"
                )

        if ev.kind == "static" and ev.duration >= DEAD_SECONDS:
            prior = [e for e in events
                     if e.kind in INTERACTIONS and e.t1 <= ev.t0 + 0.05]
            if prior and ev.t0 - prior[-1].t1 < 0.5:
                out.append(
                    f"**No response to an interaction.** The screen changed "
                    f"at {prior[-1].t0:.2f}s and then nothing moved at all "
                    f"for {ev.duration:.1f}s - no spinner, no repaint. "
                    f"Consistent with a click that was registered visually "
                    f"but produced no further effect."
                )

    if not any(e.kind in INTERACTIONS for e in events):
        # Distinguish the two very different reasons a timeline comes back
        # thin. Telling a user their frozen recording "may be all motion" is
        # worse than saying nothing.
        moving = float((sig.changed_frac > 0.01).mean())
        if moving < 0.02:
            out.append(
                f"**Nothing happened.** Across all {len(sig)} frames the "
                f"screen barely changed. Either the recording caught none of "
                f"the bug, or the UI was frozen for its entire "
                f"{sig.duration:.1f}s."
            )
        elif moving > 0.6:
            out.append(
                "**Continuously in motion.** Most frames differ from the one "
                "before, so there are no quiet stretches to segment against. "
                "This is normal for video playback or a constant animation, "
                "and it means the timeline below is thin by nature - read "
                "the key frames directly."
            )
        else:
            out.append(
                "**No discrete UI changes detected.** Things moved, but "
                "nothing crossed the threshold for a distinct event. The "
                "change may be subtler than the detection floor."
            )

    # One finding per underlying problem. Refinement can split a noisy run
    # into fragments that each generate the same sentence.
    seen, unique = set(), []
    for note in out:
        if note not in seen:
            seen.add(note)
            unique.append(note)
    return unique


def select_keyframes(events: List[Event], sig: Signals,
                     budget: int = 12) -> List[Tuple[float, str, Optional[tuple]]]:
    """
    Choose the fewest frames that let a reader reconstruct what happened.

    Every sustained event contributes its boundaries, because the whole
    point is showing the state either side of a change. Instant events get a
    before/after pair for the same reason. Priority decides what survives
    the budget when a recording is busy.
    """
    rank = {"cut": 0, "busy-indicator": 1, "region-change": 2,
            "static": 3, "local-change": 4, "scroll": 5, "pointer": 6,
            "flash": 2}
    cand: List[Tuple[int, float, str, Optional[tuple]]] = []
    eps = 1.5 / sig.fps

    for ev in events:
        r = rank.get(ev.kind, 7)
        if ev.instant:
            # The frame before carries the same region, so the pair reads as
            # a before/after of one place rather than two unrelated shots.
            cand.append((r, max(0.0, ev.t0 - eps),
                         f"just before {ev.kind} at {ev.t0:.2f}s", ev.region))
            cand.append((r, ev.t0, f"{ev.kind} at {ev.t0:.2f}s", ev.region))
        else:
            cand.append((r, ev.t0, f"start of {ev.kind} ({ev.t0:.2f}s)", ev.region))
            cand.append((r, ev.t1, f"end of {ev.kind} ({ev.t1:.2f}s)", ev.region))

    cand.append((0, 0.0, "first frame", None))
    cand.append((0, sig.duration, "last frame", None))

    chosen: List[Tuple[float, str, Optional[tuple]]] = []
    for _, t, label, region in sorted(cand, key=lambda c: (c[0], c[1])):
        if len(chosen) >= budget:
            break
        if any(abs(t - u) < eps for u, _, _ in chosen):
            continue
        chosen.append((t, label, region))
    return sorted(chosen)


def _crop(frame, region: Optional[tuple]):
    """Magnified crop of the changed region, or None if not worth one."""
    if region is None:
        return None
    x0, y0, x1, y1 = region
    w_frac, h_frac = x1 - x0, y1 - y0
    if w_frac <= 0 or h_frac <= 0 or w_frac * h_frac > CROP_MAX_AREA:
        return None

    padx, pady = w_frac * CROP_PAD, h_frac * CROP_PAD
    x0, x1 = x0 - padx, x1 + padx
    y0, y1 = y0 - pady, y1 + pady
    # Widen anything very thin -- a 2px-tall strip magnified alone gives the
    # reader nothing to judge alignment against.
    if x1 - x0 < CROP_MIN_SPAN:
        mid = (x0 + x1) / 2
        x0, x1 = mid - CROP_MIN_SPAN / 2, mid + CROP_MIN_SPAN / 2
    if y1 - y0 < CROP_MIN_SPAN:
        mid = (y0 + y1) / 2
        y0, y1 = mid - CROP_MIN_SPAN / 2, mid + CROP_MIN_SPAN / 2

    h, w = frame.shape[:2]
    px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
    px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
    if px1 - px0 < 8 or py1 - py0 < 8:
        return None

    crop = frame[py0:py1, px0:px1]
    cw = px1 - px0
    if cw < CROP_TARGET_W:
        scale = min(CROP_TARGET_W / cw, 4.0)
        crop = cv2.resize(crop, (int(cw * scale), int((py1 - py0) * scale)),
                          interpolation=cv2.INTER_CUBIC)
    elif cw > 900:
        scale = 900 / cw
        crop = cv2.resize(crop, (900, int((py1 - py0) * scale)),
                          interpolation=cv2.INTER_AREA)
    return crop


def extract_frames(path: str, stamps: List[Tuple[float, str, Optional[tuple]]],
                   outdir: str) -> List[Tuple[float, str, str, Optional[str]]]:
    """Write the chosen frames as JPEGs. Returns (t, label, relpath, croppath)."""
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    written: List[Tuple[float, str, str, Optional[str]]] = []

    for t, label, region in stamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
        if not ok:
            continue

        # Crop from the full-resolution frame, before any downscaling --
        # magnifying an already-shrunk frame just magnifies the blur.
        close = _crop(frame, region)

        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            scale = FRAME_WIDTH / w
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        stem = f"t{t:07.3f}".replace(".", "_", 1)
        cv2.imwrite(os.path.join(outdir, stem + ".jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY])

        crop_rel = None
        if close is not None:
            cv2.imwrite(os.path.join(outdir, stem + "_zoom.jpg"), close,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            crop_rel = f"frames/{stem}_zoom.jpg"
        written.append((t, label, f"frames/{stem}.jpg", crop_rel))
    cap.release()
    return written


def _fmt_span(ev: Event) -> str:
    if ev.instant:
        return f"`{ev.t0:.2f}s`"
    return f"`{ev.t0:.2f}s - {ev.t1:.2f}s`"


READING_NOTE = """\
## How to read this

The timeline is produced by per-frame pixel analysis, not by a model. It
reports *what changed, where, and for how long* -- it does not know what
any of it means. Names like `busy-indicator` describe a visual pattern
(something animating in one small place while everything else holds still),
not a diagnosis.

Where a close-up follows a frame, it is the region that changed, magnified
from the full-resolution video. Judge small visual defects -- alignment,
spacing, clipping, wrong state -- from the close-up, not the full frame.

To turn this into an explanation, read the key frames against the timeline.
The frames tell you what the UI is; the timeline tells you when it moved.
An event with no matching frame still happened -- the frame budget is
limited, and timings are exact to about one frame.
"""


def render(video: str, sig: Signals, events: List[Event], notes: List[str],
           frames: List[Tuple[float, str, str, Optional[str]]],
           transcript: Optional[List[dict]] = None) -> str:
    """Assemble the Markdown report."""
    name = os.path.basename(video)
    audio = f"{len(transcript)} speech segments" if transcript else "no speech"
    lines = [
        f"# Screen recording: {name}",
        "",
        f"**{sig.duration:.1f}s · {sig.width}x{sig.height} · "
        f"{sig.fps:.0f}fps · {len(sig)} frames analysed · {audio}**",
        "",
    ]

    lines += ["## What stands out", ""]
    if notes:
        lines += [f"- {n}" for n in notes]
    else:
        lines.append("- Nothing matched the built-in suspicious patterns. "
                     "The timeline below is still a complete record.")
    lines.append("")

    lines += ["## Timeline", "",
              "| time | event | what changed |",
              "| --- | --- | --- |"]
    for ev in events:
        detail = ev.detail.replace("|", "\\|")
        lines.append(f"| {_fmt_span(ev)} | `{ev.kind}` | {detail} |")
    lines.append("")

    if transcript:
        lines += ["## Narration", ""]
        for seg in transcript:
            lines.append(f"- `{seg['start']:.2f}s - {seg['end']:.2f}s` "
                         f"{seg['text'].strip()}")
        lines.append("")

    lines += ["## Key frames", ""]
    for t, label, rel, zoom in frames:
        lines += [f"### `{t:.2f}s` — {label}", "", f"![{label}]({rel})", ""]
        if zoom:
            lines += [f"Close-up of the region that changed:", "",
                      f"![{label} close-up]({zoom})", ""]

    lines += [READING_NOTE]
    return "\n".join(lines)
