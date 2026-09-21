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
                     budget: int = 12) -> List[Tuple[float, str]]:
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
    cand: List[Tuple[int, float, str]] = []
    eps = 1.5 / sig.fps

    for ev in events:
        r = rank.get(ev.kind, 7)
        if ev.instant:
            cand.append((r, max(0.0, ev.t0 - eps), f"just before {ev.kind} at {ev.t0:.2f}s"))
            cand.append((r, ev.t0, f"{ev.kind} at {ev.t0:.2f}s"))
        else:
            cand.append((r, ev.t0, f"start of {ev.kind} ({ev.t0:.2f}s)"))
            cand.append((r, ev.t1, f"end of {ev.kind} ({ev.t1:.2f}s)"))

    cand.append((0, 0.0, "first frame"))
    cand.append((0, sig.duration, "last frame"))

    chosen: List[Tuple[float, str]] = []
    for _, t, label in sorted(cand, key=lambda c: (c[0], c[1])):
        if len(chosen) >= budget:
            break
        if any(abs(t - u) < eps for u, _ in chosen):
            continue
        chosen.append((t, label))
    return sorted(chosen)


def extract_frames(path: str, stamps: List[Tuple[float, str]],
                   outdir: str) -> List[Tuple[float, str, str]]:
    """Write the chosen frames as JPEGs. Returns (t, label, relpath)."""
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    written: List[Tuple[float, str, str]] = []

    for t, label in stamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > FRAME_WIDTH:
            scale = FRAME_WIDTH / w
            frame = cv2.resize(frame, (FRAME_WIDTH, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        name = f"t{t:07.3f}.jpg".replace(".", "_", 1)
        cv2.imwrite(os.path.join(outdir, name), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY])
        written.append((t, label, f"frames/{name}"))
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

To turn this into an explanation, read the key frames against the timeline.
The frames tell you what the UI is; the timeline tells you when it moved.
An event with no matching frame still happened -- the frame budget is
limited, and timings are exact to about one frame.
"""


def render(video: str, sig: Signals, events: List[Event], notes: List[str],
           frames: List[Tuple[float, str, str]],
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
    for t, label, rel in frames:
        lines += [f"### `{t:.2f}s` — {label}", "", f"![{label}]({rel})", ""]

    lines += [READING_NOTE]
    return "\n".join(lines)
