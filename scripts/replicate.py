#!/usr/bin/env python3
"""
replicate: take a video of an interface apart so it can be rebuilt.

    python3 scripts/replicate.py clip.mov -o rebuild/
    python3 scripts/replicate.py clip.mov --from 6.3 --to 7.1 -o rebuild/

A recording of an interface is two things interleaved: states it rests in,
and transitions between them. They need completely different treatment.

A state is a layout. One clean full-resolution frame answers almost every
question about it, because a model can read type, spacing and colour
straight off the pixels.

A transition is motion, and a still frame is the one thing that cannot
describe it. Those get sampled densely and measured -- scale, position,
brightness, blur, colour separation, per frame, with an easing curve
fitted. That is the difference between "a bubble expands" and "scale
0.08 to 1.0 over 520ms on cubic-bezier(0.16, 1, 0.30, 1)".

Where `watch` answers "what went wrong", this answers "what is it, and
what exactly is it doing".
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure as M          # noqa: E402
import signals as sig_mod    # noqa: E402

SHEET_COL_W = 520
PAD = 0.12                   # region padding, fraction of its own size
MOVE_HIGH = 0.01             # change needed to declare motion has begun
MOVE_LOW = 0.0015            # change below which motion has truly finished
STATE_STABLE = MOVE_LOW      # kept for readability below
STATE_MIN = 0.25             # seconds before a rest counts as a state
MOVE_MIN = 0.10              # seconds before motion counts as a transition
STATE_SAME = 4.0             # thumbnail distance below which states match
STATE_MAX_W = 1600           # full frames are saved no wider than this


def segment(sig) -> List[dict]:
    """
    Split the recording into states the interface rests in and the
    transitions between them.

    Two thresholds, not one. An ease-out spends most of its duration
    barely moving, so a single "is it still?" test clips the slow tail and
    keeps only the fast opening -- and a curve truncated to its first third
    looks linear no matter what it really was. Motion has to cross a high
    bar to start and fall under a much lower one to be called finished.
    """
    cf = sig.changed_frac
    n = len(sig)
    moving = cf > MOVE_HIGH

    # Grow each burst outwards while anything at all is still changing.
    live = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        a = i
        while a > 0 and cf[a - 1] > MOVE_LOW:
            a -= 1
        b = i
        while b + 1 < n and cf[b + 1] > MOVE_LOW:
            b += 1
        live[a:b + 1] = True
        i = b + 1

    out, i = [], 0
    while i < n:
        j = i
        while j < n and live[j] == live[i]:
            j += 1
        t0, t1 = float(sig.t[i]), float(sig.t[min(j, n - 1)])
        span = t1 - t0
        if live[i] and span >= MOVE_MIN:
            out.append({"kind": "transition", "t0": t0, "t1": t1,
                        "i0": i, "i1": j - 1})
        elif not live[i] and span >= STATE_MIN:
            out.append({"kind": "state", "t0": t0, "t1": t1,
                        "i0": i, "i1": j - 1})
        i = j

    # A rest shorter than STATE_MIN is not a rest, it is a beat inside one
    # movement. Without merging across those, a single second-long
    # transition came back as four fragments, each too short to fit.
    merged: List[dict] = []
    for seg in out:
        if (merged and seg["kind"] == "transition"
                and merged[-1]["kind"] == "transition"
                and seg["t0"] - merged[-1]["t1"] < STATE_MIN):
            merged[-1]["t1"] = seg["t1"]
            merged[-1]["i1"] = seg["i1"]
        else:
            merged.append(seg)
    return merged


def dedupe_states(sig, segs: List[dict]) -> List[dict]:
    """Drop states that look the same as one already reported."""
    kept, seen = [], []
    for s in segs:
        if s["kind"] != "state":
            kept.append(s)
            continue
        mid = (s["i0"] + s["i1"]) // 2
        th = sig.thumb[mid].astype(np.float32)
        if any(float(np.abs(th - o).mean()) < STATE_SAME for o in seen):
            continue
        seen.append(th)
        kept.append(s)
    return kept


def auto_window(sig, max_len: float = 2.5) -> Tuple[float, float]:
    """
    Pick the most eventful stretch when the user has not named one.

    Effects are bursts. The longest run of sustained change is almost
    always the transition worth measuring, and guessing it saves the user
    scrubbing for timestamps.
    """
    cf = sig.changed_frac
    live = cf > max(0.01, float(np.percentile(cf[cf > 0], 60)) if (cf > 0).any() else 0.01)
    best, run_start, best_score = None, None, 0.0
    for i, on in enumerate(live):
        if on and run_start is None:
            run_start = i
        elif not on and run_start is not None:
            score = float(cf[run_start:i].sum())
            if score > best_score:
                best, best_score = (run_start, i), score
            run_start = None
    if run_start is not None:
        score = float(cf[run_start:].sum())
        if score > best_score:
            best = (run_start, len(live))
    if best is None:
        return 0.0, min(sig.duration, max_len)
    t0, t1 = float(sig.t[best[0]]), float(sig.t[min(best[1], len(sig) - 1)])
    return t0, min(t1, t0 + max_len)


def effect_region(sig, i0: int, i1: int) -> Tuple[float, float, float, float]:
    """Union of everything that moved in the window, padded."""
    boxes = sig.bbox[i0:i1 + 1]
    live = boxes[(boxes[:, 2] - boxes[:, 0]) > 0]
    if len(live) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    x0, y0 = float(live[:, 0].min()), float(live[:, 1].min())
    x1, y1 = float(live[:, 2].max()), float(live[:, 3].max())
    px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
    return (max(0.0, x0 - px), max(0.0, y0 - py),
            min(1.0, x1 + px), min(1.0, y1 + py))


def grab(path: str, wanted) -> dict:
    """
    Decode once, keeping only the frame indices asked for.

    One pass for every frame the whole report needs. Seeking per frame is
    both slower and wrong on the variable-rate files screen recorders
    produce.
    """
    wanted = set(wanted)
    cap = cv2.VideoCapture(path)
    out, i = {}, 0
    top = max(wanted) if wanted else -1
    while i <= top:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = fr
        i += 1
    cap.release()
    return out


def contact_sheet(crops: List[np.ndarray], cols: int = 4) -> np.ndarray:
    """One image of the whole motion, which reads far better than a list."""
    scaled = []
    for c in crops:
        h, w = c.shape[:2]
        s = SHEET_COL_W / w
        scaled.append(cv2.resize(c, (SHEET_COL_W, max(1, int(h * s))),
                                 interpolation=cv2.INTER_AREA))
    ch = max(s.shape[0] for s in scaled)
    rows = (len(scaled) + cols - 1) // cols
    sheet = np.zeros((rows * ch, cols * SHEET_COL_W, 3), dtype=np.uint8)
    for i, s in enumerate(scaled):
        r, c = divmod(i, cols)
        sheet[r * ch:r * ch + s.shape[0], c * SHEET_COL_W:(c + 1) * SHEET_COL_W] = s
    return sheet


def _curve(name: str, vals: List[float], unit: str = "") -> List[str]:
    v = np.asarray(vals, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return [f"- **{name}** — constant at {lo:.3g}{unit}, not animated."]
    fits = M.fit_easing(v)
    line = [f"- **{name}** — {v[0]:.3g}{unit} to {v[-1]:.3g}{unit} "
            f"(min {lo:.3g}, max {hi:.3g})"]
    if fits:
        best, rmse = fits[0]
        pts = M.EASINGS[best]
        quality = ("close fit" if rmse < 0.04 else
                   "rough fit" if rmse < 0.09 else "poor fit")
        line.append(f"  - best easing `{best}` = "
                    f"`cubic-bezier({pts[0]}, {pts[1]}, {pts[2]}, {pts[3]})` "
                    f"(rmse {rmse:.3f}, {quality})")
    return line


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="replicate",
        description="Take a video of an interface apart so it can be rebuilt.")
    ap.add_argument("video")
    ap.add_argument("--from", dest="t0", type=float, default=None,
                    help="only look at this window, seconds")
    ap.add_argument("--to", dest="t1", type=float, default=None)
    ap.add_argument("-o", "--out", default="rebuild")
    ap.add_argument("--frames", type=int, default=20,
                    help="frames sampled per transition (default 20)")
    ap.add_argument("--max-states", type=int, default=8)
    ap.add_argument("--max-transitions", type=int, default=6)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"error: no such file: {args.video}", file=sys.stderr)
        return 1

    print(f"reading {args.video} ...", file=sys.stderr)
    sig = sig_mod.extract(args.video,
                          max_seconds=(args.t1 + 0.5) if args.t1 else None)
    print(f"  {len(sig)} frames, {sig.duration:.1f}s", file=sys.stderr)

    segs = dedupe_states(sig, segment(sig))
    if args.t0 is not None or args.t1 is not None:
        lo = args.t0 if args.t0 is not None else 0.0
        hi = args.t1 if args.t1 is not None else sig.duration
        segs = [g for g in segs if g["t1"] >= lo and g["t0"] <= hi]
        # A hand-picked window that lands mid-transition still deserves
        # measuring, so fall back to treating the window itself as one.
        if not any(g["kind"] == "transition" for g in segs):
            segs.append({"kind": "transition", "t0": lo, "t1": hi,
                         "i0": sig.index_at(lo), "i1": sig.index_at(hi)})

    states = [g for g in segs if g["kind"] == "state"][:args.max_states]
    moves = [g for g in segs if g["kind"] == "transition"]
    moves.sort(key=lambda g: -float(sig.changed_frac[g["i0"]:g["i1"] + 1].sum()))
    moves = sorted(moves[:args.max_transitions], key=lambda g: g["t0"])
    print(f"  {len(states)} states, {len(moves)} transitions", file=sys.stderr)

    # Work out every frame index needed, then decode once.
    need, plan = set(), {}
    for k, st in enumerate(states):
        idx = (st["i0"] + st["i1"]) // 2
        plan[("state", k)] = [idx]
        need.add(idx)
    for k, mv in enumerate(moves):
        targets = np.linspace(mv["t0"], mv["t1"], args.frames)
        idxs = sorted({sig.index_at(t) for t in targets})
        plan[("move", k)] = idxs
        need.update(idxs)
    got = grab(args.video, need)

    os.makedirs(os.path.join(args.out, "states"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "motion"), exist_ok=True)

    lines = [f"# Rebuilding: {os.path.basename(args.video)}", "",
             f"**{sig.duration:.1f}s analysed — {len(states)} resting "
             f"state(s), {len(moves)} transition(s)**", "",
             "A state is a layout: read it off the full frame. A transition "
             "is motion: read it off the measured curves, because no still "
             "frame contains it.", ""]

    lines += ["## States", ""]
    for k, st in enumerate(states):
        idx = plan[("state", k)][0]
        fr = got.get(idx)
        if fr is None:
            continue
        h, w = fr.shape[:2]
        if w > STATE_MAX_W:
            sc = STATE_MAX_W / w
            fr = cv2.resize(fr, (STATE_MAX_W, int(h * sc)),
                            interpolation=cv2.INTER_AREA)
        rel = f"states/state{k:02d}.jpg"
        cv2.imwrite(os.path.join(args.out, rel), fr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        pal = M.palette(fr)
        lines += [f"### State {k} — held {st['t0']:.2f}s to {st['t1']:.2f}s "
                  f"({st['t1']-st['t0']:.2f}s)", "",
                  f"![state {k}]({rel})", "",
                  "- Palette: " + ", ".join(f"`{hx}` ({sh:.0%})"
                                            for hx, sh in pal[:5]), ""]

    lines += ["## Transitions", ""]
    for k, mv in enumerate(moves):
        idxs = plan[("move", k)]
        frames = [got[i] for i in idxs if i in got]
        if len(frames) < 4:
            continue
        times = [float(sig.t[i]) for i in idxs if i in got]
        region = effect_region(sig, mv["i0"], mv["i1"])
        H, W = frames[0].shape[:2]
        x0, y0 = int(region[0] * W), int(region[1] * H)
        x1, y1 = max(int(region[2] * W), x0 + 8), max(int(region[3] * H), y0 + 8)
        crops = [f[y0:y1, x0:x1] for f in frames]
        metrics = M.measure_frames(crops, times)

        sheet_rel = f"motion/transition{k:02d}_sheet.jpg"
        cv2.imwrite(os.path.join(args.out, sheet_rel), contact_sheet(crops),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        fdir = os.path.join(args.out, "motion", f"t{k:02d}")
        os.makedirs(fdir, exist_ok=True)
        for i, (c, t) in enumerate(zip(crops, times)):
            cv2.imwrite(os.path.join(fdir, f"{i:02d}_t{t:.3f}.jpg"), c,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        dur = (times[-1] - times[0]) * 1000
        lines += [f"### Transition {k} — {mv['t0']:.2f}s to {mv['t1']:.2f}s "
                  f"({dur:.0f}ms)", "",
                  f"Region [{region[0]:.0%},{region[1]:.0%} to "
                  f"{region[2]:.0%},{region[3]:.0%}] of the screen, "
                  f"{len(frames)} frames sampled.", "",
                  f"![transition {k}]({sheet_rel})", "",
                  f"Frame-by-frame crops: `motion/t{k:02d}/`", ""]
        lines += _curve("Scale (subject area, 1.0 = largest seen)",
                        [m.scale for m in metrics])
        lines += _curve("Centre X", [m.cx for m in metrics])
        lines += _curve("Centre Y", [m.cy for m in metrics])
        lines += _curve("Brightness", [m.luma for m in metrics])
        lines += _curve("Coverage (lit fraction)", [m.coverage for m in metrics])
        lines += _curve("Sharpness (rising = blur clearing)",
                        [m.sharpness for m in metrics])

        # Said once, at the end, rather than repeated under every curve.
        fits = [M.fit_easing(v)[0][1] for v in
                ([m.scale for m in metrics], [m.luma for m in metrics],
                 [m.coverage for m in metrics]) if M.fit_easing(v)]
        if fits and min(fits) >= 0.09:
            lines += ["", "> No standard easing fits any channel well "
                      "(best rmse %.2f). This is not a single tween — "
                      "expect staged keyframes, a spring, or per-frame "
                      "shader work." % min(fits)]

        fr_v = [m.fringe for m in metrics]
        pal = M.palette(crops[len(crops) // 2])
        lines += ["", "- Palette mid-transition: "
                  + ", ".join(f"`{hx}` ({sh:.0%})" for hx, sh in pal[:4])]
        if max(fr_v) > 18:
            lines.append(f"- Edge colour separation peaks at {max(fr_v):.1f} — "
                         "strong red/blue splitting on edges is chromatic "
                         "dispersion, which means a refraction or displacement "
                         "shader, not CSS.")
        elif max(fr_v) > 8:
            lines.append(f"- Edge colour separation peaks at {max(fr_v):.1f} — "
                         "mild; could be a subtle aberration pass, could be "
                         "compression.")
        else:
            lines.append("- No meaningful colour fringing; plain compositing.")

        lines += ["", "| # | t | scale | centre | bright | coverage | sharp |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for i, m in enumerate(metrics):
            lines.append(f"| {i} | {m.t:.3f}s | {m.scale:.3f} | "
                         f"{m.cx:.2f},{m.cy:.2f} | {m.luma:.0f} | "
                         f"{m.coverage:.3f} | {m.sharpness:.0f} |")
        lines.append("")

    lines += ["## How to read this", "",
              "The numbers describe what the pixels did. They do not name a "
              "technique: a scale curve looks identical whether it came from "
              "CSS, a spring, or a vertex shader. Use the curves for timing "
              "and magnitude, the state frames for layout and colour, and "
              "the contact sheet for the look of the motion.", "",
              "When you write the implementation, say which parts you "
              "measured and which you inferred. A poor easing fit is real "
              "information -- it usually means staged keyframes or per-frame "
              "shader work rather than one tween.", ""]

    with open(os.path.join(args.out, "rebuild.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nwrote {os.path.join(args.out, 'rebuild.md')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
