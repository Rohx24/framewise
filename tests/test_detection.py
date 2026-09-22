#!/usr/bin/env python3
"""
Regression tests for the detectors.

Every case here is a bug that actually shipped during development, so each
one is worth keeping:

  * the spinner vanished when deltas were measured on a downscaled frame
  * the click was swallowed by the pointer run around it
  * a compressed re-encode moved the reported hang five seconds late
  * ...and reported it four times
  * and the adaptive noise floor let a busy cell threshold itself away,
    turning a hung spinner into "the screen was frozen"

Run:  python3 tests/test_detection.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import events as E        # noqa: E402
import report as R        # noqa: E402
import signals as S       # noqa: E402

SAMPLE = os.path.join(ROOT, "examples", "sample_bug.mp4")
# Ground truth baked into examples/make_sample.py
TRUE_SPIN_START, TRUE_SPIN_END = 3.30, 12.0
TRUE_CLICK, TRUE_SCROLL_START = 3.00, 0.80
TOL = 0.20

FAILED: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(label)


def analyse(path: str):
    sig = S.extract(path)
    evs = E.coalesce(E.detect(sig))
    return sig, evs, R.findings(evs, sig)


def ensure_sample() -> None:
    if not os.path.isfile(SAMPLE):
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "examples", "make_sample.py"),
                        SAMPLE], check=True)


def reencode(src: str, dst: str, crf: int) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-c:v", "libx264", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", dst], check=True)


def synth(dst: str, filt: str, dur: str = "4") -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", filt, "-t", dur, "-r", "30",
                    "-pix_fmt", "yuv420p", dst], check=True)


def spins(evs):
    return [e for e in evs if e.kind == "busy-indicator"]


def hangs(notes) -> int:
    """
    Count hang findings.

    Matching on the bare word "hang" looks fine and is not: "changed"
    contains it, so every timeline that mentioned a change scored a hang.
    """
    return sum(n.lstrip("*").lower().startswith("likely hang") for n in notes)


def test_clean() -> None:
    print("clean recording")
    _, evs, notes = analyse(SAMPLE)
    sp = spins(evs)
    check("exactly one busy-indicator", len(sp) == 1, f"got {len(sp)}")
    if sp:
        check("hang onset is accurate",
              abs(sp[0].t0 - TRUE_SPIN_START) < TOL, f"t0={sp[0].t0:.2f}")
        check("hang runs to end of recording",
              abs(sp[0].t1 - TRUE_SPIN_END) < 0.5, f"t1={sp[0].t1:.2f}")
    check("click survives the surrounding pointer motion",
          any(e.kind in ("local-change", "region-change")
              and abs(e.t0 - TRUE_CLICK) < TOL for e in evs))
    check("scroll detected",
          any(e.kind == "scroll" and abs(e.t0 - TRUE_SCROLL_START) < 0.3
              for e in evs))
    check("spinning UI is never called frozen",
          not any(e.kind == "static" and e.t0 > TRUE_SPIN_START for e in evs))
    check("a hang is reported", hangs(notes) >= 1)
    check("the hang is reported once", hangs(notes) == 1)


def test_compressed(tmp: str) -> None:
    print("heavily compressed re-encode (crf 40)")
    dst = os.path.join(tmp, "noisy.mp4")
    reencode(SAMPLE, dst, 40)
    _, evs, notes = analyse(dst)
    sp = spins(evs)
    check("still exactly one busy-indicator", len(sp) == 1, f"got {len(sp)}")
    if sp:
        check("onset still accurate under noise",
              abs(sp[0].t0 - TRUE_SPIN_START) < TOL, f"t0={sp[0].t0:.2f}")
    check("hang still reported exactly once", hangs(notes) == 1)


def test_static(tmp: str) -> None:
    print("entirely static recording")
    dst = os.path.join(tmp, "static.mp4")
    synth(dst, "color=c=white:s=640x480")
    _, evs, notes = analyse(dst)
    check("no busy-indicator invented", not spins(evs))
    check("no hang claimed", hangs(notes) == 0)
    check("says nothing happened",
          any("nothing happened" in n.lower() for n in notes))
    # Phase correlation returns arbitrary offsets on a featureless frame,
    # which invented a scroll spanning the whole recording.
    check("no phantom scroll on a blank screen",
          not any(e.kind == "scroll" for e in evs))


def test_transient(tmp: str) -> None:
    print("three-frame flash of a different state")
    dst = os.path.join(tmp, "flash.mp4")
    # Red then green of near-identical luminance: invisible in grayscale,
    # which is how error -> success used to pass through undetected.
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=0xF5F6F8:s=640x480:d=4:r=30",
         "-f", "lavfi", "-i", "color=c=0xE04B4A:s=400x60:d=4:r=30",
         "-f", "lavfi", "-i", "color=c=0x20A06E:s=400x60:d=4:r=30",
         "-filter_complex",
         "[0][1]overlay=120:100:enable='between(t,2.0,2.1)'[a];"
         "[a][2]overlay=120:100:enable='gt(t,2.1)'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst],
        check=True)
    _, evs, notes = analyse(dst)
    tr = [e for e in evs if e.kind == "transient"]
    check("the flash is detected", len(tr) == 1, f"got {len(tr)}")
    if tr:
        check("flash is timed correctly", abs(tr[0].t0 - 2.0) < 0.15,
              f"t0={tr[0].t0:.2f}")
        check("flash duration is about 0.1s", tr[0].duration < 0.35,
              f"{tr[0].duration:.2f}s")
    check("a flash finding is reported",
          any(n.lstrip("*").lower().startswith("a state flashed") for n in notes))


def test_continuous_motion(tmp: str) -> None:
    print("continuous full-frame motion")
    dst = os.path.join(tmp, "motion.mp4")
    synth(dst, "mandelbrot=s=480x320:rate=30")
    _, evs, notes = analyse(dst)
    check("no hang claimed on constant motion", hangs(notes) == 0)


def test_signal_resolution() -> None:
    print("small-motion sensitivity")
    sig = S.extract(SAMPLE)
    # The spinner is ~20px. If deltas were measured after downscaling it
    # would read as zero here, which is the bug this asserts against.
    mid = len(sig) // 2
    check("a 20px spinner registers mid-hang",
          sig.changed_px[mid] > 0, f"changed_px={sig.changed_px[mid]}")
    check("truly static frames register zero",
          sig.changed_px[5] == 0, f"changed_px={sig.changed_px[5]}")


def test_easing_fit() -> None:
    print("easing identification")
    import numpy as np
    import measure as M
    x = np.linspace(0, 1, 24)
    cases = {"cubic-out": 1 - (1 - x) ** 3, "linear": x,
             "quart-in": x ** 4, "expo-out": 1 - 2 ** (-10 * x)}
    for expect, curve in cases.items():
        got = M.fit_easing(curve)
        check(f"{expect} identified", got and got[0][0] == expect,
              f"got {got[0][0] if got else None}")
        check(f"{expect} fits closely", got and got[0][1] < 0.05,
              f"rmse {got[0][1]:.3f}" if got else "no fit")
    # A curve no standard easing matches must report a poor fit rather than
    # confidently naming the least-bad one.
    bounce = np.abs(np.sin(x * np.pi * 3)) * (1 - x) + x
    got = M.fit_easing(bounce)
    check("a non-tween curve reports a poor fit", got and got[0][1] > 0.09,
          f"rmse {got[0][1]:.3f}" if got else "no fit")


def test_segmentation(tmp: str) -> None:
    print("state / transition segmentation")
    import replicate as R
    # Still, then a hard change, then still again.
    dst = os.path.join(tmp, "states.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "color=c=0x202020:s=480x320:d=4:r=30",
         "-f", "lavfi", "-i", "color=c=0xE0E0E0:s=300x200:d=4:r=30",
         "-filter_complex", "[0][1]overlay=90:60:enable='gt(t,2)'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", dst],
        check=True)
    sig = S.extract(dst)
    segs = R.dedupe_states(sig, R.segment(sig))
    states = [g for g in segs if g["kind"] == "state"]
    check("both resting states found", len(states) == 2, f"got {len(states)}")
    check("the change is not called a state",
          all(not (g["t0"] < 2.05 < g["t1"]) for g in states))


def test_motion_recovery() -> None:
    print("recovering a known animation")
    import numpy as np
    import measure as M
    import replicate as R
    clip = os.path.join(ROOT, "examples", "sample_motion.mp4")
    if not os.path.isfile(clip):
        subprocess.run([sys.executable,
                        os.path.join(ROOT, "examples", "make_motion_sample.py"),
                        clip], check=True)
    # Ground truth baked into make_motion_sample.py: a circle scaling on a
    # cubic ease-out, 600ms, starting at 1.0s.
    sig = S.extract(clip)
    segs = R.dedupe_states(sig, R.segment(sig))
    moves = [g for g in segs if g["kind"] == "transition"]
    check("one transition found", len(moves) == 1, f"got {len(moves)}")
    if not moves:
        return
    mv = moves[0]
    check("transition starts on time", abs(mv["t0"] - 1.0) < 0.12,
          f"t0={mv['t0']:.2f}")
    # An ease-out's last frames move sub-pixel, so some tail is always lost.
    # Losing more than a third of it means the hysteresis has regressed and
    # the curve will read as linear.
    dur = mv["t1"] - mv["t0"]
    check("most of the duration is captured", 0.40 <= dur <= 0.75,
          f"{dur:.2f}s of 0.60s")

    idxs = sorted({sig.index_at(t)
                   for t in np.linspace(mv["t0"], mv["t1"], 24)})
    got = R.grab(clip, idxs)
    region = R.effect_region(sig, mv["i0"], mv["i1"])
    h, w = got[idxs[0]].shape[:2]
    x0, y0 = int(region[0] * w), int(region[1] * h)
    x1, y1 = int(region[2] * w), int(region[3] * h)
    crops = [got[i][y0:y1, x0:x1] for i in idxs if i in got]
    met = M.measure_frames(crops, [float(sig.t[i]) for i in idxs if i in got])
    fits = M.fit_easing([m.scale for m in met])
    check("scale grows from near nothing", met[0].scale < 0.1,
          f"start scale {met[0].scale:.3f}")
    check("a decelerating curve is identified",
          fits and fits[0][0] in ("ease-out", "cubic-out", "quad-out",
                                  "quart-out", "expo-out", "circ-out"),
          f"got {fits[0][0] if fits else None}")
    check("and it fits closely", fits and fits[0][1] < 0.06,
          f"rmse {fits[0][1]:.3f}" if fits else "no fit")
    check("linear is rejected",
          fits and dict(fits)["linear"] > fits[0][1] * 2)


def main() -> int:
    ensure_sample()
    with tempfile.TemporaryDirectory() as tmp:
        test_signal_resolution()
        test_clean()
        test_compressed(tmp)
        test_static(tmp)
        test_transient(tmp)
        test_continuous_motion(tmp)
        test_easing_fit()
        test_segmentation(tmp)
        test_motion_recovery()
    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
