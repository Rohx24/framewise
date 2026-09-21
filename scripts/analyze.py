#!/usr/bin/env python3
"""
video-repro: turn a screen recording into a readable timeline.

    python3 scripts/analyze.py bug.mov -o repro/

Writes repro/timeline.md (the report), repro/frames/ (key frames) and
repro/events.json (machine-readable). Everything runs locally; no API keys
and no network beyond a one-time Whisper model download if the recording
has speech and transcription is enabled.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import events as ev_mod       # noqa: E402
import report as rep          # noqa: E402
import signals as sig_mod     # noqa: E402


def has_audio(path: str) -> bool:
    """True if the container carries at least one audio stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def transcribe(path: str, model_name: str):
    """
    Whisper transcript with timestamps, or None if unavailable.

    Narration is disproportionately valuable in a bug report -- people say
    "and now it just hangs" out loud -- but it must never be a hard
    dependency, so every failure here degrades to None rather than raising.
    """
    try:
        import whisper
    except ImportError:
        print("  whisper not installed; skipping narration", file=sys.stderr)
        return None
    try:
        model = whisper.load_model(model_name)
        result = model.transcribe(path, verbose=None)
        segs = [s for s in result.get("segments", []) if s.get("text", "").strip()]
        return segs or None
    except Exception as exc:                       # noqa: BLE001
        print(f"  transcription failed ({exc}); continuing without it",
              file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="video-repro",
        description="Turn a screen recording into a timeline a model can read.",
    )
    ap.add_argument("video", help="path to the recording (.mov/.mp4/.webm/...)")
    ap.add_argument("-o", "--out", default="repro",
                    help="output directory (default: repro)")
    ap.add_argument("--frames", type=int, default=12,
                    help="max key frames to extract (default: 12)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only analyse the first N seconds")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip narration transcription")
    ap.add_argument("--whisper-model", default="base",
                    help="whisper model size (default: base)")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"error: no such file: {args.video}", file=sys.stderr)
        return 1

    print(f"reading {args.video} ...", file=sys.stderr)
    sig = sig_mod.extract(args.video, max_seconds=args.max_seconds)
    print(f"  {len(sig)} frames, {sig.duration:.1f}s, "
          f"{sig.width}x{sig.height} @ {sig.fps:.0f}fps", file=sys.stderr)

    evs = ev_mod.coalesce(ev_mod.detect(sig))
    print(f"  {len(evs)} events", file=sys.stderr)

    notes = rep.findings(evs, sig)
    stamps = rep.select_keyframes(evs, sig, budget=args.frames)

    os.makedirs(args.out, exist_ok=True)
    frames = rep.extract_frames(args.video, stamps,
                                os.path.join(args.out, "frames"))
    print(f"  {len(frames)} key frames", file=sys.stderr)

    transcript = None
    if not args.no_audio and has_audio(args.video):
        print("  transcribing narration ...", file=sys.stderr)
        transcript = transcribe(args.video, args.whisper_model)

    md = rep.render(args.video, sig, evs, notes, frames, transcript)
    md_path = os.path.join(args.out, "timeline.md")
    with open(md_path, "w") as fh:
        fh.write(md)

    with open(os.path.join(args.out, "events.json"), "w") as fh:
        json.dump({
            "video": os.path.basename(args.video),
            "duration": round(sig.duration, 3),
            "fps": round(sig.fps, 3),
            "size": [sig.width, sig.height],
            "findings": notes,
            "events": [{"t0": round(e.t0, 3), "t1": round(e.t1, 3),
                        "kind": e.kind, "detail": e.detail,
                        "region": [round(v, 4) for v in e.region]
                        if e.region else None}
                       for e in evs],
            "frames": [{"t": round(t, 3), "label": lab, "path": p}
                       for t, lab, p in frames],
        }, fh, indent=2)

    print(f"\nwrote {md_path}", file=sys.stderr)
    if notes:
        print("\n" + "\n".join(f"  * {n}" for n in notes), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
