"""
Render a motion clip whose animation is known exactly.

Used to check that `replicate` recovers a curve it was never told about:
the circle below scales on a cubic ease-out over 600ms, so the report
should come back naming `cubic-out` with a low error.

  python3 examples/make_motion_sample.py examples/sample_motion.mp4
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

W, H, FPS, DUR = 960, 600, 60, 3.0
START, LENGTH = 1.0, 0.60          # animation starts at 1.0s, runs 600ms
BG, FG = (18, 18, 20), (240, 240, 245)
R_MIN, R_MAX = 12, 210


def ease_out_cubic(u):
    return 1 - (1 - u) ** 3


def frame(i):
    t = i / FPS
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if t < START:
        u = 0.0
    elif t < START + LENGTH:
        u = ease_out_cubic((t - START) / LENGTH)
    else:
        u = 1.0
    r = R_MIN + (R_MAX - R_MIN) * u
    cx, cy = W / 2, H / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FG)
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_motion.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(DUR * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            frame(i).save(Path(tmp) / f"f{i:05d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "18", str(out)], check=True)
    print(f"wrote {out} - circle scales on cubic ease-out over "
          f"{LENGTH*1000:.0f}ms starting at {START}s")


if __name__ == "__main__":
    main()
