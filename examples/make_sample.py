"""
Generate a synthetic screen recording of a classic hung-save bug.

Used as a fixture for tuning the detectors and as a runnable demo:
  python3 examples/make_sample.py examples/sample_bug.mp4

Timeline it renders (12s @ 30fps):
  0.0-0.8   idle form
  0.8-1.6   user scrolls the form down
  1.6-3.0   cursor travels to the Save button
  3.0-3.2   button shows a pressed state
  3.3-12.0  spinner appears and never stops   <- the bug
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

W, H, FPS, DUR = 1280, 800, 30, 12.0
BG, CARD, INK, MUTED = (245, 246, 248), (255, 255, 255), (30, 34, 40), (150, 156, 165)
BLUE, BLUE_DARK, LINE = (44, 110, 225), (28, 78, 170), (226, 229, 234)
BTN = (980, 300, 1160, 348)   # Save button box in card-space


def lerp(a, b, u):
    u = max(0.0, min(1.0, u))
    return a + (b - a) * u


def draw_cursor(d, x, y):
    pts = [(x, y), (x, y + 17), (x + 4.5, y + 13), (x + 7.5, y + 19),
           (x + 10, y + 17.5), (x + 7, y + 12), (x + 12, y + 11.5)]
    d.polygon(pts, fill=(255, 255, 255), outline=(20, 20, 20))


def frame(i):
    t = i / FPS
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # chrome: title bar stays fixed while the card scrolls under it
    d.rectangle([0, 0, W, 56], fill=(255, 255, 255))
    d.line([0, 56, W, 56], fill=LINE)
    d.rectangle([28, 20, 150, 36], fill=(225, 228, 233))

    scroll = int(lerp(0, 120, (t - 0.8) / 0.8)) if t > 0.8 else 0
    top = 96 - scroll

    d.rounded_rectangle([80, top, 1200, top + 520], 10, fill=CARD, outline=LINE)
    d.rectangle([120, top + 36, 400, top + 56], fill=(210, 214, 220))
    for row in range(4):
        y = top + 100 + row * 64
        d.rectangle([120, y, 290, y + 14], fill=(215, 219, 225))
        d.rounded_rectangle([120, y + 24, 940, y + 60], 6,
                            fill=(250, 251, 252), outline=LINE)

    # Save button, pressed briefly at t=3.0
    bx0, by0, bx1, by1 = BTN[0], BTN[1] + top, BTN[2], BTN[3] + top
    pressed = 3.00 <= t < 3.20
    d.rounded_rectangle([bx0, by0, bx1, by1], 6,
                        fill=BLUE_DARK if pressed else BLUE)
    d.rectangle([bx0 + 62, by0 + 20, bx1 - 62, by1 - 20], fill=(255, 255, 255))

    # spinner: appears after the click and never resolves
    if t >= 3.30:
        cx, cy, r = bx0 - 40, (by0 + by1) // 2, 13
        ang = (t - 3.30) * 300 % 360
        d.arc([cx - r, cy - r, cx + r, cy + r], ang, ang + 100,
              fill=BLUE, width=4)

    # cursor: idles, then travels to the button and stays
    if t < 1.6:
        cx, cy = 420, 560
    elif t < 3.0:
        u = (t - 1.6) / 1.4
        cx, cy = lerp(420, 1050, u), lerp(560, BTN[1] + top + 22, u)
    else:
        cx, cy = 1050, BTN[1] + top + 22
    draw_cursor(d, cx, cy)
    return img


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_bug.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(DUR * FPS)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            frame(i).save(Path(tmp) / f"f{i:05d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(Path(tmp) / "f%05d.png"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
            check=True,
        )
    print(f"wrote {out} ({n} frames, {DUR}s)")


if __name__ == "__main__":
    main()
