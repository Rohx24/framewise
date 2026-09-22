"""
Continuous perceptual signal extraction.

Decodes every frame once and produces a set of time-series signals. No ML,
no API calls -- this is the cheap "retina" layer that runs at full framerate
so that nothing between frames is lost.

The one non-obvious rule here: differences are computed at WORK_WIDTH and
only reduced afterwards. Downscaling first averages small-but-important
motion (a 20px spinner, a caret, a 1px focus ring) straight out of
existence. Measure change first, summarise second.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import cv2
import numpy as np

# Resolution the deltas are computed at. High enough that a spinner or a
# text caret survives; capped so long recordings stay fast.
WORK_WIDTH = 960

# Separate, much smaller resolution for whole-frame translation. Scroll is
# a global effect, so it does not need detail and phase correlation is the
# most expensive thing in the loop.
FLOW_WIDTH = 240

GRID_COLS = 16
GRID_ROWS = 10

# A tiny grayscale thumbnail is kept for every frame. Consecutive deltas
# alone cannot answer "did the screen go back to how it was?", which is the
# whole question behind a flicker -- that needs comparing two frames that
# are not next to each other. At this size a ten-minute recording costs
# about 130MB of RAM.
THUMB_W, THUMB_H = 64, 40

# A pixel counts as "changed" above this 0-255 delta. Tuned to sit above
# h.264 ringing and subpixel antialiasing shimmer.
#
# The delta is the largest change across the three colour channels, not a
# change in brightness. Red (224,75,74) and green (32,160,110) differ by
# about 3 in luminance, so a grayscale pipeline cannot see an error banner
# turn into a success banner -- which is the single most meaningful colour
# change a UI makes.
PIXEL_DELTA = 12


@dataclasses.dataclass
class Signals:
    """Per-frame time series. Index i corresponds to time t[i]."""

    t: np.ndarray              # (N,)     timestamp of each frame, seconds
    fps: float
    width: int                 # native video width
    height: int
    work_shape: tuple          # (h, w) the deltas were computed at
    changed_px: np.ndarray     # (N,)     count of changed pixels at work res
    changed_frac: np.ndarray   # (N,)     that count as a fraction of frame
    energy: np.ndarray         # (N,)     mean abs delta, 0-255
    peak: np.ndarray           # (N,)     changed pixels in the busiest cell
    grid: np.ndarray           # (N,R,C)  changed-pixel count per cell
    bbox: np.ndarray           # (N,4)    x0,y0,x1,y1 of change, normalised
    centroid: np.ndarray       # (N,2)    x,y of change centre, normalised
    shift: np.ndarray          # (N,2)    estimated dx,dy translation, px
    luma: np.ndarray           # (N,)     mean brightness, 0-255
    thumb: np.ndarray          # (N,h,w,3) tiny colour frame, for A/B compare

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0


def _grid_count(mask: np.ndarray) -> np.ndarray:
    """
    Count of changed pixels per grid cell.

    Counting rather than averaging is deliberate. A spinner changes ~20
    pixels very strongly; averaged over a cell that reads as nearly zero and
    is indistinguishable from noise, but as a count it is unmistakable. It
    also inherits PIXEL_DELTA's noise rejection instead of needing a second
    magnitude threshold that would have to be retuned per codec.
    """
    h, w = mask.shape
    rh, rw = h // GRID_ROWS, w // GRID_COLS
    if rh == 0 or rw == 0:
        return np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    crop = mask[: rh * GRID_ROWS, : rw * GRID_COLS].astype(np.float32)
    return crop.reshape(GRID_ROWS, rh, GRID_COLS, rw).sum(axis=(1, 3))


def _extent(mask: np.ndarray) -> tuple:
    """Normalised bbox and centroid of the changed region."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return (0.0, 0.0, 0.0, 0.0), (0.0, 0.0)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    bbox = (cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h)
    centroid = (float(xs.mean()) / w, float(ys.mean()) / h)
    return bbox, centroid


def extract(path: str, max_seconds: Optional[float] = None) -> Signals:
    """Single decode pass over the video, producing all signals."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:     # missing or NaN
        fps = 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, WORK_WIDTH / native_w) if native_w else 1.0
    ww, wh = max(1, int(native_w * scale)), max(1, int(native_h * scale))
    fscale = min(1.0, FLOW_WIDTH / native_w) if native_w else 1.0
    fw, fh = max(1, int(native_w * fscale)), max(1, int(native_h * fscale))
    total_px = float(ww * wh)

    t: List[float] = []
    changed_px: List[float] = []
    energy: List[float] = []
    peak: List[float] = []
    grids: List[np.ndarray] = []
    bboxes: List[tuple] = []
    centroids: List[tuple] = []
    shifts: List[tuple] = []
    lumas: List[float] = []
    thumbs: List[np.ndarray] = []

    prev: Optional[np.ndarray] = None
    prev_flow: Optional[np.ndarray] = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps
        if max_seconds is not None and ts > max_seconds:
            break

        work = cv2.resize(frame, (ww, wh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        flow = cv2.resize(gray, (fw, fh), interpolation=cv2.INTER_AREA)
        flow_f32 = flow.astype(np.float32)

        if prev is None:
            # First frame has no delta, but we keep an entry so that signal
            # indices line up with frame numbers everywhere downstream.
            changed_px.append(0.0)
            energy.append(0.0)
            peak.append(0.0)
            grids.append(np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32))
            bboxes.append((0.0, 0.0, 0.0, 0.0))
            centroids.append((0.0, 0.0))
            shifts.append((0.0, 0.0))
        else:
            # Delta at work resolution, reduced only afterwards, and taken
            # across colour channels rather than on brightness.
            delta = cv2.absdiff(work, prev).max(axis=2)
            mask = delta > PIXEL_DELTA
            cells = _grid_count(mask)
            bbox, centroid = _extent(mask)

            changed_px.append(float(mask.sum()))
            energy.append(float(delta.mean()))
            peak.append(float(cells.max()))
            grids.append(cells)
            bboxes.append(bbox)
            centroids.append(centroid)
            # Phase correlation gives whole-frame translation, which is how
            # we tell "the page scrolled" from "the page repainted".
            try:
                (dx, dy), _ = cv2.phaseCorrelate(prev_flow, flow_f32)
            except cv2.error:
                dx, dy = 0.0, 0.0
            shifts.append((float(dx) / fscale, float(dy) / fscale))

        thumbs.append(cv2.resize(work, (THUMB_W, THUMB_H),
                                 interpolation=cv2.INTER_AREA))
        lumas.append(float(gray.mean()))
        t.append(ts)
        prev, prev_flow = work, flow_f32
        idx += 1

    cap.release()

    if not t:
        raise RuntimeError(f"no frames decoded from {path}")

    cpx = np.asarray(changed_px, dtype=np.float32)
    return Signals(
        t=np.asarray(t, dtype=np.float64),
        fps=fps,
        width=native_w,
        height=native_h,
        work_shape=(wh, ww),
        changed_px=cpx,
        changed_frac=cpx / total_px,
        energy=np.asarray(energy, dtype=np.float32),
        peak=np.asarray(peak, dtype=np.float32),
        grid=np.stack(grids),
        bbox=np.asarray(bboxes, dtype=np.float32),
        centroid=np.asarray(centroids, dtype=np.float32),
        shift=np.asarray(shifts, dtype=np.float32),
        luma=np.asarray(lumas, dtype=np.float32),
        thumb=np.stack(thumbs),
    )
