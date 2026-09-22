"""
Measuring an effect so it can be rebuilt, rather than described.

Describing an animation in prose loses the only things an implementer
needs: how far it travelled, how long it took, and the shape of the curve
between. Those are all measurable from frames, and measuring them is the
difference between "a bubble expands" and "scale 0.08 -> 1.0 over 520ms on
cubic-bezier(0.16, 1, 0.30, 1)".
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Standard easing curves, as CSS cubic-bezier control points. The named CSS
# keywords plus the handful of curves that turn up constantly in motion
# libraries -- it is far more useful to report "this is expo-out" than to
# hand back four raw coefficients.
EASINGS = {
    "linear": (0.00, 0.00, 1.00, 1.00),
    "ease": (0.25, 0.10, 0.25, 1.00),
    "ease-in": (0.42, 0.00, 1.00, 1.00),
    "ease-out": (0.00, 0.00, 0.58, 1.00),
    "ease-in-out": (0.42, 0.00, 0.58, 1.00),
    "quad-out": (0.25, 0.46, 0.45, 0.94),
    "cubic-out": (0.22, 0.61, 0.36, 1.00),
    "quart-out": (0.17, 0.84, 0.44, 1.00),
    "expo-out": (0.16, 1.00, 0.30, 1.00),
    "circ-out": (0.08, 0.82, 0.17, 1.00),
    "back-out": (0.34, 1.56, 0.64, 1.00),
    "cubic-in-out": (0.65, 0.05, 0.36, 1.00),
    "expo-in-out": (0.87, 0.00, 0.13, 1.00),
    "quart-in": (0.50, 0.00, 0.75, 0.00),
    "expo-in": (0.70, 0.00, 0.84, 0.00),
}


def _bezier_y(x: float, p: Tuple[float, float, float, float]) -> float:
    """y of a cubic-bezier at a given x, the way a browser evaluates it."""
    x1, y1, x2, y2 = p
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Solve for the parameter t where bezier_x(t) == x, then read y.
    lo, hi, t = 0.0, 1.0, x
    for _ in range(24):
        mt = 1 - t
        bx = 3 * mt * mt * t * x1 + 3 * mt * t * t * x2 + t ** 3
        if abs(bx - x) < 1e-5:
            break
        if bx < x:
            lo = t
        else:
            hi = t
        t = (lo + hi) / 2
    mt = 1 - t
    return 3 * mt * mt * t * y1 + 3 * mt * t * t * y2 + t ** 3


def fit_easing(values: np.ndarray) -> List[Tuple[str, float]]:
    """
    Rank easing curves against a measured 0..1 progression.

    Returns (name, rmse) best first. A poor best fit is itself a finding:
    it means the motion is not a single tween -- a spring, a sequence of
    stages, or something driven per-frame by a shader.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 4:
        return []
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return []
    norm = (v - lo) / (hi - lo)
    if norm[0] > norm[-1]:          # a decreasing signal still has a shape
        norm = 1.0 - norm
    x = np.linspace(0.0, 1.0, len(norm))

    out = []
    for name, pts in EASINGS.items():
        pred = np.array([_bezier_y(float(xi), pts) for xi in x])
        out.append((name, float(np.sqrt(np.mean((pred - norm) ** 2)))))
    out.sort(key=lambda c: c[1])
    return out


@dataclasses.dataclass
class FrameMetrics:
    t: float
    box: Tuple[int, int, int, int]   # x, y, w, h of the visible subject
    scale: float                     # box area relative to the largest seen
    cx: float                        # centre, fraction of the crop
    cy: float
    luma: float                      # mean brightness 0-255
    coverage: float                  # fraction of the crop that is lit
    sharpness: float                 # edge energy; falls as blur rises
    fringe: float                    # colour separation at edges


def _subject_box(gray: np.ndarray, bg: float) -> Tuple[int, int, int, int]:
    """Bounding box of whatever stands out from the background level."""
    thr = max(12.0, bg + 18.0)
    mask = gray > thr if bg < 128 else gray < thr
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        h, w = gray.shape
        return (0, 0, w, h)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def measure_frames(frames: List[np.ndarray],
                   times: List[float]) -> List[FrameMetrics]:
    """Per-frame geometry, brightness, blur and colour-fringing."""
    out: List[FrameMetrics] = []
    bg = float(np.median(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)))

    for frame, t in zip(frames, times):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        box = _subject_box(gray, bg)
        thr = max(12.0, bg + 18.0)
        lit = (gray > thr) if bg < 128 else (gray < thr)

        # Colour fringing: how far apart the red and blue channels sit on
        # strong edges. Chromatic dispersion is the giveaway for a
        # refraction shader, and it is invisible in a written description.
        edges = cv2.Canny(gray, 60, 160) > 0
        if edges.sum() > 30:
            b, _, r = cv2.split(frame.astype(np.int16))
            fringe = float(np.abs(r - b)[edges].mean())
        else:
            fringe = 0.0

        out.append(FrameMetrics(
            t=t, box=box, scale=float(box[2] * box[3]),
            cx=(box[0] + box[2] / 2) / w, cy=(box[1] + box[3] / 2) / h,
            luma=float(gray.mean()),
            coverage=float(lit.mean()),
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            fringe=fringe,
        ))

    peak = max(m.scale for m in out) or 1.0
    for m in out:
        m.scale = m.scale / peak
    return out


def palette(frame: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    """Dominant colours as hex, with the share of pixels each covers."""
    small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
    data = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(data, k, None, crit, 3,
                                    cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    counts /= counts.sum()
    order = np.argsort(-counts)
    return [(f"#{int(centres[i][2]):02x}{int(centres[i][1]):02x}"
             f"{int(centres[i][0]):02x}", float(counts[i])) for i in order]
