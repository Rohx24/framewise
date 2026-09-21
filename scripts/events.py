"""
Event segmentation.

Turns the continuous signals into a sparse list of events with boundaries,
the way people remember an experience as events rather than as seconds.

Nothing here interprets meaning. It reports *what changed, where, and for
how long*. Deciding that a spinning arc means "the save request hung" is
the reading model's job. Keeping that boundary is what stops the pipeline
from inventing a story the pixels do not support.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from signals import Signals

# --- thresholds ------------------------------------------------------------
MIN_CELL_PX = 3.0        # changed pixels before a grid cell counts as active
MAX_LOCAL_CELLS = 6      # at most this many active cells is "localised"
FROZEN_PX = 2.0          # changed pixels below this is a frozen frame

CUT_FRAC = 0.30          # most of the screen repainted
TRANSITION_FRAC = 0.06   # a big region repainted (modal, panel, nav)
LOCAL_FRAC_MIN = 0.0002  # smaller than this is codec noise

POINTER_SPEED = 0.004    # normalised centroid travel/frame that reads as motion
POINTER_MAX_PX = 800.0   # a pointer is cursor-sized; bigger is a real repaint
POINTER_MIN_TRAVEL = 0.03  # net displacement before we call it movement
SCROLL_DY = 1.5          # px of vertical translation per frame
LUMA_JUMP = 12.0         # brightness step that reads as a flash

MIN_SUSTAINED = 0.7      # seconds before a sustained pattern is an event
MIN_STALL = 1.0          # seconds of stillness worth reporting
ACTIVITY_RATIO = 0.55    # fraction of a window a cell must be active


@dataclasses.dataclass
class Event:
    t0: float
    t1: float
    kind: str
    detail: str
    region: Optional[tuple] = None   # normalised x0,y0,x1,y1

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def instant(self) -> bool:
        return self.duration < 1e-6


def _runs(flags: np.ndarray) -> List[tuple]:
    """Contiguous True runs as (start_index, end_index_exclusive)."""
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def _rolling(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean along axis 0, edge-padded to preserve length."""
    if window <= 1:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    pad = window // 2
    padding = [(pad, pad)] + [(0, 0)] * (x.ndim - 1)
    padded = np.pad(x, padding, mode="edge")
    flat = padded.reshape(len(padded), -1)
    kernel = np.ones(window, dtype=np.float32) / window
    out = np.empty((len(x), flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.convolve(flat[:, j], kernel, mode="valid")[: len(x)]
    return out.reshape((len(x),) + x.shape[1:])


def _refine(a: int, b: int, instant: np.ndarray, bridge: int = 0) -> tuple:
    """
    Widen a run detected on smoothed signals back to its true boundaries.

    Rolling windows are what make sustained patterns detectable at all, but
    they cost roughly half a window at each edge -- enough to report a hang
    as starting a third of a second after it actually did. The smoothed run
    says *that* something happened; the raw per-frame predicate says *when*.

    `bridge` tolerates short dropouts. On a compressed recording a spinner's
    per-frame signature flickers, and without bridging the run truncates at
    the first dropped frame -- which reported a hang as starting five
    seconds after it really did.
    """
    misses = 0
    i = a
    while i > 0:
        if instant[i - 1]:
            a, misses = i - 1, 0
        else:
            misses += 1
            if misses > bridge:
                break
        i -= 1

    misses = 0
    j = b
    while j < len(instant):
        if instant[j]:
            b, misses = j + 1, 0
        else:
            misses += 1
            if misses > bridge:
                break
        j += 1
    return a, b


def _merge_sustained(events: List["Event"], gap: float = 0.35) -> List["Event"]:
    """
    Collapse same-kind sustained events that overlap or nearly touch.

    Refinement can widen several fragments of one noisy run to the same
    span, producing duplicate events -- and then duplicate findings, which
    is how one hung spinner got reported four times.
    """
    out: List["Event"] = []
    for ev in sorted(events, key=lambda e: (e.kind, e.t0)):
        prev = out[-1] if out else None
        if prev and prev.kind == ev.kind and ev.t0 <= prev.t1 + gap:
            if ev.t1 > prev.t1:
                region = prev.region
                if prev.region and ev.region:
                    region = (min(prev.region[0], ev.region[0]),
                              min(prev.region[1], ev.region[1]),
                              max(prev.region[2], ev.region[2]),
                              max(prev.region[3], ev.region[3]))
                out[-1] = Event(prev.t0, ev.t1, prev.kind, prev.detail, region)
        else:
            out.append(ev)
    return out


def _region_of_cells(cells: np.ndarray) -> Optional[tuple]:
    rows = np.flatnonzero(cells.any(axis=1))
    cols = np.flatnonzero(cells.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    r, c = cells.shape
    return (cols[0] / c, rows[0] / r, (cols[-1] + 1) / c, (rows[-1] + 1) / r)


def describe_region(region: Optional[tuple]) -> str:
    """Plain-English screen position, so the reader gets orientation cheaply."""
    if region is None:
        return "unknown region"
    x0, y0, x1, y1 = region
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vert = "top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle"
    horz = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    area = (x1 - x0) * (y1 - y0)
    size = "small" if area < 0.04 else "large" if area > 0.4 else "medium"
    where = "centre of screen" if (vert, horz) == ("middle", "centre") \
        else f"{vert}-{horz}"
    return f"{size} region, {where}"


def _pct(region: Optional[tuple]) -> str:
    if region is None:
        return ""
    x0, y0, x1, y1 = region
    return f"[{x0:.0%},{y0:.0%} to {x1:.0%},{y1:.0%}]"


def detect(sig: Signals) -> List[Event]:
    """Run every detector and return events ordered by start time."""
    win = max(3, int(round(sig.fps * MIN_SUSTAINED)))
    n = len(sig)
    events: List[Event] = []

    # Noise floor learned from the recording itself, because a threshold
    # tuned on a crisp capture drowns on a crf-40 re-encode and screen
    # recordings arrive compressed by whatever tool made them.
    #
    # The statistic is the *spatial* median within each frame: in any one
    # frame of a screen recording almost every cell is static, so the median
    # cell is pure codec noise. Taking a per-cell percentile over time looks
    # equivalent and is not -- a spinner occupying its cell 72% of the time
    # pushes its own percentile up into its own signal and threshold itself
    # out of existence, which is precisely how a hung spinner got reported
    # as a frozen screen.
    floor = np.median(sig.grid, axis=(1, 2)).astype(np.float32)   # (N,)
    cell_thresh = np.maximum(MIN_CELL_PX, floor * 3.0 + 1.0)
    active = sig.grid >= cell_thresh[:, None, None]  # (N,R,C)
    n_active = active.sum(axis=(1, 2))
    localised = (n_active > 0) & (n_active <= MAX_LOCAL_CELLS)

    # How far the centre of change travels per frame. This is what separates
    # a spinner from a mouse pointer: both are one small busy cell, but only
    # one of them moves across the screen.
    step = np.zeros(n, dtype=np.float32)
    if n > 1:
        d = np.linalg.norm(np.diff(sig.centroid, axis=0), axis=1)
        step[1:] = d
    step = np.where(n_active > 0, step, 0.0)
    travel = _rolling(step, win)
    stationary = travel < POINTER_SPEED

    # -- scrolling ---------------------------------------------------------
    dy = sig.shift[:, 1]
    scrolling = _rolling((np.abs(dy) > SCROLL_DY).astype(np.float32), 3) > 0.5
    for a, b in _runs(scrolling):
        if sig.t[b - 1] - sig.t[a] < 0.25:
            continue
        total = float(np.sum(dy[a:b]))
        events.append(Event(
            float(sig.t[a]), float(sig.t[b - 1]), "scroll",
            f"content scrolled {'down' if total < 0 else 'up'} "
            f"~{abs(total):.0f}px over {sig.t[b-1]-sig.t[a]:.1f}s",
        ))

    # -- sustained localised animation (spinners, progress bars, pulsing) --
    # The same small cell keeps changing while the rest of the screen holds
    # still. This is the signature of a busy indicator, and it is the single
    # most diagnostic thing in a hung-UI recording.
    ratio = _rolling(active.astype(np.float32), win)
    spinning = (
        (ratio.max(axis=(1, 2)) > ACTIVITY_RATIO)
        & localised & stationary & ~scrolling
    )
    for a, b in _runs(spinning):
        # Identify which cells are doing the animating, then widen the run
        # using raw activity in *those* cells. Refining on the smoothed
        # signals instead would inherit their lag -- and the onset of a hang
        # is the number the reader cares most about.
        cells = ratio[a:b].max(axis=0) > ACTIVITY_RATIO
        if not cells.any():
            continue
        here = active[:, cells].any(axis=1) & localised & ~scrolling
        a, b = _refine(a, b, here, bridge=int(sig.fps * 0.25))
        dur = float(sig.t[b - 1] - sig.t[a])
        if dur < MIN_SUSTAINED:
            continue
        region = _region_of_cells(cells)
        events.append(Event(
            float(sig.t[a]), float(sig.t[b - 1]), "busy-indicator",
            f"something animated continuously for {dur:.1f}s in a "
            f"{describe_region(region)} while the rest of the screen was "
            f"static {_pct(region)}",
            region,
        ))

    # -- pointer movement --------------------------------------------------
    # The size cap matters: without it a button's press-state repaint gets
    # absorbed into the surrounding pointer run and the click -- the single
    # most important moment in a bug report -- vanishes from the timeline.
    pointing = (
        localised & ~stationary & ~scrolling & ~spinning
        & (sig.changed_px <= POINTER_MAX_PX)
    )
    for a, b in _runs(pointing):
        if sig.t[b - 1] - sig.t[a] < 0.2:
            continue
        x0, y0 = sig.centroid[a]
        x1, y1 = sig.centroid[b - 1]
        if float(np.hypot(x1 - x0, y1 - y0)) < POINTER_MIN_TRAVEL:
            continue   # jitter in place, not travel
        events.append(Event(
            float(sig.t[a]), float(sig.t[b - 1]), "pointer",
            f"a small element moved across the screen from "
            f"({x0:.0%},{y0:.0%}) to ({x1:.0%},{y1:.0%}) "
            f"- consistent with the mouse pointer",
        ))

    # -- stalls: nothing moves at all --------------------------------------
    # Reuse the adaptive cell mask rather than thresholding raw pixel counts.
    # A global threshold gets dragged upward by whatever is moving -- derived
    # from the pixel counts directly, a spinner raises the bar above its own
    # signal and the hang gets reported as a frozen screen.
    frozen = n_active == 0
    frozen[0] = False
    for a, b in _runs(frozen):
        dur = float(sig.t[b - 1] - sig.t[a])
        if dur < MIN_STALL:
            continue
        events.append(Event(
            float(sig.t[a]), float(sig.t[b - 1]), "static",
            f"not a single pixel changed for {dur:.1f}s",
        ))

    # -- discrete changes: cuts, transitions, local updates ----------------
    busy = spinning | scrolling | pointing
    for i in range(1, n):
        if busy[i]:
            continue
        frac = float(sig.changed_frac[i])
        region = tuple(float(v) for v in sig.bbox[i])
        if frac >= CUT_FRAC:
            kind, detail = "cut", f"{frac:.0%} of the screen repainted"
        elif frac >= TRANSITION_FRAC:
            kind = "region-change"
            detail = (f"{frac:.0%} of the screen changed "
                      f"({describe_region(region)}) {_pct(region)}")
        elif frac >= LOCAL_FRAC_MIN:
            kind = "local-change"
            detail = (f"{frac:.2%} of the screen changed "
                      f"({describe_region(region)}) {_pct(region)}")
        else:
            continue
        events.append(Event(float(sig.t[i]), float(sig.t[i]), kind, detail, region))

    # -- flashes: a sudden global brightness step --------------------------
    dl = np.diff(sig.luma, prepend=sig.luma[0])
    for i in np.flatnonzero(np.abs(dl) > LUMA_JUMP):
        events.append(Event(
            float(sig.t[i]), float(sig.t[i]), "flash",
            f"screen brightness jumped "
            f"{'brighter' if dl[i] > 0 else 'darker'} "
            f"({sig.luma[i-1]:.0f} to {sig.luma[i]:.0f})",
        ))

    sustained = [e for e in events
                 if e.kind in ("busy-indicator", "static", "scroll")]
    rest = [e for e in events if e not in sustained]
    events = _merge_sustained(sustained) + rest
    events.sort(key=lambda e: (e.t0, e.t1))
    return events


MERGEABLE = ("local-change", "region-change", "flash", "pointer")


def coalesce(events: List[Event], gap: float = 0.15) -> List[Event]:
    """
    Collapse bursts of same-kind events into one.

    A single click produces a flurry of local-changes as the button presses,
    the ripple plays and the focus ring lands. A reader wants one line, and
    the model reading this has a finite context.
    """
    out: List[Event] = []
    for ev in events:
        prev = out[-1] if out else None
        if (
            prev is not None
            and ev.kind == prev.kind
            and ev.kind in MERGEABLE
            and ev.t0 - prev.t1 <= gap
        ):
            region = prev.region
            if prev.region and ev.region:
                region = (min(prev.region[0], ev.region[0]),
                          min(prev.region[1], ev.region[1]),
                          max(prev.region[2], ev.region[2]),
                          max(prev.region[3], ev.region[3]))
            if ev.kind == "pointer":
                detail = prev.detail
            else:
                detail = (f"activity in a {describe_region(region)} "
                          f"{_pct(region)}").strip()
            out[-1] = Event(prev.t0, ev.t1, prev.kind, detail, region)
        else:
            out.append(ev)
    return out
