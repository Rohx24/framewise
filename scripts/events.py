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
SCROLL_MIN_CHANGE = 0.005  # a real scroll repaints at least this much
LUMA_JUMP = 12.0         # brightness step that reads as a flash

TRANSIENT_MAX = 0.60     # a change that reverts slower than this is a real state
TRANSIENT_MIN_FRAMES = 1 # at least one frame showing the other state
TRANSIENT_SETTLE = 0.10  # seconds of stillness required on each side
TRANSIENT_STABLE = 0.02  # fraction of screen that may change and still count

JANK_MIN_BREAKS = 3      # irregular frames before motion counts as stuttering
JANK_STALL = 0.3         # of the run's typical step: below this is a stall
JANK_JUMP = 2.5          # above this is a lurch

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
    # Phase correlation is undefined on a featureless frame and returns
    # arbitrary offsets, so a blank screen reported a scroll running the
    # whole length of the recording while the stillness detector correctly
    # said not one pixel had moved. Translation only counts as scrolling
    # when pixels actually changed.
    moved = (np.abs(dy) > SCROLL_DY) & (sig.changed_frac > SCROLL_MIN_CHANGE)
    scrolling = _rolling(moved.astype(np.float32), 3) > 0.5
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

    # Pointer motion is deliberately not disqualifying here: the cursor is
    # almost always moving when a click causes a flash, and treating that as
    # "busy" rejected every real candidate. Only whole-screen motion and
    # running animations make a flash impossible to isolate.
    events = _find_transients(events, sig, scrolling | spinning)
    events += _find_jank(events, sig)

    sustained = [e for e in events
                 if e.kind in ("busy-indicator", "static", "scroll")]
    rest = [e for e in events if e not in sustained]
    events = _merge_sustained(sustained) + rest
    events.sort(key=lambda e: (e.t0, e.t1))
    return events


def _thumb_diff(sig: Signals, i: int, j: int) -> float:
    """Mean absolute difference between two frames, anywhere in the video."""
    n = len(sig)
    i, j = max(0, min(n - 1, i)), max(0, min(n - 1, j))
    a = sig.thumb[i].astype(np.float32)
    b = sig.thumb[j].astype(np.float32)
    return float(np.abs(a - b).mean())


def _find_transients(events: List[Event], sig: Signals,
                     busy: np.ndarray) -> List[Event]:
    """
    Find brief intermediate states: a flash of something that was not the
    before and was not the after either.

    This is the class of bug a person cannot report. It is over in a few
    frames, so it cannot be screenshotted, and it leaves them saying only
    "something flickered" because they never saw what it was.

    The test is deliberately not "did the screen go back". An error banner
    that shows for three frames before the success banner never goes back --
    it settles somewhere new, and requiring a revert misses it entirely.
    What all of these share is a middle that differs from *both* ends, which
    consecutive frame deltas cannot express at all: it needs two frames that
    are not adjacent to be compared.
    """
    n = len(sig)
    consecutive = np.array(
        [_thumb_diff(sig, i - 1, i) for i in range(1, min(n, 400))],
        dtype=np.float32) if n > 1 else np.zeros(1, dtype=np.float32)
    base = float(np.median(consecutive)) if consecutive.size else 0.0
    revert_max = max(1.2, base * 3.0)     # "back to how it was"
    differ_min = max(2.0, base * 6.0)     # "visibly something else"

    idx = sig.index_at

    # A flash only means anything if the screen had settled either side of
    # it. Without this, continuous motion is one unbroken run of transients:
    # while a page scrolls, every single frame differs from the frame before
    # it and from the frame after it, so the test fires on all of them and
    # buries the real finding under dozens of whole-screen false positives.
    stable = (sig.changed_frac < TRANSIENT_STABLE) & ~busy
    settle = max(2, int(round(sig.fps * TRANSIENT_SETTLE)))

    def settled_before(i: int) -> bool:
        lo = max(0, i - settle - 1)
        return bool(stable[lo:max(lo + 1, i - 1)].all())

    def settled_after(i: int) -> bool:
        hi = min(len(sig), i + settle + 2)
        return bool(stable[min(i + 2, len(sig) - 1):hi].all())

    instants = [e for e in events
                if e.instant and e.kind in ("local-change", "region-change", "cut")]

    # Collect every valid pair, then choose. Taking the first match and
    # moving on meant the widest pair won -- the click and the settled state
    # half a second later -- which spans the flash instead of isolating it.
    cands = []
    for a in range(len(instants)):
        e1 = instants[a]
        for b in range(a + 1, len(instants)):
            e2 = instants[b]
            gap = e2.t0 - e1.t0
            if gap <= 0 or gap > TRANSIENT_MAX:
                break
            i1, i2 = idx(e1.t0), idx(e2.t0)
            if i2 - i1 < TRANSIENT_MIN_FRAMES:
                continue
            if busy[i1:i2 + 1].any():
                continue
            if not (settled_before(i1) and settled_after(i2)):
                continue
            # The state that flashed has to have been *displayed*, not
            # animated. A scroll that starts and stops inside the window
            # also has a still frame either side and a different-looking
            # middle, and is otherwise indistinguishable from a flash.
            span = i2 - i1
            if span > 2 and int((~stable[i1 + 1:i2]).sum()) > max(1, span // 4):
                continue
            before, after = i1 - 2, i2 + 2
            during = (i1 + i2) // 2
            # The flashed state has to hold still while it is up. Slow
            # continuous drift -- a long cursor drag, a lazy settle -- never
            # moves much in any one frame, so every frame reads as stable,
            # yet the drift accumulates until the middle differs from both
            # ends and a half-second of creep is reported as a flash.
            if _thumb_diff(sig, i1 + 1, max(i1 + 1, i2 - 1)) > revert_max:
                continue
            reverted = _thumb_diff(sig, before, after) <= revert_max
            if (_thumb_diff(sig, before, during) >= differ_min
                    and _thumb_diff(sig, during, after) >= differ_min):
                region = e1.region or e2.region
                if e1.region and e2.region:
                    region = (min(e1.region[0], e2.region[0]),
                              min(e1.region[1], e2.region[1]),
                              max(e1.region[2], e2.region[2]),
                              max(e1.region[3], e2.region[3]))
                tail = ("and went back to how it started" if reverted
                        else "then settled into a third, different state")
                cands.append((Event(
                    e1.t0, e2.t0, "transient",
                    f"a {describe_region(region)} showed something for only "
                    f"{gap:.2f}s ({int(round(gap * sig.fps))} frames) {tail} "
                    f"{_pct(region)}",
                    region,
                ), e1, e2))

    # Tightest first, skipping anything that overlaps one already taken.
    consumed, kept = set(), []
    for ev, e1, e2 in sorted(cands, key=lambda c: c[0].duration):
        if any(ev.t0 < k.t1 and k.t0 < ev.t1 for k in kept):
            continue
        kept.append(ev)
        consumed.add(id(e1))
        consumed.add(id(e2))

    return [e for e in events if id(e) not in consumed] + kept


def _find_jank(events: List[Event], sig: Signals) -> List[Event]:
    """
    Find motion that does not advance evenly.

    Smooth scrolling moves by a similar amount every frame. Stutter shows up
    as frames that barely move followed by frames that lurch. A person can
    only report this as "it feels janky", and a screenshot of a stutter is
    indistinguishable from a screenshot of smooth motion.
    """
    out: List[Event] = []
    idx = sig.index_at

    for ev in events:
        if ev.kind != "scroll" or ev.duration < 0.4:
            continue
        a, b = idx(ev.t0), idx(ev.t1) + 1
        steps = np.abs(sig.shift[a:b, 1])
        if steps.size < 6:
            continue
        typical = float(np.median(steps))
        if typical < 1.0:
            continue
        stalls = int((steps < typical * JANK_STALL).sum())
        jumps = int((steps > typical * JANK_JUMP).sum())
        if stalls + jumps < JANK_MIN_BREAKS:
            continue
        out.append(Event(
            ev.t0, ev.t1, "jank",
            f"motion advanced unevenly: of {steps.size} frames, {stalls} "
            f"barely moved and {jumps} lurched, against a typical step of "
            f"{typical:.0f}px",
        ))
    return out


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
