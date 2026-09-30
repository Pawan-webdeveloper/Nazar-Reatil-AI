"""Entry / exit counting with a virtual line (hysteresis + side memory).

Each track has a *committed side* of the line. The side only changes when the foot point is
more than `margin` pixels away from the line on the other side, and the crossing point lies
within the line segment. This gives two properties that plain segment-intersection counting
lacks (measured on MOT17, see outputs/people_counting):

* people standing on / near the line no longer produce alternating in/out counts (jitter)
* a crossing is still counted when the tracker missed the frames right at the line, because
  the committed side is remembered instead of requiring two consecutive observations

Every track id can be counted at most once per direction within `recount_s` seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..geometry import Point


@dataclass
class CrossEvent:
    ts: float
    track_id: int
    direction: str  # "in" | "out"
    point: Optional[Point] = None


@dataclass
class LineCounter:
    line: Tuple[Point, Point]
    entry_direction: str = "down"  # movement direction that means ENTRY
    recount_s: float = 5.0
    margin: float = 8.0            # pixels a foot point must be beyond the line to commit to a side
    stitch_s: float = 1.5          # a new track may inherit the side of a track lost this recently...
    stitch_px: float = 0.0         # ...if it starts this close to where it was last seen (0 = off; MOT17 showed
                                   # stitching hurts in crowds, see outputs/people_counting)
    entries: int = 0
    exits: int = 0
    _side: Dict[int, int] = field(default_factory=dict)         # committed side (+1 / -1)
    _last_pt: Dict[int, Point] = field(default_factory=dict)    # last committed position
    _last_count: Dict[Tuple[int, str], float] = field(default_factory=dict)
    _seen: Dict[int, Tuple[float, Point]] = field(default_factory=dict)  # last time / position of every track
    events: List[CrossEvent] = field(default_factory=list)

    def __post_init__(self):
        (ax, ay), (bx, by) = self.line
        self._len = max(1e-9, ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
        # which signed side is the "entry" destination for the configured direction
        probe = {"down": (0, 1), "up": (0, -1), "right": (1, 0), "left": (-1, 0)}[self.entry_direction]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        s = self._signed((mx + probe[0] * 10, my + probe[1] * 10))
        self._entry_side = 1 if s > 0 else -1

    def _signed(self, p: Point) -> float:
        (ax, ay), (bx, by) = self.line
        return ((bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)) / self._len

    def _within_segment(self, p: Point, q: Point) -> bool:
        """Projection of the midpoint of p->q lies on the line segment (with a small tolerance)."""
        (ax, ay), (bx, by) = self.line
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        t = ((mx - ax) * (bx - ax) + (my - ay) * (by - ay)) / (self._len ** 2)
        return -0.02 <= t <= 1.02

    def update(self, ts: float, tracks: Dict[int, Point]) -> List[CrossEvent]:
        """tracks: {track_id: foot_point}. Returns crossings detected in this frame."""
        new = []
        for tid, pt in tracks.items():
            if tid not in self._seen and tid not in self._side:
                self._stitch(ts, tid, pt, tracks)
            self._seen[tid] = (ts, pt)
            d = self._signed(pt)
            if abs(d) < self.margin:
                continue  # inside the dead band around the line: no decision
            side = 1 if d > 0 else -1
            prev_side: Optional[int] = self._side.get(tid)
            prev_pt = self._last_pt.get(tid)
            self._side[tid] = side
            self._last_pt[tid] = pt
            if prev_side is None or prev_side == side or not self._within_segment(prev_pt, pt):
                continue
            direction = "in" if side == self._entry_side else "out"
            key = (tid, direction)
            if ts - self._last_count.get(key, -1e18) < self.recount_s:
                continue
            self._last_count[key] = ts
            if direction == "in":
                self.entries += 1
            else:
                self.exits += 1
            ev = CrossEvent(ts, tid, direction, pt)
            new.append(ev)
            self.events.append(ev)
        return new

    def _stitch(self, ts: float, tid: int, pt: Point, current) -> None:
        """Tracker id switch repair: a brand-new id that appears right where another id vanished a
        moment ago inherits that id's committed side (and its recount history)."""
        best, best_d = None, self.stitch_px
        for old, (t_old, p_old) in self._seen.items():
            if old in current or old not in self._side or ts - t_old > self.stitch_s:
                continue
            dist = ((pt[0] - p_old[0]) ** 2 + (pt[1] - p_old[1]) ** 2) ** 0.5
            if dist < best_d:
                best, best_d = old, dist
        if best is None:
            return
        self._side[tid] = self._side.pop(best)
        self._last_pt[tid] = self._last_pt.pop(best)
        for direction in ("in", "out"):
            if (best, direction) in self._last_count:
                self._last_count[(tid, direction)] = self._last_count.pop((best, direction))
        self._seen.pop(best, None)

    def forget(self, active_ids) -> None:
        """Kept for API compatibility. Side memory is retained so a track that disappears for a
        few frames and reappears on the other side is still counted; memory is bounded by pruning
        the oldest ids."""
        if len(self._seen) > 5000:
            newest = max(t for t, _ in self._seen.values())
            for tid in [t for t, (ts, _) in self._seen.items() if newest - ts > 10 * self.stitch_s]:
                self._seen.pop(tid, None)
        if len(self._side) > 5000:
            for tid in list(self._side)[:1000]:
                self._side.pop(tid, None)
                self._last_pt.pop(tid, None)

    @property
    def occupancy(self) -> int:
        return max(0, self.entries - self.exits)
