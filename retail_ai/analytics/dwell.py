"""Zone dwell-time measurement.

A visit starts when a track's foot point enters a zone polygon and ends when it
leaves the zone, or when the track has not been seen for `lost_timeout_s`.
Visits shorter than `min_visit_s` are ignored (people just walking past);
visits longer than `engaged_s` count as an "engagement" with the display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from ..geometry import Point, point_in_polygon


@dataclass
class Visit:
    zone: str
    track_id: int
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class ZoneDwellTracker:
    zones: Dict[str, Sequence[Point]]
    min_visit_s: float = 1.0
    engaged_s: float = 10.0
    lost_timeout_s: float = 2.0
    _open: Dict[Tuple[str, int], List[float]] = field(default_factory=dict)  # (zone,tid) -> [start,last_seen]
    visits: List[Visit] = field(default_factory=list)

    def update(self, ts: float, tracks: Dict[int, Point]) -> List[Visit]:
        closed = []
        for tid, pt in tracks.items():
            for zname, poly in self.zones.items():
                key = (zname, tid)
                inside = point_in_polygon(pt, poly)
                if inside:
                    if key in self._open:
                        self._open[key][1] = ts
                    else:
                        self._open[key] = [ts, ts]
                elif key in self._open:
                    closed += self._close(key)
        # tracks that vanished
        for key, (start, last) in list(self._open.items()):
            if ts - last > self.lost_timeout_s:
                closed += self._close(key)
        return closed

    def _close(self, key) -> List[Visit]:
        start, last = self._open.pop(key)
        if last - start < self.min_visit_s:
            return []
        v = Visit(key[0], key[1], start, last)
        self.visits.append(v)
        return [v]

    def flush(self) -> List[Visit]:
        out = []
        for key in list(self._open):
            out += self._close(key)
        return out

    def current_occupancy(self) -> Dict[str, int]:
        occ = {z: 0 for z in self.zones}
        for (z, _tid) in self._open:
            occ[z] += 1
        return occ

    def summary(self) -> Dict[str, dict]:
        out = {}
        for z in self.zones:
            vs = [v.seconds for v in self.visits if v.zone == z]
            out[z] = {
                "visits": len(vs),
                "avg_dwell_s": round(sum(vs) / len(vs), 2) if vs else 0.0,
                "max_dwell_s": round(max(vs), 2) if vs else 0.0,
                "engaged_visits": sum(1 for s in vs if s >= self.engaged_s),
            }
        return out
