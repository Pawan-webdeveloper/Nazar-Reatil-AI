"""Checkout queue intelligence for one or more billing counters.

Per counter we define two polygons:
  * queue_polygon   - where customers wait
  * service_polygon - where the customer being billed stands

From anonymous tracks we measure:
  * live queue length (people waiting, excluding the one being served)
  * waiting time  = time from first seen in queue -> first seen in service area
  * service time  = time spent in the service area
  * arrival rate  = new people joining the queue per minute (rolling window)
and estimate the expected wait for a newcomer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

from ..geometry import Point, point_in_polygon


@dataclass
class _TrackState:
    queue_start: Optional[float] = None
    service_start: Optional[float] = None
    last_seen: float = 0.0
    wait_recorded: bool = False


@dataclass
class CounterState:
    name: str
    queue_polygon: Sequence[Point]
    service_polygon: Sequence[Point]
    tracks: Dict[int, _TrackState] = field(default_factory=dict)
    waits: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    services: Deque[float] = field(default_factory=lambda: deque(maxlen=200))
    arrivals: Deque[float] = field(default_factory=lambda: deque(maxlen=5000))
    queue_len: int = 0
    in_service: int = 0


class QueueMonitor:
    def __init__(self, counters: List[dict], lost_timeout_s: float = 3.0,
                 min_service_s: float = 5.0, prior_service_s: float = 120.0):
        self.counters = [CounterState(c["name"], c["queue_polygon"], c["service_polygon"]) for c in counters]
        self.lost_timeout_s = lost_timeout_s
        self.min_service_s = min_service_s
        self.prior_service_s = prior_service_s

    def update(self, ts: float, tracks: Dict[int, Point]) -> None:
        for c in self.counters:
            qlen, served = 0, 0
            for tid, pt in tracks.items():
                in_service = point_in_polygon(pt, c.service_polygon)
                in_queue = (not in_service) and point_in_polygon(pt, c.queue_polygon)
                if not (in_service or in_queue):
                    continue
                st = c.tracks.setdefault(tid, _TrackState())
                st.last_seen = ts
                if in_queue:
                    qlen += 1
                    if st.queue_start is None:
                        st.queue_start = ts
                        c.arrivals.append(ts)
                else:
                    served += 1
                    if st.service_start is None:
                        st.service_start = ts
                        if st.queue_start is None:  # walked straight to the counter
                            c.arrivals.append(ts)
                            st.queue_start = ts
                        if not st.wait_recorded:
                            c.waits.append(ts - st.queue_start)
                            st.wait_recorded = True
            c.queue_len, c.in_service = qlen, served
            # close tracks that left / were lost
            for tid, st in list(c.tracks.items()):
                if tid in tracks and (point_in_polygon(tracks[tid], c.service_polygon)
                                      or point_in_polygon(tracks[tid], c.queue_polygon)):
                    continue
                if tid not in tracks and ts - st.last_seen <= self.lost_timeout_s:
                    continue  # short occlusion, keep state
                if st.service_start is not None:
                    dur = st.last_seen - st.service_start
                    if dur >= self.min_service_s:
                        c.services.append(dur)
                del c.tracks[tid]

    # ----------------------------------------------------------------- metrics
    def arrival_rate_per_min(self, c: CounterState, now: float, window_s: float = 600) -> float:
        n = sum(1 for t in c.arrivals if now - t <= window_s)
        span = min(window_s, max(60.0, now - (c.arrivals[0] if c.arrivals else now)))
        return n / (span / 60.0)

    def status(self, now: float) -> List[dict]:
        out = []
        for c in self.counters:
            avg_service = (sum(c.services) / len(c.services)) if c.services else self.prior_service_s
            avg_wait = (sum(c.waits) / len(c.waits)) if c.waits else 0.0
            # newcomer waits for everyone ahead + remaining service of the current customer (~half)
            expected_wait = c.queue_len * avg_service + (0.5 * avg_service if c.in_service else 0.0)
            out.append({
                "counter": c.name,
                "queue_length": c.queue_len,
                "in_service": c.in_service,
                "avg_wait_s": round(avg_wait, 1),
                "avg_service_s": round(avg_service, 1),
                "measured_services": len(c.services),
                "arrival_rate_per_min": round(self.arrival_rate_per_min(c, now), 2),
                "expected_wait_new_customer_s": round(expected_wait, 1),
            })
        return out
