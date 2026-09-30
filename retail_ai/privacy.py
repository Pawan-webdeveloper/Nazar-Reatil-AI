"""Privacy-by-design utilities.

* No face recognition and no demographic inference (age / gender). That removes PII
  and also removes the largest source of demographic bias by construction.
* Tracker ids become opaque ids with a salt that rotates every N hours, so a shopper
  cannot be re-identified across days and ids are meaningless outside the store.
* Raw frames are never written to disk. Preview frames have people pixelated.
* Aggregates computed on fewer than k people are suppressed (k-anonymity).
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Iterable, Optional

import numpy as np


class AnonymousIdMapper:
    def __init__(self, rotation_hours: int = 24, secret: Optional[bytes] = None):
        self.rotation_s = max(1, rotation_hours) * 3600
        # secret is never persisted -> ids are unlinkable after a restart
        self.secret = secret if secret is not None else os.urandom(16)

    def _salt(self, ts: float) -> bytes:
        epoch = int(ts // self.rotation_s)
        return hashlib.sha256(self.secret + epoch.to_bytes(8, "big")).digest()

    def anon(self, camera_id: str, track_id: int, ts: Optional[float] = None) -> str:
        ts = time.time() if ts is None else ts
        h = hashlib.sha256(self._salt(ts) + f"{camera_id}:{track_id}".encode()).hexdigest()
        return h[:12]


def blur_regions(frame: np.ndarray, boxes: Iterable, blocks: int = 8) -> np.ndarray:
    """Return a copy of the frame with every xyxy box pixelated (previews only)."""
    out = frame.copy()
    h, w = out.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in list(b)[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = out[y1:y2, x1:x2]
        small = roi[:: max(1, (y2 - y1) // blocks), :: max(1, (x2 - x1) // blocks)]
        ry = np.linspace(0, small.shape[0] - 1, roi.shape[0]).astype(int)
        rx = np.linspace(0, small.shape[1] - 1, roi.shape[1]).astype(int)
        out[y1:y2, x1:x2] = small[ry][:, rx]
    return out


def k_anonymize(value, group_size: int, k: int = 3):
    """Suppress a statistic if it is derived from fewer than k individuals."""
    return value if group_size >= k else None
