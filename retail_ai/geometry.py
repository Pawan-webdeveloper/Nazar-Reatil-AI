"""Small geometry helpers (pure numpy, easy to unit-test)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def side_of_line(pt: Point, a: Point, b: Point) -> float:
    """Cross product sign: >0 on one side of a->b, <0 on the other, 0 on the line."""
    return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1 = side_of_line(q1, p1, p2)
    d2 = side_of_line(q2, p1, p2)
    d3 = side_of_line(p1, q1, q2)
    d4 = side_of_line(p2, q1, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def foot_point(box_xyxy) -> Point:
    """Bottom-centre of a person box = ground contact point."""
    x1, y1, x2, y2 = box_xyxy[:4]
    return (float(x1 + x2) / 2.0, float(y2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4) if len(a) else np.zeros((0, 4), np.float32)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4) if len(b) else np.zeros((0, 4), np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def greedy_match(iou: np.ndarray, thr: float = 0.5):
    """Greedy one-to-one matching by IoU (highest first). Returns list of (i, j)."""
    pairs = []
    if iou.size == 0:
        return pairs
    used_i, used_j = set(), set()
    flat = np.argsort(-iou, axis=None)
    for idx in flat:
        i, j = np.unravel_index(idx, iou.shape)
        if iou[i, j] < thr:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(int(i))
        used_j.add(int(j))
        pairs.append((int(i), int(j)))
    return pairs
