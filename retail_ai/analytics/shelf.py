"""Shelf / inventory analytics from product detections.

Pipeline (per shelf image or per shelf section):
 1. product boxes come from the SKU-110K-trained detector
 2. boxes are grouped into shelf rows by vertical position
 3. inside each row, horizontal gaps wider than `gap_ratio` x the median product width
    are reported as *voids* (out-of-stock / missing facings); the leading and trailing
    part of a row is also checked against the shelf extent
 4. stock level = detected facings / estimated capacity (facings + void slots), and,
    when a planogram is configured, facings / expected facings
 5. planogram compliance compares detected rows / facings with the expected layout,
    and (optionally) checks product appearance per section with `SkuMatcher`

This follows the "void detection on top of product detection" approach used in
out-of-stock literature (e.g. Enhanced Out-of-Stock Detection in Retail Shelf Images,
Sensors 2024) while staying light enough for CPU-only edge boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..geometry import point_in_polygon


@dataclass
class Void:
    row: int
    box: Tuple[float, float, float, float]
    missing_facings: int


@dataclass
class ShelfReport:
    section: str
    facings: int
    rows: int
    voids: List[Void] = field(default_factory=list)
    capacity_estimate: int = 0
    fill_ratio: float = 1.0
    stock_ratio_vs_planogram: Optional[float] = None
    planogram_compliance: Optional[float] = None
    status: str = "OK"  # OK | LOW_STOCK | OUT_OF_STOCK
    issues: List[str] = field(default_factory=list)
    boxes: List[List[int]] = field(default_factory=list)  # product positions (for the fixed-camera baseline)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "facings": self.facings,
            "rows": self.rows,
            "voids": [{"row": v.row, "box": [round(x, 1) for x in v.box], "missing_facings": v.missing_facings}
                      for v in self.voids],
            "capacity_estimate": self.capacity_estimate,
            "fill_ratio": round(self.fill_ratio, 3),
            "stock_ratio_vs_planogram": None if self.stock_ratio_vs_planogram is None
            else round(self.stock_ratio_vs_planogram, 3),
            "planogram_compliance": None if self.planogram_compliance is None else round(self.planogram_compliance, 3),
            "status": self.status,
            "issues": self.issues,
            "boxes": self.boxes,
        }


def cluster_rows(boxes: np.ndarray, tol: float = 0.5) -> List[np.ndarray]:
    """Group boxes into shelf rows. Returns list of index arrays sorted top->bottom."""
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h_med = float(np.median(boxes[:, 3] - boxes[:, 1]))
    order = np.argsort(cy)
    rows, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        # new row when the next centre is clearly lower than the running row centre
        if cy[b] - np.mean(cy[cur]) > tol * h_med:
            rows.append(np.array(cur))
            cur = [b]
        else:
            cur.append(b)
    rows.append(np.array(cur))
    return rows


class ShelfAnalyzer:
    """Product boxes -> shelf rows -> empty-slot (void) candidates -> verified voids -> stock status.

    False-alarm controls (each one measured on the SKU-110K validation split):
      * rows need >= `min_row_items` products (stray boxes do not form a row)
      * the left / right end of a row is compared with the neighbouring rows (or the configured
        shelf polygon), never with the whole image, so short or cut-off rows are not voids
      * a candidate gap already covered by other detected boxes is rejected
      * texture check: an empty shelf is flat, products have many edges. A candidate whose
        edge density is >= `texture_ratio` x the median product edge density is rejected
    """

    def __init__(self, gap_ratio: float = 1.5, low_stock_ratio: float = 0.5,
                 min_void_area_ratio: float = 0.0, min_row_items: int = 3,
                 texture_ratio: float = 0.6, max_overlap: float = 0.3,
                 baseline_min_cover: float = 0.3, baseline_texture_ratio: float = 0.0):
        self.baseline_min_cover = baseline_min_cover
        self.baseline_texture_ratio = baseline_texture_ratio
        self.gap_ratio = gap_ratio
        self.low_stock_ratio = low_stock_ratio
        self.min_void_area_ratio = min_void_area_ratio
        self.min_row_items = min_row_items
        self.texture_ratio = texture_ratio
        self.max_overlap = max_overlap

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _edge_map(image):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150)

    @staticmethod
    def _density(edges, box) -> float:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = edges.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return 0.0
        return float((edges[y1:y2, x1:x2] > 0).mean())

    @staticmethod
    def _covered(box, boxes) -> float:
        """Fraction of `box` covered by other boxes (sum of intersections, capped at 1)."""
        x1, y1, x2, y2 = box
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        iw = np.clip(np.minimum(x2, boxes[:, 2]) - np.maximum(x1, boxes[:, 0]), 0, None)
        ih = np.clip(np.minimum(y2, boxes[:, 3]) - np.maximum(y1, boxes[:, 1]), 0, None)
        return float(min(1.0, (iw * ih).sum() / area))

    # ---------------------------------------------------------------- main
    def analyze(self, boxes, image_size: Tuple[int, int], section: str = "shelf",
                polygon: Optional[Sequence[Tuple[float, float]]] = None,
                expected_facings: int = 0, expected_rows: int = 0, image=None,
                baseline_boxes=None) -> ShelfReport:
        """If `baseline_boxes` (product positions of the 'fully stocked' snapshot of this fixed camera)
        are given, voids come from baseline comparison; otherwise from gap analysis."""
        w_img, h_img = image_size
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.zeros((0, 4), np.float32)
        poly_left = poly_right = None
        if polygon:
            keep = [point_in_polygon(((x1 + x2) / 2, (y1 + y2) / 2), polygon) for x1, y1, x2, y2 in b]
            b = b[np.array(keep, dtype=bool)] if len(b) else b
            xs = [p[0] for p in polygon]
            poly_left, poly_right = float(min(xs)), float(max(xs))
            area_total = float((max(xs) - min(xs)) * (max(p[1] for p in polygon) - min(p[1] for p in polygon)))
        else:
            area_total = float(w_img * h_img)

        rep = ShelfReport(section=section, facings=int(len(b)), rows=0)
        rep.boxes = np.round(b).astype(int).tolist()
        if len(b) == 0:
            rep.status = "OUT_OF_STOCK"
            rep.fill_ratio = 0.0
            rep.issues.append("no products detected in section")
            if expected_facings:
                rep.stock_ratio_vs_planogram = 0.0
                rep.planogram_compliance = 0.0
            return rep

        rows = [r for r in cluster_rows(b) if len(r) >= self.min_row_items]
        rep.rows = len(rows)
        # rows of very short boxes (price-tag strips, shelf lips) are not product rows -> no voids from them
        h_all = float(np.median(b[:, 3] - b[:, 1]))
        rows = [r for r in rows if float(np.median(b[r][:, 3] - b[r][:, 1])) >= 0.4 * h_all]
        extents = [(float(b[r][:, 0].min()), float(b[r][:, 2].max())) for r in rows]
        edges = self._edge_map(image) if image is not None else None
        ref_density = None
        if edges is not None:
            sample = b[np.linspace(0, len(b) - 1, min(len(b), 60)).astype(int)]
            dens = [self._density(edges, bx) for bx in sample]
            ref_density = float(np.median(dens)) if dens else None

        void_slots = 0
        if baseline_boxes is not None and len(baseline_boxes):
            rep.voids = baseline_voids(baseline_boxes, b, image, self.baseline_min_cover, self.baseline_texture_ratio)
            void_slots = sum(v.missing_facings for v in rep.voids)
            rows = []  # skip gap analysis
        for ri, idx in enumerate(rows):
            rb = b[idx]
            rb = rb[np.argsort(rb[:, 0])]
            widths = rb[:, 2] - rb[:, 0]
            w_med = float(np.median(widths))
            y1, y2 = float(np.median(rb[:, 1])), float(np.median(rb[:, 3]))
            cands = []
            for i in range(len(rb) - 1):
                gap = float(rb[i + 1, 0] - rb[i, 2])
                ref = max(w_med, float(widths[i] + widths[i + 1]) / 2)
                if gap > self.gap_ratio * ref:
                    cands.append((float(rb[i, 2]), float(rb[i + 1, 0]), ref))
            # row ends: compare with the configured shelf, else with the neighbouring rows
            neigh = [extents[j] for j in (ri - 1, ri + 1) if 0 <= j < len(rows)]
            left_ref = poly_left if poly_left is not None else (min(e[0] for e in neigh) if neigh else None)
            right_ref = poly_right if poly_right is not None else (max(e[1] for e in neigh) if neigh else None)
            if left_ref is not None and rb[0, 0] - left_ref > self.gap_ratio * w_med:
                cands.append((left_ref, float(rb[0, 0]), w_med))
            if right_ref is not None and right_ref - rb[-1, 2] > self.gap_ratio * w_med:
                cands.append((float(rb[-1, 2]), right_ref, w_med))
            for gx1, gx2, ref in cands:
                box = (gx1, y1, gx2, y2)
                area = (gx2 - gx1) * (y2 - y1)
                if area_total > 0 and area / area_total < self.min_void_area_ratio:
                    continue
                if self._covered(box, b) > self.max_overlap:
                    continue  # other detections already occupy this space (row-clustering artefact)
                if ref_density and self._density(edges, box) >= self.texture_ratio * ref_density:
                    continue  # looks like products, not an empty shelf (missed detections)
                missing = max(1, int(round((gx2 - gx1) / ref)))
                void_slots += missing
                rep.voids.append(Void(ri, box, missing))

        rep.capacity_estimate = rep.facings + void_slots
        rep.fill_ratio = rep.facings / max(1, rep.capacity_estimate)
        ratio = rep.fill_ratio
        if expected_facings:
            rep.stock_ratio_vs_planogram = min(1.0, rep.facings / expected_facings)
            ratio = min(ratio, rep.stock_ratio_vs_planogram)
            comp = [rep.stock_ratio_vs_planogram]
            if expected_rows:
                comp.append(1.0 - min(1.0, abs(rep.rows - expected_rows) / expected_rows))
                if rep.rows != expected_rows:
                    rep.issues.append(f"expected {expected_rows} shelf rows, detected {rep.rows}")
            rep.planogram_compliance = float(np.mean(comp))
        if rep.voids:
            rep.issues.append(f"{len(rep.voids)} empty slot(s), ~{void_slots} facing(s) missing")
        # status: severe shortage -> OUT_OF_STOCK, below threshold -> LOW_STOCK,
        # otherwise OK (individual voids are still listed in `issues` / `voids`)
        if rep.facings == 0 or ratio < self.low_stock_ratio * 0.5:
            rep.status = "OUT_OF_STOCK"
        elif ratio < self.low_stock_ratio:
            rep.status = "LOW_STOCK"
        elif rep.voids and ratio < 0.9:
            rep.status = "LOW_STOCK"
        else:
            rep.status = "OK"
        return rep


def baseline_voids(baseline_boxes, current_boxes, image=None, min_cover: float = 0.3,
                   texture_ratio: float = 0.0, merge_gap: float = 0.5) -> List[Void]:
    """Fixed-camera out-of-stock detection by comparison with a 'fully stocked' baseline.

    A baseline product position is *missing* when current detections cover less than
    `min_cover` of it. Optionally the region must also look empty (low edge density vs the
    current products). Neighbouring missing positions in the same row are merged into one void.
    This avoids the geometric guesswork of gap analysis (perspective, row clustering) because the
    camera and the shelf do not move.
    """
    base = np.asarray(baseline_boxes, dtype=np.float32).reshape(-1, 4) if len(baseline_boxes) else np.zeros((0, 4), np.float32)
    cur = np.asarray(current_boxes, dtype=np.float32).reshape(-1, 4) if len(current_boxes) else np.zeros((0, 4), np.float32)
    if len(base) == 0:
        return []
    edges = ref = None
    if image is not None and texture_ratio > 0 and len(cur):
        edges = ShelfAnalyzer._edge_map(image)
        sample = cur[np.linspace(0, len(cur) - 1, min(len(cur), 60)).astype(int)]
        ref = float(np.median([ShelfAnalyzer._density(edges, bx) for bx in sample]))
    missing = []
    for bx in base:
        cov = ShelfAnalyzer._covered(tuple(bx), cur) if len(cur) else 0.0
        if cov >= min_cover:
            continue
        if edges is not None and ref and ShelfAnalyzer._density(edges, bx) >= texture_ratio * ref:
            continue  # something with product-like texture is still there (missed detection / occlusion)
        missing.append(bx)
    if not missing:
        return []
    miss = np.array(missing)
    voids: List[Void] = []
    for ri, idx in enumerate(cluster_rows(miss, tol=0.5)):
        rb = miss[idx][np.argsort(miss[idx][:, 0])]
        w_med = float(np.median(rb[:, 2] - rb[:, 0]))
        start = 0
        for i in range(1, len(rb) + 1):
            if i == len(rb) or rb[i, 0] - rb[i - 1, 2] > merge_gap * w_med:
                grp = rb[start:i]
                voids.append(Void(ri, (float(grp[:, 0].min()), float(grp[:, 1].min()),
                                       float(grp[:, 2].max()), float(grp[:, 3].max())), len(grp)))
                start = i
    return voids


class SkuMatcher:
    """Lightweight appearance check for planogram placement (baseline).

    Reference crops per SKU group -> HSV colour histograms. A detected product is
    assigned to the closest reference; products whose group differs from the section's
    planned group are reported as misplaced. Replace with a metric-learning embedding
    model when a store's SKU catalogue images are available.
    """

    def __init__(self):
        self.refs: Dict[str, List[np.ndarray]] = {}

    @staticmethod
    def _hist(crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    def add_reference(self, group: str, crop_bgr: np.ndarray) -> None:
        self.refs.setdefault(group, []).append(self._hist(crop_bgr))

    def classify(self, crop_bgr: np.ndarray) -> Tuple[Optional[str], float]:
        import cv2

        if not self.refs or crop_bgr.size == 0:
            return None, 0.0
        h = self._hist(crop_bgr)
        best, score = None, -1.0
        for g, hs in self.refs.items():
            s = max(float(cv2.compareHist(h, r, cv2.HISTCMP_CORREL)) for r in hs)
            if s > score:
                best, score = g, s
        return best, score

    def misplaced(self, image_bgr: np.ndarray, boxes, planned_group: str, min_score: float = 0.5) -> List[int]:
        out = []
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=int).reshape(-1, 4)):
            g, s = self.classify(image_bgr[max(0, y1):y2, max(0, x1):x2])
            if g is not None and s >= min_score and g != planned_group:
                out.append(i)
        return out
