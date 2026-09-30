"""Movement heatmap: foot points accumulated on a coarse grid (no images stored)."""
from __future__ import annotations

import numpy as np


class Heatmap:
    def __init__(self, width: int, height: int, cell: int = 16):
        self.cell = max(4, int(cell))
        self.w, self.h = int(width), int(height)
        self.grid = np.zeros((self.h // self.cell + 1, self.w // self.cell + 1), dtype=np.float64)

    def add(self, points, weight: float = 1.0) -> None:
        for x, y in points:
            gx, gy = int(x) // self.cell, int(y) // self.cell
            if 0 <= gy < self.grid.shape[0] and 0 <= gx < self.grid.shape[1]:
                self.grid[gy, gx] += weight

    def normalized(self) -> np.ndarray:
        g = self._smooth(self.grid)
        m = g.max()
        return g / m if m > 0 else g

    @staticmethod
    def _smooth(g: np.ndarray) -> np.ndarray:
        k = np.array([1, 4, 6, 4, 1], dtype=np.float64)
        k /= k.sum()
        pad = np.pad(g, 2, mode="edge")
        tmp = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)
        return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, tmp)

    def render(self, background: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
        """BGR uint8 visualisation (JET-like colour map), optionally over a background."""
        import cv2

        norm = (self.normalized() * 255).astype(np.uint8)
        norm = cv2.resize(norm, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if background is None:
            return color
        bg = cv2.resize(background, (self.w, self.h))
        return cv2.addWeighted(bg, 1 - alpha, color, alpha, 0)

    def to_list(self):
        return self.grid.tolist()

    def merge(self, other_grid) -> None:
        og = np.asarray(other_grid, dtype=np.float64)
        if og.shape == self.grid.shape:
            self.grid += og
