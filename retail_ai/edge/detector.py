"""On-device detectors (Ultralytics YOLO; .pt, .onnx and OpenVINO exports all work).

* PersonTracker - COCO person class + ByteTrack multi-object tracking. Only boxes and
  anonymous integer track ids leave this module (no crops, no embeddings).
* ProductDetector - single-class "product" detector trained on SKU-110K.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import ROOT

log = logging.getLogger("retail_ai.detector")


def _resolve_weights(path: str, fallback: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        return str(p)
    fb = ROOT / "models" / fallback
    log.warning("weights %s not found, falling back to %s", p, fb.name)
    return str(fb) if fb.exists() else fallback  # ultralytics downloads official weights if needed


class PersonTracker:
    def __init__(self, weights: str, conf: float = 0.35, imgsz: int = 640, device: str = "cpu",
                 tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO

        self.model = YOLO(_resolve_weights(weights, "yolo11n.pt"))
        self.conf, self.imgsz, self.device, self.tracker = conf, imgsz, device, tracker
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.person_cls = next((k for k, v in names.items() if v == "person"), 0)

    def track(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray, float]]:
        """Returns [(track_id, xyxy, conf)] for confirmed tracks in this frame."""
        res = self.model.track(frame, persist=True, tracker=self.tracker, classes=[self.person_cls],
                               conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.boxes is None or res.boxes.id is None:
            return []
        ids = res.boxes.id.int().cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        return [(int(i), b, float(c)) for i, b, c in zip(ids, xyxy, confs)]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        res = self.model.predict(frame, classes=[self.person_cls], conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        return res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))


class ProductDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(_resolve_weights(weights, "shelf_detector.pt"))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 max_det=1000, verbose=False)[0]
        if res.boxes is None:
            return np.zeros((0, 4)), np.zeros((0,))
        return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
