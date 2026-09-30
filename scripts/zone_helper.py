"""Grab one frame per camera and draw a pixel grid plus the configured lines / zones on it.

Use the saved images to read coordinates for `configs/store_config.yaml`
(count_line, zones, queue_polygon, service_polygon, shelf section polygons).

    python scripts/zone_helper.py                  # all cameras
    python scripts/zone_helper.py --camera cam_entrance --grid 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retail_ai.config import load_config  # noqa: E402
from retail_ai.edge.pipeline import FrameSource  # noqa: E402


def draw(frame, cam, grid):
    h, w = frame.shape[:2]
    if cam.role != "shelf" and w != cam.process_width:
        frame = cv2.resize(frame, (cam.process_width, int(round(h * cam.process_width / w))))
        h, w = frame.shape[:2]
    vis = frame.copy()
    for x in range(0, w, grid):
        cv2.line(vis, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(vis, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    for y in range(0, h, grid):
        cv2.line(vis, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(vis, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    vis = cv2.addWeighted(frame, 0.55, vis, 0.45, 0)
    if cam.count_line:
        (ax, ay), (bx, by) = cam.count_line
        cv2.line(vis, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 255), 3)
        cv2.putText(vis, f"count line (entry = moving {cam.entry_direction})", (int(ax) + 5, int(ay) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for z in cam.zones:
        cv2.polylines(vis, [np.array(z.polygon, np.int32)], True, (255, 180, 0), 2)
        cv2.putText(vis, z.name, tuple(int(v) + 5 for v in z.polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)
    for c in cam.counters:
        cv2.polylines(vis, [np.array(c.queue_polygon, np.int32)], True, (255, 0, 255), 2)
        cv2.polylines(vis, [np.array(c.service_polygon, np.int32)], True, (0, 128, 255), 2)
    for s in cam.shelf_sections:
        if s.polygon:
            cv2.polylines(vis, [np.array(s.polygon, np.int32)], True, (0, 200, 0), 2)
    cv2.putText(vis, f"{cam.id} ({cam.role}) {w}x{h}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "store_config.yaml"))
    ap.add_argument("--camera", default="")
    ap.add_argument("--grid", type=int, default=50)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = ROOT / "outputs" / "zone_helper"
    out.mkdir(parents=True, exist_ok=True)
    for cam in cfg.cameras:
        if a.camera and cam.id != a.camera:
            continue
        src_path = cam.source if (cam.source.isdigit() or "://" in cam.source) else str(cfg.resolve(cam.source))
        try:
            src = FrameSource(src_path, realtime=False)
            ok, frame = src.read()
            src.release()
        except Exception as exc:
            print(f"{cam.id}: cannot open {cam.source}: {exc}")
            continue
        if not ok:
            print(f"{cam.id}: no frame")
            continue
        path = out / f"{cam.id}.jpg"
        cv2.imwrite(str(path), draw(frame, cam, a.grid))
        print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
