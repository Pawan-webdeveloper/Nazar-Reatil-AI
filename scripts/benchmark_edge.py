"""Edge benchmark: export models to ONNX and measure CPU latency + bandwidth savings.

Run when the machine is otherwise idle (latency numbers are meaningless under load):
    python scripts/benchmark_edge.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "benchmark"


def bench(model, img, n=40, **kw):
    for _ in range(5):
        model.predict(img, verbose=False, **kw)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(img, verbose=False, **kw)
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 1), round(float(np.percentile(ts, 90)), 1)


def main() -> int:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cpu": _cpu_name(), "models": {}}
    rng = np.random.default_rng(0)
    frame = (rng.random((432, 768, 3)) * 255).astype(np.uint8)
    shelf = (rng.random((1280, 720, 3)) * 255).astype(np.uint8)
    targets = [("yolov8n.pt", frame, 640), ("yolo11n.pt", frame, 640), ("yolo11s.pt", frame, 640),
               ("shelf_detector.pt", shelf, 640)]
    for name, img, sz in targets:
        w = ROOT / "models" / name
        if not w.exists():
            print("skip (missing)", w)
            continue
        pt = YOLO(str(w))
        onnx_path = pt.export(format="onnx", imgsz=sz, simplify=True, dynamic=False)
        onnx = YOLO(onnx_path, task="detect")
        p50, p90 = bench(pt, img, imgsz=sz, device="cpu", max_det=1000)
        o50, o90 = bench(onnx, img, imgsz=sz, device="cpu", max_det=1000)
        res["models"][name] = {"pytorch_ms_p50": p50, "pytorch_ms_p90": p90, "onnx_ms_p50": o50, "onnx_ms_p90": o90,
                               "max_fps_onnx": round(1000 / o50, 1),
                               "onnx_file_MB": round(Path(onnx_path).stat().st_size / 1e6, 1)}
        print(name, res["models"][name], flush=True)

    # bandwidth: what leaves the store per camera per day
    db_rows_per_min = 12              # typical minute aggregates per camera (measured schema)
    bytes_per_row = 120               # JSON row size incl. keys
    edge_mb_day = db_rows_per_min * bytes_per_row * 60 * 14 / 1e6
    cloud_mb_day = 2.0 * 3600 * 14 / 8  # 2 Mbit/s 1080p H.264 stream, 14 h opening hours
    res["bandwidth_per_camera_per_day"] = {"edge_analytics_MB": round(edge_mb_day, 2),
                                           "cloud_video_streaming_MB": round(cloud_mb_day, 0),
                                           "reduction_%": round(100 * (1 - edge_mb_day / cloud_mb_day), 3)}
    (OUT / "edge_benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def _cpu_name() -> str:
    import platform
    return platform.processor() or platform.machine()


if __name__ == "__main__":
    sys.exit(main())
