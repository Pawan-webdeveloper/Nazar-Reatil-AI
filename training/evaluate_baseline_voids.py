"""Evaluate fixed-camera out-of-stock detection by comparison with a 'fully stocked' baseline.

For each shelf photo:
  baseline = detections on the original photo (staff pressed "Mark as full")
  current  = detections on a later snapshot of the same shelf in which 3-6 consecutive products
             were removed, PLUS realistic capture changes: brightness/contrast +-20 %, sensor noise,
             camera shake of up to +-3 px and JPEG re-compression
Metrics
  void_recall                removed region covered >= 50 % by reported voids
  void_precision             reported voids that overlap the removed region / all reported voids
  false_voids_per_image      voids reported elsewhere (nothing was removed there)
  false_voids_no_change      voids reported when NOTHING was removed, only capture changes
Parameters are tuned on the validation split and reported once on the test split.

Usage: python training/evaluate_baseline_voids.py --data <sku110k.yaml> [--images 100]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from retail_ai.analytics.shelf import baseline_voids  # noqa: E402
from evaluate_shelf import make_void, read_labels, void_hits  # noqa: E402

OUT = ROOT / "outputs" / "shelf_eval"


def capture_change(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0.8, 1.2)
    b = rng.uniform(-20, 20)
    out = np.clip(img.astype(np.float32) * a + b + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    dx, dy = rng.integers(-3, 4, 2)
    out = cv2.warpAffine(out, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]),
                         borderMode=cv2.BORDER_REPLICATE)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 95))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def collect(model, paths, limit, seed, imgsz, conf):
    """Run the detector once per image; return cached detections for the parameter sweep."""
    rng = np.random.default_rng(seed)
    cache = []
    for p in paths:
        if len(cache) >= limit:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        made = make_void(img, read_labels(Path(str(p).replace("images", "labels")).with_suffix(".txt"), w, h), rng)
        if made is None:
            continue
        mod, region, _ = made
        cur = capture_change(mod, rng)
        same = capture_change(img, rng)  # nothing removed, only capture differences
        det = lambda im: model.predict(im, imgsz=imgsz, conf=conf, max_det=1000, verbose=False,  # noqa: E731
                                       device="cpu")[0].boxes.xyxy.cpu().numpy()
        cache.append({"base": det(img), "cur": det(cur), "cur_img": cur, "same": det(same), "same_img": same,
                      "region": region})
    return cache


def score(cache, min_cover, tex):
    found = tp = allv = fp = fp_same = 0
    for c in cache:
        v = baseline_voids(c["base"], c["cur"], c["cur_img"], min_cover, tex)
        ok, matched = void_hits(v, c["region"])
        found += int(ok)
        tp += matched
        allv += len(v)
        fp += len(v) - matched
        fp_same += len(baseline_voids(c["base"], c["same"], c["same_img"], min_cover, tex))
    n = max(1, len(cache))
    rec, prec = found / n, tp / max(1, allv)
    return {"min_cover": min_cover, "texture_ratio": tex, "void_recall": round(rec, 3),
            "void_precision": round(prec, 3), "f1": round(2 * rec * prec / max(1e-9, rec + prec), 3),
            "false_voids_per_image": round(fp / n, 2), "false_voids_no_change": round(fp_same / n, 2), "images": n}


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "shelf_detector.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg["path"])
    model = YOLO(a.weights)
    grid = [(mc, tx) for mc in (0.2, 0.3, 0.5) for tx in (0.0, 0.5, 0.7)]

    print("validation: detecting ...", flush=True)
    val = collect(model, sorted((root / cfg["val"]).glob("*.jpg")), a.images, 1, a.imgsz, a.conf)
    sweep = {f"cover{mc}_tex{tx}": score(val, mc, tx) for mc, tx in grid}
    for k, v in sweep.items():
        print(k, v, flush=True)
    best = max(sweep.values(), key=lambda v: v["f1"])
    print("selected:", best["min_cover"], best["texture_ratio"], flush=True)

    print("test: detecting ...", flush=True)
    test = collect(model, sorted((root / cfg["test"]).glob("*.jpg")), a.images, 2, a.imgsz, a.conf)
    res = score(test, best["min_cover"], best["texture_ratio"])
    out = {"method": "baseline comparison (fixed camera)", "conf": a.conf, "validation_sweep": sweep,
           "selected": {"min_cover": best["min_cover"], "texture_ratio": best["texture_ratio"]}, "test": res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_void_eval.json").write_text(json.dumps(out, indent=2))
    print("TEST:", json.dumps(res), flush=True)

    # illustration
    c = test[0]
    v = baseline_voids(c["base"], c["cur"], c["cur_img"], best["min_cover"], best["texture_ratio"])
    vis = c["cur_img"].copy()
    for x1, y1, x2, y2 in c["cur"].astype(int):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    for vd in v:
        x1, y1, x2, y2 = map(int, vd.box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 5)
    rx1, ry1, rx2, ry2 = map(int, c["region"])
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
    cv2.imwrite(str(OUT / "baseline_void_example.jpg"), cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
