"""Evaluate the shelf detector + out-of-stock (void) logic.

1. Detection quality on the untouched SKU-110K TEST split (mAP50, mAP50-95, P, R).
2. Product-count accuracy per image (what stock-level estimation depends on), with
   robustness slices by lighting and shelf density (bias check).
3. Out-of-stock (void) detection: on real shelf photos we remove k consecutive
   products from one row (inpainting or dark "empty shelf" fill), then check whether
   the pipeline reports a void there.
     - gap_ratio threshold is TUNED on the validation split, REPORTED on the test split
     - recall: synthetic empty slots found (void covers >= 50 % of the removed region)
     - precision (lower bound): voids overlapping a removed region / all voids reported.
       It is a lower bound because genuine gaps already present in the photos are counted
       as false positives.

Usage: python training/evaluate_shelf.py --weights models/shelf_detector.pt --data <sku110k.yaml>
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
from retail_ai.analytics.shelf import ShelfAnalyzer, cluster_rows  # noqa: E402

OUT = ROOT / "outputs" / "shelf_eval"


def read_labels(lbl: Path, w: int, h: int) -> np.ndarray:
    rows = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            _, cx, cy, bw, bh = map(float, p)
            rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def make_void(img: np.ndarray, gt: np.ndarray, rng: np.random.Generator):
    """Remove 3-6 consecutive products in a random well-populated row. Returns (img, region) or None."""
    rows = [r for r in cluster_rows(gt) if len(r) >= 8]
    if not rows:
        return None
    r = rows[rng.integers(len(rows))]
    rb = gt[r][np.argsort(gt[r][:, 0])]
    k = int(rng.integers(3, 7))
    s = int(rng.integers(1, len(rb) - k)) if len(rb) - k > 1 else 0
    sel = rb[s:s + k]
    x1, x2 = float(sel[:, 0].min()), float(sel[:, 2].max())
    y1, y2 = float(np.median(sel[:, 1])), float(np.median(sel[:, 3]))
    out = img.copy()
    mask = np.zeros(img.shape[:2], np.uint8)
    for b in sel:  # remove each product box fully
        bx1, by1, bx2, by2 = b.astype(int)
        mask[max(0, by1):by2, max(0, bx1):bx2] = 255
    if rng.random() < 0.5:
        out = cv2.inpaint(out, mask, 7, cv2.INPAINT_TELEA)
    else:  # dark empty shelf back with texture
        shade = np.clip(img[mask > 0].mean(axis=0) * 0.35, 10, 90)
        noise = rng.normal(0, 6, (int(mask.sum() // 255), 3))
        out[mask > 0] = np.clip(shade + noise, 0, 255).astype(np.uint8)
        out = np.where(mask[..., None] > 0, cv2.GaussianBlur(out, (5, 5), 0), out)
    # all remaining GT boxes (for reference); removed ones excluded
    keep = np.ones(len(gt), bool)
    for b in sel:
        keep &= ~np.all(np.isclose(gt, b), axis=1)
    return out, (x1, y1, x2, y2), gt[keep]


def void_hits(voids, region, min_cover=0.5):
    rx1, ry1, rx2, ry2 = region
    area = max(1e-6, (rx2 - rx1) * (ry2 - ry1))
    hit_any = False
    matched = 0
    for v in voids:
        vx1, vy1, vx2, vy2 = v.box
        iw = max(0.0, min(rx2, vx2) - max(rx1, vx1))
        ih = max(0.0, min(ry2, vy2) - max(ry1, vy1))
        if iw * ih > 0:
            matched += 1
        if iw * ih / area >= min_cover:
            hit_any = True
        # a wide void may be split in two by a stray detection -> accumulate coverage
    cover = sum(max(0.0, min(rx2, v.box[2]) - max(rx1, v.box[0])) * max(0.0, min(ry2, v.box[3]) - max(ry1, v.box[1]))
                for v in voids) / area
    return hit_any or cover >= min_cover, matched


def run_void_eval(model, img_paths, ratios, rng_seed, imgsz, conf, limit):
    """ratios: list of (gap_ratio, texture_ratio). Detections are computed once per image."""
    rng = np.random.default_rng(rng_seed)
    per_ratio = {r: {"found": 0, "total": 0, "tp_voids": 0, "all_voids": 0, "clean_voids": 0, "clean_imgs": 0}
                 for r in ratios}
    used = 0
    for p in img_paths:
        if used >= limit:
            break
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_labels(Path(str(p).replace("images", "labels")).with_suffix(".txt"), w, h)
        made = make_void(img, gt, rng)
        if made is None:
            continue
        mod, region, _ = made
        used += 1
        det_mod = model.predict(mod, imgsz=imgsz, conf=conf, max_det=1000, verbose=False, device="cpu")[0].boxes.xyxy.cpu().numpy()
        det_org = model.predict(img, imgsz=imgsz, conf=conf, max_det=1000, verbose=False, device="cpu")[0].boxes.xyxy.cpu().numpy()
        for r in ratios:
            an = ShelfAnalyzer(gap_ratio=r[0], texture_ratio=r[1])
            rep = an.analyze(det_mod, (w, h), image=mod)
            ok, matched = void_hits(rep.voids, region)
            s = per_ratio[r]
            s["found"] += int(ok)
            s["total"] += 1
            s["tp_voids"] += matched
            s["all_voids"] += len(rep.voids)
            s["clean_voids"] += len(an.analyze(det_org, (w, h), image=img).voids)
            s["clean_imgs"] += 1
    out = {}
    for r, s in per_ratio.items():
        rec = s["found"] / max(1, s["total"])
        prec = s["tp_voids"] / max(1, s["all_voids"])
        out[f"gap{r[0]}_tex{r[1]}"] = {"gap_ratio": r[0], "texture_ratio": r[1],
                       "void_recall": round(rec, 3), "void_precision_lower_bound": round(prec, 3),
                       "f1": round(2 * rec * prec / max(1e-9, rec + prec), 3),
                       "voids_per_unmodified_image": round(s["clean_voids"] / max(1, s["clean_imgs"]), 2),
                       "images": s["total"]}
    return out


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "shelf_detector.pt"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--void-images", type=int, default=120)
    ap.add_argument("--tag", default="")
    ap.add_argument("--void-only", action="store_true", help="skip mAP / counting, only evaluate void detection")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.data).read_text())
    root = Path(cfg["path"])
    model = YOLO(a.weights)
    test_imgs = sorted((root / cfg["test"]).glob("*.jpg"))
    det = count = slices = None
    if a.void_only:
        prev = OUT / f"shelf_eval{a.tag}.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            det, count, slices = old.get("detection_test"), old.get("counting_test"), old.get("robustness_slices")
    else:
        det, count, slices = _detection_and_counting(model, a, test_imgs)

    print("3) void (out-of-stock) detection: tune on VAL, report on TEST ...", flush=True)
    combos = [(g, t) for g in (1.0, 1.5, 2.0) for t in (0.4, 0.6, 0.8, 99.0)]  # 99 = texture check off
    val_imgs = sorted((root / cfg["val"]).glob("*.jpg"))
    val_res = run_void_eval(model, val_imgs, combos, 1, a.imgsz, a.conf, a.void_images)
    best_key = max(val_res, key=lambda k: val_res[k]["f1"])
    best = (val_res[best_key]["gap_ratio"], val_res[best_key]["texture_ratio"])
    test_res = run_void_eval(model, test_imgs, [best], 2, a.imgsz, a.conf, a.void_images)
    print("val sweep:", json.dumps(val_res, indent=1), "\nselected (gap, texture):", best,
          "\ntest:", json.dumps(test_res), flush=True)

    summary = {"weights": str(a.weights), "conf": a.conf, "detection_test": det, "counting_test": count,
               "robustness_slices": slices,
               "void_detection": {"validation_sweep": val_res, "selected_gap_ratio": best[0],
                                  "selected_texture_ratio": best[1], "test": list(test_res.values())[0]}}
    (OUT / f"shelf_eval{a.tag}.json").write_text(json.dumps(summary, indent=2))
    _example(model, test_imgs, best, a)
    print(json.dumps(summary, indent=2))
    return 0


def _detection_and_counting(model, a, test_imgs):
    print("1) detection metrics on TEST split ...", flush=True)
    r = model.val(data=a.data, split="test", imgsz=a.imgsz, batch=4, max_det=1000, device="cpu", workers=0,
                  plots=True, project=str(OUT), name=f"test{a.tag}", exist_ok=True, verbose=False)
    det = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
           "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}
    print(det, flush=True)

    print("2) counting accuracy + robustness slices ...", flush=True)
    rows = []
    for p in test_imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        n_gt = len(read_labels(Path(str(p).replace("images", "labels")).with_suffix(".txt"), w, h))
        n_pr = len(model.predict(img, imgsz=a.imgsz, conf=a.conf, max_det=1000, verbose=False, device="cpu")[0].boxes)
        rows.append((n_gt, n_pr, float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())))
    gt, pr, br = (np.array(x, dtype=float) for x in zip(*rows))
    ape = np.abs(gt - pr) / np.maximum(gt, 1)
    count = {"images": len(rows), "count_MAE": round(float(np.abs(gt - pr).mean()), 2),
             "count_accuracy_%": round(float(100 * (1 - ape.mean())), 2),
             "within_10pct_%": round(float(100 * (ape <= 0.10).mean()), 1)}
    slices = {}
    for label, m in {"dark (<90)": br < 90, "normal (90-160)": (br >= 90) & (br < 160), "bright (>=160)": br >= 160,
                     "sparse (<100 items)": gt < 100, "medium (100-199)": (gt >= 100) & (gt < 200),
                     "dense (>=200)": gt >= 200}.items():
        if m.sum():
            slices[label] = {"images": int(m.sum()), "count_accuracy_%": round(float(100 * (1 - ape[m].mean())), 2)}
    print(count, json.dumps(slices), flush=True)
    return det, count, slices


def _example(model, imgs, ratio, a):
    """Save an illustrated example: detections (green) and detected voids (red)."""
    rng = np.random.default_rng(5)
    for p in imgs[:40]:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        made = make_void(img, read_labels(Path(str(p).replace("images", "labels")).with_suffix(".txt"), w, h), rng)
        if made is None:
            continue
        mod, region, _ = made
        det = model.predict(mod, imgsz=a.imgsz, conf=a.conf, max_det=1000, verbose=False, device="cpu")[0].boxes.xyxy.cpu().numpy()
        rep = ShelfAnalyzer(gap_ratio=ratio[0], texture_ratio=ratio[1]).analyze(det, (w, h), image=mod)
        vis = mod.copy()
        for x1, y1, x2, y2 in det.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for v in rep.voids:
            x1, y1, x2, y2 = map(int, v.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
        rx1, ry1, rx2, ry2 = map(int, region)
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
        cv2.putText(vis, f"{rep.facings} products | {len(rep.voids)} voids | {rep.status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.imwrite(str(OUT / f"void_example{a.tag}.jpg"), cv2.resize(vis, (w // 2, h // 2)))
        return


if __name__ == "__main__":
    sys.exit(main())
