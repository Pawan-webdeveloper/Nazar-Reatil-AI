"""Compare pretrained person detectors for the shopper-analytics cameras.

Data  : COCO 2017 validation images (Hugging Face `detection-datasets/coco`, never used
        for training the official YOLO weights) -> person class only, plus person-free
        images to measure false alarms.
Models: YOLOv8n, YOLO11n, YOLO11s (COCO-pretrained; person is a COCO class, so no
        retraining is needed - fine-tuning on store footage is optional).
Output: AP50 / AP50-95 (Ultralytics validator), recall/precision at the deployed
        confidence, CPU latency, and robustness slices (person size, crowd density,
        low light) to expose any systematic bias.

Usage: python training/evaluate_person_detectors.py --parquet <coco val parquet> --n 600
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retail_ai.geometry import greedy_match, iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "person_detector"


def build_dataset(parquet: Path, dst: Path, n_pos: int, n_neg: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    (dst / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "val").mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    pos, neg, meta = 0, 0, []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["image_id", "image", "width", "height", "objects"])
        for iid, im, w, h, ob in zip(*(t.column(c).to_pylist() for c in ("image_id", "image", "width", "height", "objects"))):
            persons = [b for b, c in zip(ob["bbox"], ob["category"]) if c == 0]
            if persons and pos >= n_pos:
                continue
            if not persons and (neg >= n_neg or rng.random() > 0.3):
                continue
            img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            W, H = img.size
            lines = []
            for x1, y1, x2, y2 in persons:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            name = f"coco_{iid:012d}"
            img.save(dst / "images" / "val" / f"{name}.jpg", quality=95)
            (dst / "labels" / "val" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            meta.append({"name": name, "persons": [[float(v) for v in b] for b in persons], "w": W, "h": H,
                         "brightness": float(gray.mean())})
            if persons:
                pos += 1
            else:
                neg += 1
        if pos >= n_pos and neg >= n_neg:
            break
    (dst / "person.yaml").write_text(f"path: {dst.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n  0: person\n")
    (dst / "meta.json").write_text(json.dumps(meta))
    print(f"dataset: {pos} images with people, {neg} without")
    return dst / "person.yaml", meta


def slice_eval(model, meta, img_dir: Path, conf: float, imgsz: int):
    """Recall per slice at IoU 0.5 and deployed confidence; false positives on empty images."""
    stats = {k: [0, 0] for k in ("all", "small(<32px)", "medium", "large(>96px)", "crowd(>=5)", "sparse(<5)",
                                   "dark(<80)", "normal_light")}
    fp_total, tp_total, neg_fp_imgs, neg_imgs = 0, 0, 0, 0
    for m in meta:
        res = model.predict(str(img_dir / f"{m['name']}.jpg"), classes=[0], conf=conf, imgsz=imgsz,
                            device="cpu", verbose=False)[0]
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        gt = np.array(m["persons"], dtype=np.float32).reshape(-1, 4)
        if len(gt) == 0:
            neg_imgs += 1
            neg_fp_imgs += int(len(pred) > 0)
            fp_total += len(pred)
            continue
        pairs = greedy_match(iou_matrix(gt, pred), 0.5)
        matched = {i for i, _ in pairs}
        tp_total += len(pairs)
        fp_total += len(pred) - len(pairs)
        crowd = len(gt) >= 5
        dark = m["brightness"] < 80
        for i, b in enumerate(gt):
            area = (b[2] - b[0]) * (b[3] - b[1])
            size = "small(<32px)" if area < 32 ** 2 else ("large(>96px)" if area > 96 ** 2 else "medium")
            hit = int(i in matched)
            for k in ("all", size, "crowd(>=5)" if crowd else "sparse(<5)", "dark(<80)" if dark else "normal_light"):
                stats[k][0] += hit
                stats[k][1] += 1
    out = {k: {"recall": round(h / n, 3) if n else None, "persons": n} for k, (h, n) in stats.items()}
    out["precision_at_conf"] = round(tp_total / max(1, tp_total + fp_total), 3)
    out["false_alarm_rate_on_empty_images"] = round(neg_fp_imgs / max(1, neg_imgs), 3)
    return out


def latency(model, imgsz: int, n: int = 30) -> float:
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
    for _ in range(3):
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main(argv=None) -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--data-dir", default=str(Path.home() / "retail_ai_data" / "coco_person"))
    ap.add_argument("--n", type=int, default=600, help="images containing people")
    ap.add_argument("--n-neg", type=int, default=150, help="images without people")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo11s.pt"])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    data_yaml, meta = build_dataset(Path(a.parquet), Path(a.data_dir), a.n, a.n_neg)

    results = {}
    for name in a.models:
        w = ROOT / "models" / name
        model = YOLO(str(w if w.exists() else name))
        t = time.time()
        r = model.val(data=str(data_yaml), split="val", classes=[0], imgsz=a.imgsz, batch=8, device="cpu",
                      plots=False, verbose=False, workers=0)
        res = {"AP50": round(float(r.box.ap50[0]), 4), "AP50_95": round(float(r.box.ap[0]), 4),
               "cpu_latency_ms": round(latency(model, a.imgsz), 1),
               "slices": slice_eval(model, meta, Path(a.data_dir) / "images" / "val", a.conf, a.imgsz)}
        res["eval_minutes"] = round((time.time() - t) / 60, 1)
        results[name] = res
        print(name, json.dumps(res), flush=True)

    # selection: edge cameras run ~10 FPS on CPU -> best AP50-95 among models under 120 ms
    eligible = {k: v for k, v in results.items() if v["cpu_latency_ms"] <= 120} or results
    best = max(eligible, key=lambda k: eligible[k]["AP50_95"])
    summary = {"dataset": f"COCO val2017 subset: {a.n} images with people + {a.n_neg} without",
               "conf": a.conf, "imgsz": a.imgsz, "results": results, "selected": best,
               "rule": "highest person AP50-95 with CPU latency <= 120 ms (10 FPS budget)"}
    (OUT / "person_detector_comparison.json").write_text(json.dumps(summary, indent=2))
    import shutil
    src = ROOT / "models" / best
    if src.exists():
        shutil.copy(src, ROOT / "models" / "person_detector.pt")
    print("selected:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
