"""Train / compare shelf product detectors on SKU-110K (YOLO format).

Modes
-----
screen : train every candidate with identical small budget, measure val mAP and
         CPU latency, pick the best accuracy/latency trade-off for edge hardware.
final  : train one model with the full budget, copy best weights to models/.

Examples
--------
python training/train_shelf_detector.py screen --data D:/sku110k_yolo/sku110k.yaml
python training/train_shelf_detector.py final  --data ... --model yolo11n.pt --epochs 20
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "training"


def cpu_latency_ms(weights: str, imgsz: int, n: int = 20) -> float:
    """Median single-image CPU latency (what an edge box without GPU sees)."""
    from ultralytics import YOLO
    import torch

    m = YOLO(weights)
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)
    for _ in range(3):
        m.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    with torch.inference_mode():
        for _ in range(n):
            t = time.perf_counter()
            m.predict(img, imgsz=imgsz, device="cpu", verbose=False, max_det=1000)
            times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def make_subset_yaml(data_yaml: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    """Write a yaml that uses a fixed random subset of train/val images (txt lists)."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    rng = random.Random(seed)
    out = {}
    for split, n in (("train", n_train), ("val", n_val)):
        files = sorted((root / cfg[split]).glob("*.jpg"))
        rng.shuffle(files)
        lst = root / f"subset_{split}_{n}.txt"
        lst.write_text("\n".join(str(f) for f in files[:n]) + "\n")
        out[split] = str(lst)
    sub = dict(cfg)
    sub.update(out)
    sub_path = root / f"subset_{n_train}_{n_val}.yaml"
    sub_path.write_text(yaml.safe_dump(sub))
    return sub_path


def train_one(model: str, data: str, epochs: int, imgsz: int, batch: int, name: str,
              workers: int, device: str, patience: int, extra: dict | None = None):
    from ultralytics import YOLO

    m = YOLO(model)
    args = dict(data=data, epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, device=device,
                project=str(OUT), name=name, exist_ok=True, seed=0, deterministic=True, max_det=1000,
                patience=patience, cos_lr=True, plots=True, verbose=True, amp=False,
                # dense retail shelves: products are small and many -> keep mosaic but close it at the end,
                # mild HSV/brightness jitter so store lighting differences do not bias the model
                close_mosaic=max(1, epochs // 5), hsv_v=0.4, hsv_s=0.6, degrees=0.0, fliplr=0.5, scale=0.4)
    if extra:
        args.update(extra)
    m.train(**args)
    return OUT / name / "weights" / "best.pt"


def evaluate(weights: Path, data: str, imgsz: int, split: str = "val") -> dict:
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=data, split=split, imgsz=imgsz, batch=4, max_det=1000,
                               device="cpu", plots=False, verbose=False, workers=0)
    return {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}


def _epochs_done(run_dir: Path) -> int:
    csv = run_dir / "results.csv"
    return max(0, len(csv.read_text().strip().splitlines()) - 1) if csv.exists() else 0


def cmd_screen(a) -> int:
    """Resumable: finished candidates are skipped, an interrupted one resumes from last.pt.
    Latency is measured only after ALL training is done, so every model is timed on an idle CPU."""
    from ultralytics import YOLO

    sub = make_subset_yaml(Path(a.data), a.n_train, a.n_val)
    bests = {}
    for cand in a.candidates:
        name = "screen_" + Path(cand).stem
        run_dir = OUT / name
        done = _epochs_done(run_dir)
        print(f"===== screening {cand} ({done}/{a.epochs} epochs done) =====", flush=True)
        t0 = time.time()
        if done >= a.epochs and (run_dir / "weights" / "best.pt").exists():
            pass
        elif (run_dir / "weights" / "last.pt").exists() and done > 0:
            YOLO(str(run_dir / "weights" / "last.pt")).train(resume=True)
        else:
            train_one(cand, str(sub), a.epochs, a.imgsz, a.batch, name, a.workers, "cpu", patience=100)
        bests[cand] = run_dir / "weights" / "best.pt"
        print(f"{cand}: training finished ({(time.time() - t0) / 60:.1f} min this session)", flush=True)
    results = {}
    for cand, best in bests.items():
        metrics = evaluate(best, str(sub), a.imgsz)
        metrics["epochs"] = _epochs_done(best.parent.parent)
        metrics["cpu_latency_ms"] = round(cpu_latency_ms(str(best), a.imgsz), 1)
        results[cand] = metrics
        print(cand, metrics, flush=True)
    # winner rule: highest mAP50-95; if another model is within 0.015 and >=1.5x faster, prefer it
    ranked = sorted(results.items(), key=lambda kv: kv[1]["mAP50_95"], reverse=True)
    best_c, best_v = ranked[0]
    winner = best_c
    for c, v in ranked[1:]:
        if best_v["mAP50_95"] - v["mAP50_95"] <= 0.015 and v["cpu_latency_ms"] * 1.5 <= results[winner]["cpu_latency_ms"]:
            winner = c
    results = {Path(k).name: v for k, v in results.items()}
    winner = Path(winner).name
    summary = {"subset": {"train": a.n_train, "val": a.n_val}, "epochs": a.epochs, "imgsz": a.imgsz,
               "results": results, "winner": winner,
               "rule": "max mAP50-95; a >=1.5x faster model within 1.5 mAP points wins (edge CPU)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screening_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_final(a) -> int:
    data = a.data
    if a.n_train:
        data = str(make_subset_yaml(Path(a.data), a.n_train, a.n_val or 10_000))
    name = a.name or ("final_" + Path(a.model).stem)
    last = OUT / name / "weights" / "last.pt"
    if a.resume and last.exists():
        from ultralytics import YOLO
        YOLO(str(last)).train(resume=True)
        best = OUT / name / "weights" / "best.pt"
    else:
        best = train_one(a.model, data, a.epochs, a.imgsz, a.batch, name, a.workers, a.device, a.patience)
    dst = ROOT / "models" / "shelf_detector.pt"
    dst.parent.mkdir(exist_ok=True)
    shutil.copy(best, dst)
    print("copied best weights ->", dst)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("screen")
    s.add_argument("--data", required=True)
    s.add_argument("--candidates", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolo11s.pt"])
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--n-train", type=int, default=300)
    s.add_argument("--n-val", type=int, default=120)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--workers", type=int, default=2)
    f = sp.add_parser("final")
    f.add_argument("--data", required=True)
    f.add_argument("--model", default="yolo11n.pt")
    f.add_argument("--epochs", type=int, default=20)
    f.add_argument("--patience", type=int, default=8)
    f.add_argument("--imgsz", type=int, default=640)
    f.add_argument("--batch", type=int, default=8)
    f.add_argument("--workers", type=int, default=2)
    f.add_argument("--device", default="cpu")
    f.add_argument("--n-train", type=int, default=0, help="0 = all train images")
    f.add_argument("--n-val", type=int, default=0)
    f.add_argument("--name", default="")
    f.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    return cmd_screen(a) if a.cmd == "screen" else cmd_final(a)


if __name__ == "__main__":
    sys.exit(main())
