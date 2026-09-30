"""Generates inventory.ipynb - the Colab T4 GPU training notebook (RESUMABLE).

Run:  python scripts/build_colab_notebook.py

Everything that matters is saved to Google Drive (MyDrive/retail_ai):
  * the converted dataset (zip)      -> a new runtime unzips it in ~1 min instead of re-downloading
  * every training run (last.pt each epoch)
  * a progress.json with finished stages
So if Colab disconnects or you stop it, just press Run All again: it continues from the last epoch.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# RetailSense AI - Shelf detector training on Colab T4 GPU (auto-resume)

**How to run (VS Code):** `Select Kernel` -> `Colab` -> choose **T4 GPU** -> **Run All**.
Allow Google Drive access when asked (needed so training survives disconnects).

**Stopped or disconnected?** Just press **Run All** again. Nothing restarts from zero:
the dataset, finished stages and the last epoch checkpoint are loaded from Drive (`MyDrive/retail_ai`).

What it does:
1. SKU-110K dataset (11,743 dense shelf images, ~1.7M boxes) -> YOLO format, cached on Drive shard by shard
2. Screening: YOLOv8n vs YOLO11n vs YOLO11s, identical budget, compared on validation
3. Full training of the winner with early stopping
4. Held-out TEST evaluation: mAP, count accuracy, robustness (lighting / density) slices
5. ONNX export + weights saved to Drive and embedded in this notebook's output
""")

code("""
# ---- Settings ----
SCREEN_EPOCHS   = 10      # identical short run per candidate
SCREEN_FRACTION = 0.35    # share of train images used for screening (saves GPU time)
FINAL_EPOCHS    = 50      # winner, early-stopped with PATIENCE
PATIENCE        = 10
IMGSZ           = 640
BATCH           = 16      # T4 16 GB
CANDIDATES      = ["yolov8n.pt", "yolo11n.pt", "yolo11s.pt"]
TRAIN_SHARDS, VAL_SHARDS, TEST_SHARDS = 19, 2, 7
""")

code(r'''
# ---- 1. GPU check FIRST (nothing is installed unless we are on a Colab GPU) ----
import subprocess, sys, os, json, time, shutil, pathlib
on_colab = os.path.exists("/content")
try:
    import torch
    gpu = torch.cuda.is_available()
except ImportError:
    gpu = False
print("Running on Colab:", on_colab, "| GPU available:", gpu)
if not (on_colab and gpu):
    raise SystemExit("STOP: this kernel is NOT a Colab GPU. Click the kernel name (top right) -> "
                     "Select Another Kernel -> Colab -> New Colab Server -> GPU -> T4, then Run All again.")
print("GPU:", torch.cuda.get_device_name(0))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pyarrow", "onnx", "onnxslim"], check=True)
DEVICE = 0
''')

code(r'''
# ---- 2. Persistent storage on Google Drive (this is what makes resume work) ----
PERSIST = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = pathlib.Path("/content/drive/MyDrive/retail_ai")
except Exception as e:
    print("!! Google Drive could not be mounted:", e)
    print("!! Training will still run, but a disconnect would lose progress.")
    PERSIST = pathlib.Path("/content/retail_persist")
PERSIST.mkdir(parents=True, exist_ok=True)
RUNS = PERSIST / "runs"; RUNS.mkdir(exist_ok=True)
STATE_FILE = PERSIST / "progress.json"
STATE = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
def save_state():
    STATE_FILE.write_text(json.dumps(STATE, indent=2))
print("Persistent folder:", PERSIST)
print("Progress so far:", json.dumps(STATE, indent=1) if STATE else "starting fresh")
''')

code(r'''
# ---- 3. Dataset (resumable shard by shard) ----
# * a shard that is already converted (zip on Drive) is just unpacked      -> seconds
# * a parquet shard already on your Drive (any MyDrive folder) is reused    -> no re-download
# * otherwise the shard is downloaded to the Colab disk, converted, deleted -> Drive space stays small
import io, glob, urllib.request, zipfile
import pyarrow.parquet as pq
from PIL import Image
LOCAL = pathlib.Path("/content/retail"); LOCAL.mkdir(exist_ok=True)
DST = LOCAL / "sku110k_yolo"; DATA_YAML = DST / "sku110k.yaml"
CONV = PERSIST / "converted"; CONV.mkdir(exist_ok=True)
BASE = "https://huggingface.co/api/datasets/benjamintli/sku110k/parquet/default"
SPLIT = {"train": "train", "validation": "val", "test": "test"}
jobs = [("validation", i) for i in range(VAL_SHARDS)] + [("train", i) for i in range(TRAIN_SHARDS)] + [("test", i) for i in range(TEST_SHARDS)]

def valid_parquet(f):
    try:
        return pq.ParquetFile(f).metadata.num_rows > 0
    except Exception:
        return False

# parquet shards you already have on Drive (e.g. from an earlier download)
existing = {}
for f in glob.glob("/content/drive/MyDrive/**/*.parquet", recursive=True):
    existing.setdefault(pathlib.Path(f).name, f)
print(f"parquet shards found on Drive: {len(existing)}")

def convert(pfile, split, i, out_root):
    s = SPLIT[split]; n = 0
    (out_root / "images" / s).mkdir(parents=True, exist_ok=True); (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(pfile)
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["image", "objects"])
        for j, (im, ob) in enumerate(zip(tb.column("image").to_pylist(), tb.column("objects").to_pylist())):
            stem = f"{split}_{i}_{rg:02d}_{j:04d}"
            try:
                img = Image.open(io.BytesIO(im["bytes"])).convert("RGB")
            except Exception as e:
                print("  corrupt image skipped", stem, e); continue
            w, h = img.size; sc = min(1.0, 960 / max(w, h))
            lines = []
            for x, y, bw, bh in ob["bbox"]:       # COCO xywh, original pixels
                x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                if x2 - x1 < 2 or y2 - y1 < 2: continue
                lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
            if not lines: continue
            if sc < 1: img = img.resize((round(w * sc), round(h * sc)), Image.BILINEAR)
            img.save(out_root / "images" / s / f"{stem}.jpg", quality=90)
            (out_root / "labels" / s / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1
    return n

for split, i in jobs:
    name = f"{split}_{i}"
    shard_zip = CONV / f"{name}.zip"
    done_mark = DST / f".{name}.done"
    if done_mark.exists():
        continue
    if not shard_zip.exists():
        t = time.time()
        src = existing.get(f"{name}.parquet")
        if src and valid_parquet(src):
            how = "from your Drive"
        else:
            src = LOCAL / f"{name}.parquet"
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(f"{BASE}/{split}/{i}.parquet", str(src) + ".part")
                    os.replace(str(src) + ".part", src)
                    if valid_parquet(src): break
                except Exception as e:
                    print("  retry download", name, e); time.sleep(5 * (attempt + 1))
            how = "downloaded"
        tmp = LOCAL / "tmp_shard"; shutil.rmtree(tmp, ignore_errors=True)
        n = convert(src, split, i, tmp)
        if how == "downloaded": pathlib.Path(src).unlink(missing_ok=True)
        part = shard_zip.with_suffix(".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
            for f in tmp.rglob("*"):
                if f.is_file(): z.write(f, f.relative_to(tmp))
        os.replace(part, shard_zip); shutil.rmtree(tmp, ignore_errors=True)
        print(f"{name}: {n} images converted ({how}) in {time.time() - t:.0f}s, saved to Drive", flush=True)
    with zipfile.ZipFile(shard_zip) as z: z.extractall(DST)
    done_mark.write_text("ok")

DATA_YAML.write_text(f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: product\n")
for s_ in ("train", "val", "test"):
    print(s_, len(list((DST / "images" / s_).glob("*.jpg"))), "images")
print("Dataset ready. (Optional: the raw .parquet files on your Drive can now be deleted to free space.)")
''')

code(r'''
# ---- 4. Screening (resumable per candidate) ----
from ultralytics import YOLO
import random
def resume(run):
    """Resume from last.pt; if Ultralytics says the run already finished, that's fine."""
    try:
        YOLO(str(run / "weights" / "last.pt")).train(resume=True)
    except AssertionError as e:
        print("nothing to resume:", e)

def epochs_done(run):
    csv = run / "results.csv"
    return max(0, len(csv.read_text().strip().splitlines()) - 1) if csv.exists() else 0

# fixed screening subset of the train split
sub_list = DST / "screen_train.txt"
imgs = sorted((DST / "images" / "train").glob("*.jpg")); random.Random(0).shuffle(imgs)
sub_list.write_text("\n".join(str(p) for p in imgs[: int(len(imgs) * SCREEN_FRACTION)]) + "\n")
SCREEN_YAML = DST / "screen.yaml"
SCREEN_YAML.write_text(f"path: {DST}\ntrain: {sub_list}\nval: images/val\nnames:\n  0: product\n")

screen = STATE.get("screen", {})
for cand in CANDIDATES:
    if cand in screen:
        print(cand, "already screened:", screen[cand]); continue
    name = "screen_" + cand.replace(".pt", ""); run = RUNS / name
    done = epochs_done(run)
    if done and (run / "weights" / "last.pt").exists() and done < SCREEN_EPOCHS:
        print(f"resuming {cand} from epoch {done}")
        resume(run)
    elif done < SCREEN_EPOCHS:
        YOLO(cand).train(data=str(SCREEN_YAML), epochs=SCREEN_EPOCHS, imgsz=IMGSZ, batch=BATCH, device=DEVICE, workers=2,
                         seed=0, project=str(RUNS), name=name, exist_ok=True, max_det=1000, cos_lr=True,
                         close_mosaic=2, hsv_v=0.4, hsv_s=0.6, plots=False, save_period=1)
    m = YOLO(str(run / "weights" / "best.pt"))
    r = m.val(data=str(DATA_YAML), split="val", imgsz=IMGSZ, batch=BATCH, device=DEVICE, max_det=1000, plots=False, verbose=False)
    screen[cand] = {"mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4),
                    "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                    "gpu_ms_per_img": round(float(r.speed["inference"]), 2),
                    "params_M": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)}
    STATE["screen"] = screen; save_state()
    print(cand, screen[cand], flush=True)
print(json.dumps(screen, indent=2))
''')

code(r'''
# ---- 5. Pick the winner: best mAP50-95; a smaller model wins if within 1.5 points (edge CPU speed) ----
ranked = sorted(screen.items(), key=lambda kv: kv[1]["mAP50_95"], reverse=True)
top = ranked[0][1]["mAP50_95"]
WINNER = min((c for c, v in ranked if top - v["mAP50_95"] <= 0.015), key=lambda c: screen[c]["params_M"])
STATE["winner"] = WINNER; save_state()
print("ranking:", [(c, v["mAP50_95"]) for c, v in ranked]); print("WINNER:", WINNER)
''')

code(r'''
# ---- 6. Full training of the winner (resumes from the last epoch automatically) ----
final_name = "final_" + WINNER.replace(".pt", ""); FINAL = RUNS / final_name
done = epochs_done(FINAL)
if STATE.get("final_done"):
    print("final training already finished")
elif done and (FINAL / "weights" / "last.pt").exists():
    print(f"resuming final training from epoch {done}")
    resume(FINAL)
else:
    # warm start from the screening weights of the winner (already trained 10 epochs on train data only)
    warm = RUNS / ("screen_" + WINNER.replace(".pt", "")) / "weights" / "best.pt"
    START = str(warm) if warm.exists() else WINNER
    print("final training starts from:", START)
    YOLO(START).train(data=str(DATA_YAML), epochs=FINAL_EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                       workers=2, seed=0, project=str(RUNS), name=final_name, exist_ok=True, max_det=1000, cos_lr=True,
                       close_mosaic=10, hsv_v=0.4, hsv_s=0.6, save_period=1)
STATE["final_done"] = True; save_state()
BEST = FINAL / "weights" / "best.pt"
print("best weights:", BEST)
''')

code(r'''
# ---- 7. Held-out TEST evaluation + counting accuracy + robustness (bias) slices ----
import numpy as np, glob, cv2
model = YOLO(str(BEST))
t = model.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE, max_det=1000, plots=True,
              project=str(RUNS), name="test_eval", exist_ok=True)
test_metrics = {"mAP50": round(float(t.box.map50), 4), "mAP50_95": round(float(t.box.map), 4),
                "precision": round(float(t.box.mp), 4), "recall": round(float(t.box.mr), 4)}
print("TEST:", test_metrics)
imgs = sorted(glob.glob(str(DST / "images" / "test" / "*.jpg"))); rows = []
for i in range(0, len(imgs), 32):
    batch = imgs[i:i + 32]
    for p, res in zip(batch, model.predict(batch, imgsz=IMGSZ, conf=0.25, max_det=1000, device=DEVICE, verbose=False)):
        gt = sum(1 for _ in open(p.replace("images", "labels").replace(".jpg", ".txt")))
        rows.append((gt, len(res.boxes), float(cv2.imread(p, cv2.IMREAD_GRAYSCALE).mean())))
gt, pr, br = (np.array(x, dtype=float) for x in zip(*rows))
ape = np.abs(gt - pr) / gt
count = {"images": len(rows), "count_MAE": round(float(np.abs(gt - pr).mean()), 2),
         "count_accuracy_%": round(float(100 * (1 - ape.mean())), 2), "within_10pct_%": round(float(100 * (ape <= 0.1).mean()), 1)}
slices = {}
for label, m in {"dark (<90)": br < 90, "normal (90-160)": (br >= 90) & (br < 160), "bright (>=160)": br >= 160,
                 "sparse (<100 items)": gt < 100, "medium (100-199)": (gt >= 100) & (gt < 200), "dense (>=200)": gt >= 200}.items():
    if m.sum(): slices[label] = {"images": int(m.sum()), "count_accuracy_%": round(float(100 * (1 - ape[m].mean())), 2)}
print("COUNT:", count); print("SLICES:", json.dumps(slices, indent=1))
''')

code(r'''
# ---- 8. Export for edge + save everything to Drive ----
onnx_path = YOLO(str(BEST)).export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=False)
summary = {"screening_val": screen, "winner": WINNER, "test": test_metrics, "count": count, "slices": slices,
           "final_epochs_run": epochs_done(FINAL), "imgsz": IMGSZ}
shutil.copy(BEST, PERSIST / "shelf_detector_best.pt"); shutil.copy(onnx_path, PERSIST / "shelf_detector_best.onnx")
(PERSIST / "colab_results.json").write_text(json.dumps(summary, indent=2))
print("Saved to Drive:", PERSIST); print(json.dumps(summary, indent=2))
''')

code(r'''
# ---- 9. Weights inside this notebook's output (press Ctrl+S afterwards) ----
# On the laptop:  python scripts/extract_weights_from_notebook.py
import base64, hashlib
raw = BEST.read_bytes()
print("RESULTS_JSON::" + json.dumps(summary))
print("WEIGHTS_SHA256::" + hashlib.sha256(raw).hexdigest())
b64 = base64.b64encode(raw).decode(); CH = 900_000
for k in range(0, len(b64), CH):
    print(f"WEIGHTS_CHUNK::{k // CH}::{b64[k:k + CH]}")
print("WEIGHTS_END::", len(raw), "bytes  -> now press Ctrl+S")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell{i:02d}"
(ROOT / "inventory.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", ROOT / "inventory.ipynb", len(cells), "cells")
