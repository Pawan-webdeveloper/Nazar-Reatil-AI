# RetailSense AI - Edge Retail Intelligence Platform

Privacy-first, offline-capable retail analytics that runs on a store's own hardware.
It turns camera streams into shopper analytics, inventory visibility and checkout
queue intelligence, with alerts, reports and a multi-store dashboard.

Built for Indian retail: kirana / neighbourhood stores, supermarkets, pharmacies and
hypermarkets, including Tier-2 / Tier-3 locations with unreliable internet.

## Results at a glance

All numbers come from held-out data. Thresholds were tuned on validation splits only.
Raw result files are in `outputs/`.

### Output



### Shelf product detector (SKU-110K, trained on Colab T4)

| Model (screening, identical budget, val) | mAP50 | mAP50-95 | GPU ms/img | Params |
|---|---|---|---|---|
| YOLOv8n | 0.868 | 0.508 | 3.5 | 3.0 M |
| YOLO11n | 0.863 | 0.503 | 3.3 | 2.6 M |
| **YOLO11s (winner)** | **0.897** | **0.545** | 7.8 | 9.4 M |

Winner trained 50 epochs on the full train split, then evaluated once on the 2,936-image test split:

| Test metric | Value |
|---|---|
| mAP50 | **0.930** |
| mAP50-95 | **0.583** |
| Precision / Recall | 0.914 / 0.869 |
| Product-count accuracy per image | 89.1 % (conf 0.25), 89.0 % on local test shard at tuned conf 0.30 |

Bias / robustness slices (count accuracy): normal light 89 %, dark 87 %, medium density 91 %,
dense 91 %. Sparse shelves were weak (42 % at conf 0.25). Inspection showed most of that gap is
incomplete or case-level labelling in SKU-110K, plus a real weakness on peg-hook merchandise.
Mitigations: counting threshold tuned on validation (sparse 55 % -> 63 %), and a per-camera
"Mark as full" baseline so stock % is measured against the same camera's own full shelf.

### Out-of-stock (empty slot) detection

Evaluated by removing 3-6 products from real shelf photos (inpainting or dark empty-shelf fill),
100 validation images for tuning and 100 test images for reporting.

| Method | Alert at the right place | Precision | False alerts / image |
|---|---|---|---|
| Gap analysis (before calibration) | 32 % strict recall | >= 21 % (lower bound) | 1.6 |
| **Baseline comparison (after "Mark as full")** | **83 %** (86 % on dark empty shelf) | **65 %** | **0.75** (0.51 with no change) |

The baseline test includes brightness +-20 %, sensor noise, +-3 px camera shake and JPEG
re-compression between the two snapshots.

### Person detector (COCO val2017, 600 images with people + 150 without)

| Model | AP50 | AP50-95 | Recall | Small | Crowd | Dark | CPU ms |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.730 | 0.501 | 0.59 | 0.24 | 0.52 | 0.62 | 86 |
| **YOLO11n (deployed)** | 0.735 | 0.508 | 0.59 | 0.24 | 0.50 | 0.61 | 92 |
| YOLO11s | **0.789** | **0.570** | **0.65** | **0.34** | **0.58** | **0.65** | 176 |

YOLO11n is deployed so three cameras fit on one CPU box. Switch to YOLO11s for boxes with 1-2
cameras. False alarms on empty images: 0.7 %. On the entrance demo clip all 5 line crossings were
counted correctly (verified from track trajectories).

### Footfall counting against human ground truth (MOT17 benchmark)

Three static-camera MOT17 sequences (street, plaza, mall storefront; up to ~35 people in view) have
every pedestrian annotated in every frame. Each sequence gets a horizontal and a vertical counting
line, giving 59 true crossings, 46 of them made while the person was visible. Crossing events are
matched by direction, time (+-1 s) and position.

| Setup (CPU, ByteTrack) | Count accuracy | Event precision | Recall, visible people | Occupancy accuracy (crowded plaza) |
|---|---|---|---|---|
| YOLO11n, 640 px, 6 FPS | 47.5 % | 0.78 | 0.54 | 42 % |
| YOLO11n, 960 px, 6 FPS | 59.3 % | 0.76 | 0.65 | 67 % |
| **YOLO11n, 960 px, 10 FPS** | **64.4 %** | **0.78** | **0.70** | - |
| YOLO11s, 960 px, 6 FPS | 57.6 % | 0.79 | 0.63 | 63 % |

What this showed and what changed:

- The first counter (plain segment crossing) produced in/out pairs for people standing on the line and
  missed crossings when the tracker skipped frames. It was replaced by a hysteresis counter with a dead
  band and remembered sides (`retail_ai/analytics/footfall.py`).
- Most remaining misses happen when people cross while hidden behind others, or when the tracker
  changes their id after an occlusion. Track stitching and looser tracker matching were tested and
  made results worse in crowds, so they are off by default.
- Recommendation used in `configs/store_config.yaml`: run the entrance camera at 960 px and the highest
  FPS the box allows, mounted over a doorway where people pass one or two at a time. Dense street
  scenes like MOT17 are harder than a shop door.

### Queue forecaster (15 minutes ahead)

| Model (test, last 2 weeks) | MAE | R2 | Congestion alert F1 |
|---|---|---|---|
| Persistence baseline | 1.109 | 0.974 | 0.845 |
| Ridge | 1.236 | 0.980 | 0.865 |
| Random forest | 0.819 | 0.991 | 0.887 |
| **LightGBM (deployed)** | **0.814** | **0.991** | **0.885** |

Unseen store format (trained without it), LightGBM vs persistence MAE: hypermarket 1.09 vs 1.22,
neighbourhood 0.90 vs 0.97, supermarket 1.04 vs 1.14. Entrance footfall 5-30 minutes earlier is
among the strongest features. **Caveat:** trained on simulated traffic because no public
minute-level Indian store queue data exists. Retrain on the store's own logs with `--from-db`.

### Edge performance (Intel i5-1334U laptop, no GPU)

| Item | Value |
|---|---|
| Three video cameras at once | 3.5-5.7 FPS each (target 3) |
| Shelf snapshot | ~190 ms per image |
| Data leaving the store | ~1.2 MB per camera per day vs ~12.6 GB for cloud video (-99.99 %) |
| Offline operation | Full; SQLite store-and-forward sync when the network returns |

PyTorch was faster than ONNX Runtime and OpenVINO FP32 on this CPU; exports are kept for other hardware.


## What it does

| Module | Capabilities |
|---|---|
| **Shopper analytics** | Entry / exit counting, live occupancy, footfall by hour / day / weekday, zone dwell time and engagement, movement heatmaps |
| **Inventory monitoring** | Product detection on shelf images, empty-slot (void) detection, stock level per section, low / out-of-stock alerts, planogram compliance, ERP-ready replenishment orders |
| **Queue intelligence** | Live queue length per counter, measured waiting and service time, 15-minute congestion forecast, "open N counters" recommendation (Erlang-C) |
| **Edge AI** | All models run locally on CPU (PyTorch or ONNX). No video leaves the store. Works with the internet down |
| **Privacy** | No face recognition, no age / gender inference, rotating anonymous ids, pixelated previews, frames never stored, k-anonymity on reports |
| **Dashboard** | Real-time alerts, KPIs (footfall, conversion, queue, shelf availability, bills per counter-hour), daily and weekly reports, live camera view |
| **Scale-out** | Same software at HQ receives store-and-forward syncs from many stores; POS CSV / REST ingestion; ERP replenishment JSON |

## Output previews

<p align="center">
  <img src="outputs/dashboard_overview.png" alt="RetailSense dashboard overview" width="900" />
</p>

<p align="center">
  <img src="outputs/live_cameras.png" alt="RetailSense live camera dashboard" width="900" />
</p>

## Architecture

```mermaid
flowchart LR
  subgraph Store["Store edge box (CPU, offline-capable)"]
    C1[Entrance cam] --> P
    C2[Floor cam] --> P
    C3[Checkout cam] --> P
    C4[Shelf cam] --> P
    P["Edge pipeline<br/>YOLO + ByteTrack<br/>shelf detector"] --> A["Analytics<br/>footfall, dwell, heatmap,<br/>queue, voids, forecast"]
    A --> DB[(SQLite WAL<br/>aggregates only)]
    A --> AL[Alert engine]
    POS[POS / billing] --> DB
    DB --> API[FastAPI + dashboard]
    DB --> S[Sync agent]
  end
  S -- "when online: a few KB/min" --> HQ["HQ server<br/>same app, multi-store view"]
  AL -- webhook --> Staff[Staff phone / Slack / ERP]
```

Only anonymous aggregates are stored and synced. A camera never sends video out of the store.

## Project layout

```
retail_ai/
  analytics/    footfall.py, dwell.py, heatmap.py, queue_monitor.py, shelf.py, forecasting.py
  edge/         detector.py (YOLO + ByteTrack), pipeline.py (camera workers, EdgeNode)
  storage/      db.py (SQLite, offline-first, sync flags)
  integrations/ pos.py (POS CSV / REST, ERP replenishment), history.py (retraining data)
  api/          server.py (REST API, HQ ingest, dashboard hosting)
  dashboard/    static/ (offline HTML/JS dashboard, no CDN)
  alerts.py, sync.py, privacy.py, reports.py, simulation.py, demo_data.py, config.py
training/       prepare_sku110k.py, train_shelf_detector.py, evaluate_shelf.py,
                evaluate_person_detectors.py, train_queue_forecaster.py
scripts/        build_colab_notebook.py, extract_weights_from_notebook.py, benchmark_edge.py
configs/        store_config.yaml (cameras, zones, counters, thresholds)
models/         trained weights (person_detector.pt, shelf_detector.pt, queue_forecaster.joblib)
outputs/        all evaluation results, plots and logs
tests/          33 unit / integration tests
inventory.ipynb Colab notebook for the VS Code Colab extension
RetailSense_inventory_colab.ipynb  same notebook for Colab in the browser (used for the final run)
docs/RESEARCH.md papers, datasets, bias analysis and design decisions
docs/DEPLOYMENT.md step-by-step deployment: one store, a chain, privacy (DPDP Act)
```

## Setup (Windows, CPU only)

```bash
python -m venv C:\Users\itanm\retail_ai_env\venv
```

```bash
C:\Users\itanm\retail_ai_env\venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
C:\Users\itanm\retail_ai_env\venv\Scripts\python -m pip install -r requirements.txt
```

The virtual environment and datasets live outside OneDrive on purpose, so OneDrive
does not try to sync gigabytes of images and packages.

## Run

Explore the dashboard with simulated history (marked "(simulated)" everywhere):

```bash
python run.py demo-data --days 28
```

```bash
python run.py dashboard
```

Then open http://localhost:8000.

Run the real edge pipeline on the configured cameras, with the dashboard and live view:

```bash
python run.py edge
```

Process the demo videos as fast as possible and save privacy-blurred annotated videos:

```bash
python run.py edge --no-api --fast --video-out outputs/demo_videos
```

Run the tests:

```bash
python -m pytest -q
```

## Connecting real cameras

Edit `configs/store_config.yaml`. For each camera set `source` to an RTSP URL
(for example `rtsp://user:pass@192.168.1.20:554/stream1`), a webcam index such as `"0"`,
or a video file. Then draw the count line, zones, queue and service polygons in pixel
coordinates of the processed frame. Dropped RTSP streams reconnect automatically.

## Shelf calibration (recommended for every shelf camera)

After a shelf section is fully stocked, open the dashboard, go to **Inventory** and press
**Mark as full** for that section. The latest snapshot becomes the 100 % reference: stock % and
empty slots are then measured against this camera's own full shelf. This removes counting bias
(cases vs units, peg hooks) and made out-of-stock alerts about 4x more precise in testing.

## Retraining

| Model | Command |
|---|---|
| Shelf detector, GPU (recommended) | Upload `RetailSense_inventory_colab.ipynb` to Colab, Runtime -> T4 GPU, Run all. Progress lives in Drive `MyDrive/retail_ai`, so Run all again resumes after a disconnect. Download `runs/final_yolo11s/weights/best.pt` to `models/shelf_detector.pt` |
| Shelf detector, CPU | `python training/train_shelf_detector.py final --data <sku110k.yaml> --model models/yolo11n.pt --epochs 20` |
| Person detector comparison | `python training/evaluate_person_detectors.py --parquet <coco val parquet>` |
| Queue forecaster on real store data | `python training/train_queue_forecaster.py --from-db data/retail_edge.db` |

## Deployment

The full step-by-step guide (hardware, camera placement, calibration, HQ, privacy) is in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). In short:


- **Single store:** `docker compose up -d edge` on any x86 mini-PC (i5 class, 8 GB RAM).
- **Chain / HQ:** run `docker compose --profile hq up -d hq` centrally and set `sync.central_url`
  in every store's config. Stores keep working offline and upload when connectivity returns.
- **Faster hardware:** export to ONNX / OpenVINO (`scripts/benchmark_edge.py`) or run on a Jetson
  with a CUDA build of PyTorch. Set `models.device` accordingly.
- **Security:** set `RETAIL_API_KEY` to protect write endpoints and `RETAIL_SYNC_KEY` for HQ ingestion.
# Nazar-Reatil-AI
