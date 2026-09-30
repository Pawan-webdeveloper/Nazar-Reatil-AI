# RetailSense AI - Edge Retail Intelligence Platform

> Privacy-first, offline-capable retail analytics that runs on a store's own hardware.
> Turns camera streams into shopper analytics, inventory visibility, and checkout queue
> intelligence — with alerts, reports, and a multi-store dashboard.

Built for Indian retail: kirana / neighbourhood stores, supermarkets, pharmacies, and
hypermarkets — including Tier-2 / Tier-3 locations with unreliable internet.

---

## Dashboard

### Overview

![RetailSense AI — Dashboard Overview](presentation/assets/ui_overview.png)

The home screen shows every KPI a store manager needs at a glance: footfall (entries/exits),
POS bills, conversion rate, average queue length, on-shelf availability, and open alerts.
Below the KPI cards are hourly footfall, daily footfall trends, and a weekly traffic heatmap
so managers can see exactly when the store is busiest.

### Live Cameras

![RetailSense AI — Live Camera View](presentation/assets/ui_live.png)

Real-time annotated feeds from every camera in the store. The entrance camera counts people
crossing the virtual line, the floor camera tracks zone engagement, the checkout camera
measures queue length, and the shelf camera monitors stock. All processing happens on the
edge device — no video leaves the store. People are pixelated in previews for privacy.

### Inventory & Shelf Monitoring

![RetailSense AI — Inventory Dashboard](presentation/assets/ui_inventory.png)

Every shelf section is monitored for stock levels. The system detects empty slots, flags
low-stock and out-of-stock sections, and measures planogram compliance. After restocking,
staff press **Mark as full** to calibrate — this removes counting bias (cases vs units,
peg hooks) and makes out-of-stock alerts ~4x more precise.

### Shopper Analytics

![RetailSense AI — Shopper Analytics](presentation/assets/ui_shoppers.png)

Zone-level engagement metrics: visits per zone, average dwell time, and the percentage of
visitors who stayed long enough to engage (10+ seconds). Zones with fewer than 3 visits are
hidden for privacy (k-anonymity).

---

## What Makes RetailSense Different

| | Cloud video analytics | RetailSense AI |
|---|---|---|
| **Where models run** | Cloud servers | On the store's own CPU box |
| **Internet needed** | Always | Fully offline; syncs when online |
| **Video leaves the store?** | Yes (full streams) | No — only anonymous aggregates (~1.2 MB/day vs ~12.6 GB) |
| **Privacy** | Faces stored, demographics inferred | No face recognition, no age/gender, rotating anonymous IDs, pixelated previews |
| **Latency** | Seconds to minutes (round trip) | Real-time, on-device |
| **Cost** | Per-camera cloud subscription | One-time hardware, no recurring cloud fees |
| **Tier-2/3 ready** | No — needs stable broadband | Yes — works with intermittent connectivity |

---

## What It Does

| Module | Capabilities |
|---|---|
| **Shopper analytics** | Entry/exit counting, live occupancy, footfall by hour/day/weekday, zone dwell time and engagement, movement heatmaps |
| **Inventory monitoring** | Product detection on shelf images, empty-slot (void) detection, stock level per section, low/out-of-stock alerts, planogram compliance, ERP-ready replenishment orders |
| **Queue intelligence** | Live queue length per counter, measured waiting and service time, 15-minute congestion forecast, "open N counters" recommendation (Erlang-C) |
| **Edge AI** | All models run locally on CPU (PyTorch or ONNX). No video leaves the store. Works with the internet down |
| **Privacy** | No face recognition, no age/gender inference, rotating anonymous IDs, pixelated previews, frames never stored, k-anonymity on reports |
| **Dashboard** | Real-time alerts, KPIs (footfall, conversion, queue, shelf availability, bills per counter-hour), daily and weekly reports, live camera view |
| **Scale-out** | Same software at HQ receives store-and-forward syncs from many stores; POS CSV/REST ingestion; ERP replenishment JSON |

---

## How It Works

### Architecture

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

### The Pipeline

1. **Detection** — YOLO11n (person) and YOLO11s (shelf products) run on every frame from every
   camera. ByteTrack assigns persistent IDs to people across frames.
2. **Counting** — A hysteresis counter with a dead band and remembered sides counts people
   crossing the entrance line. This avoids jitter double-counts and still catches crossings
   when the tracker skips frames.
3. **Tracking & zones** — Each person's trajectory is checked against zone polygons (aisles,
   promo displays, walkways). Dwell time and engagement are computed per zone.
4. **Queue** — People in the queue polygon are counted; people in the service polygon are
   timed. A LightGBM model forecasts queue length 15 minutes ahead.
5. **Shelf** — Every 60 seconds a shelf snapshot is analysed: products are detected, empty
   slots are found via baseline comparison against the last "Mark as full" reference image.
6. **Storage** — Only aggregates (counts, averages, alerts) are written to a local SQLite
   database. Raw frames are never persisted.
7. **Sync** — A background agent uploads unsynced rows to HQ when the network is available,
   with exponential backoff on failure. Nothing is lost.
8. **Dashboard** — A FastAPI server serves an offline HTML/JS dashboard (no CDN) with live
   camera previews, KPIs, charts, and alerts.

---

## Results

All numbers come from held-out data. Thresholds were tuned on validation splits only.
Raw result files are in `outputs/`.

### Shelf Product Detector (SKU-110K, trained on Colab T4)

| Model | mAP50 | mAP50-95 | GPU ms/img | Params |
|---|---|---|---|---|
| YOLOv8n | 0.868 | 0.508 | 3.5 | 3.0 M |
| YOLO11n | 0.863 | 0.503 | 3.3 | 2.6 M |
| **YOLO11s (winner)** | **0.897** | **0.545** | 7.8 | 9.4 M |

Winner evaluated on the 2,936-image test split:

| Test metric | Value |
|---|---|
| mAP50 | **0.930** |
| mAP50-95 | **0.583** |
| Precision / Recall | 0.914 / 0.869 |
| Product-count accuracy | 89.1 % (conf 0.25) |

### Out-of-Stock Detection

Evaluated by removing 3-6 products from real shelf photos, 100 validation + 100 test images.

| Method | Recall | Precision | False alerts/image |
|---|---|---|---|
| Gap analysis (before calibration) | 32 % | >= 21 % | 1.6 |
| **Baseline comparison (after "Mark as full")** | **83 %** | **65 %** | **0.75** |

### Person Detector (COCO val2017)

| Model | AP50 | AP50-95 | Recall | CPU ms |
|---|---|---|---|---|
| YOLOv8n | 0.730 | 0.501 | 0.59 | 86 |
| **YOLO11n (deployed)** | 0.735 | 0.508 | 0.59 | 92 |
| YOLO11s | **0.789** | **0.570** | **0.65** | 176 |

YOLO11n is deployed so three cameras fit on one CPU box.

### Footfall Counting (MOT17 benchmark)

| Setup (CPU, ByteTrack) | Count accuracy | Event precision | Recall |
|---|---|---|---|
| YOLO11n, 640 px, 6 FPS | 47.5 % | 0.78 | 0.54 |
| YOLO11n, 960 px, 6 FPS | 59.3 % | 0.76 | 0.65 |
| **YOLO11n, 960 px, 10 FPS** | **64.4 %** | **0.78** | **0.70** |

### Queue Forecaster (15 min ahead)

| Model | MAE | R2 | Congestion alert F1 |
|---|---|---|---|
| Persistence baseline | 1.109 | 0.974 | 0.845 |
| **LightGBM (deployed)** | **0.814** | **0.991** | **0.885** |

### Edge Performance (Intel i5-1334U, no GPU)

| Item | Value |
|---|---|
| Three cameras at once | 3.5-5.7 FPS each (target 3) |
| Shelf snapshot | ~190 ms per image |
| Data leaving the store | ~1.2 MB/camera/day vs ~12.6 GB cloud video (**-99.99 %**) |
| Offline operation | Full; SQLite store-and-forward sync |

---

## Visualizations

### Queue Forecast (15 min ahead)

![Queue forecast vs actual on busiest test day](presentation/assets/forecast_clean.png)

The red line (our LightGBM forecast made 15 minutes earlier) tracks the blue line
(actual people waiting) closely through both morning and evening rush hours.

### Movement Heatmap

![Shopper heatmap overlaid on aisle](presentation/assets/heatmap_overlay_aisle.jpg)

Aggregated anonymous trajectories show where shoppers spend time. Hot zones indicate
high engagement; cold zones may need better product placement.

### Shelf Void Detection

![Shelf with detected products and void regions](presentation/assets/shelf_void_baseline.jpg)

Green boxes mark detected products. Red boxes mark empty slots (voids) detected by
comparing the current snapshot against the calibrated "full shelf" baseline.

---

## Project Layout

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
presentation/assets/  UI screenshots and demo images
docs/RESEARCH.md      papers, datasets, bias analysis and design decisions
docs/DEPLOYMENT.md    step-by-step deployment: one store, a chain, privacy (DPDP Act)
```

---

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

Process demo videos as fast as possible and save privacy-blurred annotated videos:

```bash
python run.py edge --no-api --fast --video-out outputs/demo_videos
```

Run the tests:

```bash
python -m pytest -q
```

## Connecting Real Cameras

Edit `configs/store_config.yaml`. For each camera set `source` to an RTSP URL
(for example `rtsp://user:pass@192.168.1.20:554/stream1`), a webcam index such as `"0"`,
or a video file. Then draw the count line, zones, queue and service polygons in pixel
coordinates of the processed frame. Dropped RTSP streams reconnect automatically.

## Shelf Calibration (Recommended for Every Shelf Camera)

After a shelf section is fully stocked, open the dashboard, go to **Inventory** and press
**Mark as full** for that section. The latest snapshot becomes the 100 % reference: stock %
and empty slots are then measured against this camera's own full shelf. This removes
counting bias (cases vs units, peg hooks) and made out-of-stock alerts about 4x more
precise in testing.

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

## License

Proprietary. See LICENSE file for details.
