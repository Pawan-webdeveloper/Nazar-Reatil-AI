# Research notes and design decisions

This file records what was studied before building RetailSense AI and why each
technical choice was made.

## 1. Problem decomposition

| Requirement (problem statement) | Technique chosen | Where in code |
|---|---|---|
| Count customers entering / exiting | Person detection + multi-object tracking + virtual line crossing | `retail_ai/analytics/footfall.py` |
| Footfall by time / day / zone | Minute aggregates in SQLite, hourly and weekday x hour views | `retail_ai/edge/pipeline.py`, `retail_ai/reports.py` |
| Dwell time near products / promos | Zone polygons, per-track visit start/end, engagement at >= 10 s | `retail_ai/analytics/dwell.py` |
| Heatmaps | Foot-point accumulation on a grid, smoothed; no images stored | `retail_ai/analytics/heatmap.py` |
| Low / out-of-stock detection | Dense product detector (SKU-110K) + shelf-row clustering + gap (void) analysis | `retail_ai/analytics/shelf.py` |
| Planogram compliance | Expected rows / facings per section; optional appearance check per SKU group | `retail_ai/analytics/shelf.py` |
| Queue length, waiting and service time | Queue and service polygons per counter, per-track timing | `retail_ai/analytics/queue_monitor.py` |
| Predict congestion before it happens | LightGBM forecaster (15 min ahead) using entrance footfall as a leading signal | `retail_ai/analytics/forecasting.py` |
| Recommend opening counters | Erlang-C (M/M/c) queueing model on forecast arrival rate | `retail_ai/analytics/forecasting.py` |
| Edge processing, offline operation | All inference local, SQLite WAL store, store-and-forward sync | `retail_ai/edge`, `retail_ai/storage`, `retail_ai/sync.py` |
| Privacy | No face / demographic models, rotating salted ids, pixelated previews, k-anonymity | `retail_ai/privacy.py` |
| Dashboard, alerts, reports | FastAPI + offline dashboard, cooldown alerts, daily / weekly reports | `retail_ai/api`, `retail_ai/alerts.py` |
| POS / ERP integration, multi-store | POS CSV and REST ingest, replenishment orders, HQ ingest endpoint | `retail_ai/integrations`, `/api/ingest` |

## 2. Literature and prior work reviewed

**Shelf / out-of-stock detection**
- Goldman et al., *Precise Detection in Densely Packed Scenes*, CVPR 2019 - introduces
  SKU-110K (11,762 shelf images, ~1.7 M boxes). Densely packed, similar-looking objects
  are the core difficulty; we train on this dataset.
- *Enhanced Out-of-Stock Detection in Retail Shelf Images Based on Deep Learning*, Sensors 2024
  (PMC10819825) - separates "fully empty" and "frontal" OOS; reports 86.3 % / 83.7 % AP.
  Motivated our void analysis on top of product detection instead of a separate empty-shelf class,
  because public empty-shelf labels are scarce and product detection transfers across stores.
- *Revolutionizing Retail Stores with Computer Vision and Edge AI: A Novel Shelf Management System*,
  IEEE 2023 - semi-supervised YOLO on the edge; confirms YOLO-n/s class models are suitable for
  edge shelf monitoring.
- *LSR-YOLO: a lightweight and fast model for retail products detection* (PMC12543153, 2025) -
  channel-pruned YOLOv8 for supermarket edge devices; supports choosing nano-sized models.
- *Real-Time Retail Shelf-Stock Detection with YOLOv7* (2025) - high mAP on single-store data;
  such numbers do not transfer to multi-store benchmarks like SKU-110K, which is why we evaluate
  on the SKU-110K test split rather than on a single store.

**People analytics**
- ByteTrack (Zhang et al., ECCV 2022) - associates low-confidence detections too, which keeps
  identities through partial occlusion in crowded aisles. Used through Ultralytics.
- Open-source retail pipelines reviewed on GitHub (YOLOv8 + ByteTrack people counting, dwell
  and heatmaps, e.g. *Retail-Crowd-Analytics*, *cctv-store-analytics*, *aisle-iq*). They validated
  the general architecture but typically lack: offline-first storage, privacy controls,
  forecasting, POS integration and evaluation. Those gaps are what this project adds.

**Queue intelligence**
- *Queue Time Estimation in Checkout Counters Using Computer Vision and Deep Neural Network* -
  per-counter timing from video; we measure waiting and service time from tracks in the same way.
- *Airport Terminal Passenger Queue Forecasting* (arXiv 2606.07622) and *Q-Net* (arXiv 2509.24725) -
  learn from recent queue history and upstream flow; we use entrance footfall 10-45 minutes
  earlier as the upstream signal for the checkout.
- Classical Erlang-C (M/M/c) is used for staffing because it is explainable to store managers
  and needs only arrival rate, service time and counters.

## 3. Datasets

| Dataset | Used for | Notes |
|---|---|---|
| SKU-110K (HF mirror `benjamintli/sku110k`) | Shelf product detector train / val / test | Local CPU run uses 4 of 19 train shards; the Colab notebook uses all of them |
| COCO val2017 (HF `detection-datasets/coco`) | Person detector comparison and bias slices | Validation images were not used to train the official weights |
| Intel IoT DevKit sample videos | End-to-end demo of entrance, floor and queue cameras | Public sample footage |
| Simulated store traffic (`retail_ai/simulation.py`) | Queue forecaster development | No public minute-level Indian retail queue dataset exists; retrain on real logs with `--from-db` |

## 4. Bias and robustness

- **No demographic inference.** Age / gender / face models are deliberately absent. This removes the
  most common source of demographic bias in retail analytics and avoids collecting PII.
- **Person detector (measured, YOLO11n recall).** Large people 0.85, medium 0.60, small 0.24;
  crowded scenes 0.50 vs sparse 0.79; dark images 0.61 vs normal 0.58 (`outputs/person_detector`).
  Small and crowded people are the weak spots. Mitigations: mount cameras so shoppers appear at
  least ~40 px tall, use YOLO11s (small 0.34, crowd 0.58) where compute allows, and let ByteTrack
  keep identities through short misses. New tracks are confirmed after 1-2 frames, so previews blur
  the whole frame, not only tracked people.
- **Footfall counting (measured on MOT17 with human ground truth).** 64 % count accuracy at 960 px /
  10 FPS in dense street scenes, event precision 0.78, recall 0.70 for people visible while crossing.
  Crossings made while fully occluded are the main loss. A first counter produced jitter pairs and
  missed crossings over skipped frames; the hysteresis counter fixed both. Tracker stitching and
  looser matching were measured and rejected because they swap identities in crowds.
- **Shelf detector (measured, count accuracy on SKU-110K test).** Normal light 89 %, dark 87 %,
  medium density 91 %, dense 91 %, sparse 42 % at conf 0.25. The worst sparse images have case-level
  or missing labels (the model counts individual units), plus one genuine failure mode: peg-hook
  merchandise such as hanging toothbrushes. The counting threshold was re-tuned on validation
  (0.30; sparse 55 % -> 63 % on the local test shard). A per-camera "fully stocked" baseline makes
  stock % relative to each camera, which cancels systematic counting bias.
- **Out-of-stock detection.** Pure gap analysis on handheld, angled photos first produced 12 false
  alerts per image. Neighbour-row extents, overlap rejection, an empty-shelf texture test and a
  thin-row filter cut that to 1.6. For fixed store cameras, baseline comparison reaches 83 % alerts
  at the right place with 0.75 false alerts per image, even with brightness, noise, shake and
  compression changes between snapshots.
- **Forecaster.** A first version trained on absolute queue length failed on festival days (tree
  models cannot extrapolate above the training range) and on unseen store formats. Learning the
  per-counter change in queue length fixed both: festival-day MAE is 8 % of the mean queue, and the
  model beats persistence on every held-out store format.

## 5. Why these models

- **Shelf: YOLO11s.** Best screening mAP50-95 (0.545 vs 0.508 and 0.503), more than 1.5 points ahead,
  so the smaller-model rule did not apply. Full training gave test mAP50 0.930 / mAP50-95 0.583.
- **People: YOLO11n deployed, YOLO11s optional.** YOLO11s is more accurate on every slice but needs
  176 ms vs 92 ms per frame on the target CPU. Three cameras per box need the faster model.
- **Queue: LightGBM.** Best validation error among persistence, seasonal naive, moving average,
  ridge and random forest; fast on CPU; 3 MB model file.
- **Runtime: PyTorch on CPU.** Faster than ONNX Runtime and OpenVINO FP32 on the i5-1334U test machine.
