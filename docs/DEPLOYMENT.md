# Deployment guide

RetailSense AI has two parts:

- **Edge node (in every store).** Reads the cameras, runs all AI on a small local computer, stores
  results in SQLite and serves the store dashboard. It keeps working without internet.
- **HQ server (optional, one for the whole chain).** The same software in `dashboard` mode. Stores push
  a few KB of anonymous numbers per minute to it, and managers see all stores in one place.

Video never leaves the store. Only aggregated numbers are synced.

## Stage 0: demo on your laptop

```bash
python run.py demo-data --days 28
```

```bash
python run.py dashboard
```

Open http://localhost:8000. For the camera pipeline on the sample videos:

```bash
python run.py edge --config configs/video_demo.yaml --port 8001 --loop
```

## Stage 1: pilot in one store

### 1. Hardware

| Item | Suggested | Approx. cost (INR) |
|---|---|---|
| Edge computer | Intel i5 / Ryzen 5 mini PC, 16 GB RAM, 256 GB SSD | 35,000 - 50,000 |
| or | NVIDIA Jetson Orin Nano (more cameras per box) | 45,000 - 60,000 |
| Cameras | Existing CCTV works if the NVR / camera exposes RTSP (Hikvision, CP Plus, Dahua do) | 0 - 3,000 each |
| Network | PoE switch, LAN cable to the edge PC | 3,000 - 6,000 |
| Power | Small UPS so power cuts do not corrupt data | 3,000 - 5,000 |

One i5 box handles three people-cameras at about 3-4 FPS each plus shelf snapshots (measured).
Use one box per 3 cameras, or YOLO11n everywhere and a Jetson for bigger stores.

### 2. Camera placement

| Camera | Placement | Why |
|---|---|---|
| Entrance | Above the door, tilted 30-45 degrees, whole doorway visible | Top-down views are harder for the person model; people should be at least ~40 px tall |
| Floor / promo | Corner, looking along the aisle or at the display | Zones and heatmap need the floor area in view |
| Checkout | Behind or beside the counter, looking at the queue | Queue and service areas must be separate polygons |
| Shelf | Facing the shelf straight on, 1.5-3 m away, fixed mount | Out-of-stock detection compares with a "full shelf" snapshot from the same view |

Find each camera's RTSP address in the NVR manual, for example
`rtsp://user:password@192.168.1.64:554/Streaming/Channels/101` for Hikvision.

### 3. Install

Option A, Docker (recommended on Linux):

```bash
docker compose up -d edge
```

Option B, Python on Windows:

```bash
python -m venv venv
```

```bash
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
venv\Scripts\python -m pip install -r requirements.txt
```

Copy the `models/` folder (shelf_detector.pt, person_detector.pt, queue_forecaster.joblib).
To start automatically after a reboot, register `python run.py edge` as a Windows service
(for example with NSSM) or a Task Scheduler task "At startup".

### 4. Configure the store

Edit `configs/store_config.yaml`:

1. `store.store_id`, `store.name`, opening hours.
2. For each camera: `source` (RTSP URL), `role` (entrance / floor / queue / shelf).
3. Draw the geometry in pixel coordinates of the processed frame: the entrance `count_line` and
   `entry_direction`, floor `zones`, checkout `queue_polygon` and `service_polygon`, shelf sections.
   Run `python scripts/zone_helper.py`: it saves one frame per camera with a pixel grid and the
   current lines / zones drawn on it (in `outputs/zone_helper/`), so you can read the coordinates.
4. `queue.counters_total`, `queue.target_wait_minutes`, alert thresholds.

### 5. Calibrate and verify on day one

- Stand at the entrance and count 20 people in and out by hand. Compare with the dashboard. If "in"
  and "out" are swapped, change `entry_direction`.
- When every shelf section is fully stocked, press **Mark as full** in the Inventory tab.
- Check the Live cameras tab: people must be blurred and zones must be where you expect.

### 6. Use it

- Staff open `http://<edge-pc-ip>:8000` on any phone or PC on the store Wi-Fi.
- Set `RETAIL_API_KEY` so only staff apps can acknowledge alerts or push POS bills.
- Alerts to phones: set `alerts.webhook_url` to a WhatsApp Business API, Telegram bot or Slack webhook.
- POS: let the billing software drop CSV files into `data/pos_exports`, or POST bills to
  `/api/pos/transactions`. This unlocks conversion rate and bills per counter-hour.

## Stage 2: many stores (chain)

1. Rent a small cloud VM in the Mumbai region (AWS Lightsail, Azure, DigitalOcean; 2 vCPU / 4 GB is
   enough for dozens of stores because only numbers arrive).
2. Run the HQ profile:

```bash
docker compose --profile hq up -d hq
```

3. Put HTTPS in front of it (Caddy or Nginx with Let's Encrypt) and set `RETAIL_INGEST_KEY`.
4. In every store config set `sync.central_url: https://hq.yourcompany.in` and the same key as
   `RETAIL_SYNC_KEY`. Stores upload every minute; during an internet outage they keep data locally
   and send it when the connection returns.
5. For remote support, install Tailscale (or any VPN) on each edge box instead of opening ports.

## Stage 3: keep it accurate

| When | What |
|---|---|
| After 3-4 weeks | Retrain the queue forecaster on the store's own data: `python training/train_queue_forecaster.py --from-db data/retail_edge.db` |
| After a layout change | Redraw zones and lines, press Mark as full again |
| Every quarter | Label ~200 shelf photos from your stores and fine-tune the shelf detector with `inventory.ipynb` on Colab |
| Daily (automatic) | `EdgeDB.purge_older_than(days)` removes synced detail rows (data retention) |

## Privacy and legal (India, DPDP Act 2023)

- Put up visible signage: "Camera-based analytics in use. No faces or identities are stored."
- The system stores no images, no faces, no demographic attributes, and uses daily-rotating anonymous
  ids, which keeps personal-data processing to a minimum.
- Keep `privacy.store_frames: false` and `preview_mode: full` in production.
- Document a retention period and who can access the dashboard.

## Public demo (optional)

To show the dashboard to judges or customers without cameras, deploy the Docker image to any container
host (Render, Railway, Azure App Service), run `python run.py demo-data` once, then `python run.py dashboard
--host 0.0.0.0`. The stores are clearly labelled "(simulated)".
