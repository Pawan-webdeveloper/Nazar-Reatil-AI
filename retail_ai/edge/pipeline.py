"""Edge inference pipeline: one worker thread per camera, all processing on-device.

    camera -> frame sampler -> YOLO (+ByteTrack) -> role analytics -> minute aggregates
           -> SQLite (offline) -> alerts / dashboard / optional HQ sync

Camera roles
  entrance : entries / exits / occupancy via virtual line
  floor    : zone dwell time, engagement, movement heatmap
  queue    : queue length, waiting & service time, congestion forecast, counter advice
  shelf    : periodic shelf snapshots -> facings, voids (OOS), stock level, planogram
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ..alerts import AlertManager
from ..analytics.dwell import ZoneDwellTracker
from ..analytics.footfall import LineCounter
from ..analytics.forecasting import QueueForecaster, recommend_counters
from ..analytics.heatmap import Heatmap
from ..analytics.queue_monitor import QueueMonitor
from ..analytics.shelf import ShelfAnalyzer
from ..config import AppConfig, CameraCfg
from ..geometry import foot_point
from ..privacy import AnonymousIdMapper, blur_regions
from ..storage.db import EdgeDB

log = logging.getLogger("retail_ai.pipeline")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class StoreState:
    """Cross-camera state shared inside one store (e.g. entrance footfall feeds the queue forecast)."""

    def __init__(self, counters_open: int):
        self.lock = threading.Lock()
        self.minute_entries: Dict[int, int] = defaultdict(int)
        self.counters_open = counters_open
        self.previews: Dict[str, bytes] = {}
        self.live: Dict[str, dict] = {}

    def add_entries(self, minute: int, n: int):
        with self.lock:
            self.minute_entries[minute] += n

    def entries_in(self, minute: int) -> int:
        with self.lock:
            return self.minute_entries.get(minute, 0)


class FrameSource:
    """Video file, webcam index, RTSP/HTTP stream, or a folder of images (shelf snapshots)."""

    def __init__(self, source: str, realtime: bool):
        self.source = source
        self.realtime = realtime
        self.images: Optional[List[Path]] = None
        p = Path(source)
        if p.is_dir():
            self.images = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
            self.idx = 0
            self.fps = 1.0
            self.is_live = False
            return
        self.is_live = source.isdigit() or "://" in source
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera source {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read(self):
        if self.images is not None:
            if self.idx >= len(self.images):
                return False, None
            img = cv2.imread(str(self.images[self.idx]))
            self.idx += 1
            return img is not None, img
        return self.cap.read()

    def reopen(self):
        if self.images is not None:
            self.idx = 0
            return
        if self.images is None:
            self.cap.release()
            self.cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)

    def release(self):
        if self.images is None:
            self.cap.release()


class MinuteBuffer:
    """Accumulates per-minute metrics and flushes them to the DB at minute boundaries."""

    def __init__(self, db: EdgeDB, store_id: str, camera_id: str):
        self.db, self.store_id, self.camera_id = db, store_id, camera_id
        self.minute: Optional[int] = None
        self.sums: Dict[tuple, float] = defaultdict(float)
        self.samples: Dict[tuple, List[float]] = defaultdict(list)
        self.on_flush: List[Callable[[int, Dict[tuple, float], Dict[tuple, List[float]]], None]] = []

    def add(self, ts: float, metric: str, key: str, value: float):
        self._roll(ts)
        self.sums[(metric, key)] += value

    def sample(self, ts: float, metric: str, key: str, value: float):
        """Gauge value: stored as the per-minute average (and max as metric_max)."""
        self._roll(ts)
        self.samples[(metric, key)].append(value)

    def _roll(self, ts: float):
        m = int(ts // 60)
        if self.minute is None:
            self.minute = m
        if m != self.minute:
            self.flush()
            self.minute = m

    def flush(self):
        if self.minute is None:
            return
        t = self.minute * 60.0
        rows = [(t, self.store_id, self.camera_id, met, key, float(v)) for (met, key), v in self.sums.items()]
        for (met, key), vals in self.samples.items():
            if vals:
                rows.append((t, self.store_id, self.camera_id, met, key, float(np.mean(vals))))
                rows.append((t, self.store_id, self.camera_id, met + "_max", key, float(np.max(vals))))
        self.db.add_metrics(rows)
        for cb in self.on_flush:
            cb(self.minute, dict(self.sums), {k: list(v) for k, v in self.samples.items()})
        self.sums.clear()
        self.samples.clear()


class CameraWorker(threading.Thread):
    def __init__(self, cam: CameraCfg, cfg: AppConfig, db: EdgeDB, alerts: AlertManager, state: StoreState,
                 realtime: bool = True, max_frames: int = 0, start_ts: Optional[float] = None,
                 video_out: Optional[str] = None, loop_files: bool = False):
        super().__init__(daemon=True, name=f"cam-{cam.id}")
        self.loop_files = loop_files
        self.cam, self.cfg, self.db, self.alerts, self.state = cam, cfg, db, alerts, state
        self.realtime, self.max_frames = realtime, max_frames
        self.start_ts = start_ts if start_ts is not None else time.time()
        self.video_out = video_out
        self.stop_event = threading.Event()
        self.frames_processed = 0
        self.infer_ms: Deque[float] = deque(maxlen=200)
        self.buf = MinuteBuffer(db, cfg.store.store_id, cam.id)
        self.anon = AnonymousIdMapper(cfg.privacy.id_salt_rotation_hours)
        self.error: Optional[str] = None
        self._setup_role()

    # ---------------------------------------------------------------- setup
    def _setup_role(self):
        c, m = self.cam, self.cfg.models
        if c.role in ("entrance", "floor", "queue"):
            from .detector import PersonTracker
            self.tracker = PersonTracker(m.person_detector, m.person_conf, c.imgsz or m.imgsz, m.device)
        if c.role == "entrance":
            line = c.count_line or [[0, 0], [1, 0]]
            self.counter = LineCounter((tuple(line[0]), tuple(line[1])), c.entry_direction, margin=c.count_margin)
        elif c.role == "floor":
            self.dwell = ZoneDwellTracker({z.name: [tuple(p) for p in z.polygon] for z in c.zones})
            self.heatmap: Optional[Heatmap] = None
            self._last_heat_save = self.start_ts
        elif c.role == "queue":
            q = self.cfg.queue
            self.queue = QueueMonitor([c_.model_dump() for c_ in c.counters], prior_service_s=q.avg_service_minutes * 60)
            self.forecaster = QueueForecaster(self.cfg.resolve(m.queue_forecaster))
            self.history: Deque[dict] = deque(maxlen=120)
            self.buf.on_flush.append(self._on_queue_minute)
            self._arrivals_seen = 0
        elif c.role == "shelf":
            from .detector import ProductDetector
            self.products = ProductDetector(m.shelf_detector, m.shelf_conf, m.imgsz, m.device)
            inv = self.cfg.inventory
            self.shelf = ShelfAnalyzer(inv.out_of_stock_gap_ratio, inv.low_stock_ratio, inv.min_void_area_ratio,
                                      texture_ratio=inv.texture_ratio,
                                      baseline_min_cover=inv.baseline_min_cover,
                                      baseline_texture_ratio=inv.baseline_texture_ratio)
            self._last_snapshot = -1e18

    # ---------------------------------------------------------------- loop
    def run(self):
        try:
            self._loop()
        except Exception as exc:  # a crashing camera must never take the store down
            self.error = repr(exc)
            log.exception("camera %s crashed", self.cam.id)
        finally:
            self.buf.flush()
            if self.cam.role == "floor":
                for v in self.dwell.flush():
                    self._record_visit(v)
                self._save_heatmap(time.time() if self.realtime else self._ts)

    def _loop(self):
        source = self.cam.source
        if not (source.isdigit() or "://" in source):
            source = str(self.cfg.resolve(source))
        src = FrameSource(source, self.realtime)
        stride = 1 if src.images is not None else max(1, int(round(src.fps / max(self.cam.target_fps, 0.1))))
        writer = None
        idx, fails = -1, 0
        self._ts = self.start_ts
        while not self.stop_event.is_set():
            ok, frame = src.read()
            if not ok:
                if src.is_live and fails < 1000:  # reconnect with back-off (RTSP drops are common)
                    fails += 1
                    time.sleep(min(30, 2 ** min(fails, 5)))
                    src.reopen()
                    continue
                if self.loop_files and self.frames_processed:  # demo mode: replay the file / image folder
                    src.reopen()
                    continue
                break
            fails = 0
            idx += 1
            if idx % stride:
                continue
            if src.is_live or self.realtime:
                ts = time.time()
            elif src.images is not None:
                ts = self.start_ts + idx * self.cam.snapshot_interval_s
            else:
                ts = self.start_ts + idx / src.fps
            self._ts = ts
            h, w = frame.shape[:2]
            if self.cam.role != "shelf" and w != self.cam.process_width:
                s = self.cam.process_width / w
                frame = cv2.resize(frame, (self.cam.process_width, int(round(h * s))))
            t0 = time.perf_counter()
            vis = self.process(frame, ts)
            if vis is None and self.cam.role == "shelf":
                if self.realtime or src.is_live:
                    time.sleep(0.5)  # shelf snapshot not due yet: do not spin the CPU
                continue
            self.infer_ms.append((time.perf_counter() - t0) * 1000)
            self.frames_processed += 1
            if vis is not None:
                ok_, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok_:
                    self.state.previews[self.cam.id] = jpg.tobytes()  # memory only, never disk
                if self.video_out:
                    if writer is None:
                        Path(self.video_out).parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(self.video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 max(1.0, src.fps / stride), (vis.shape[1], vis.shape[0]))
                        writer_size = (vis.shape[1], vis.shape[0])
                    if (vis.shape[1], vis.shape[0]) != writer_size:  # e.g. shelf snapshots of different sizes
                        vis = cv2.resize(vis, writer_size)
                    writer.write(vis)
            if self.max_frames and self.frames_processed >= self.max_frames:
                break
        src.release()
        if writer is not None:
            writer.release()

    # ---------------------------------------------------------------- roles
    def process(self, frame: np.ndarray, ts: float) -> Optional[np.ndarray]:
        role = self.cam.role
        if role == "shelf":
            return self._process_shelf(frame, ts)
        tracks = self.tracker.track(frame)
        feet = {tid: foot_point(b) for tid, b, _ in tracks}
        live = {"ts": ts, "people": len(tracks), "fps": self.fps()}
        if role == "entrance":
            for ev in self.counter.update(ts, feet):
                self.buf.add(ts, "entries" if ev.direction == "in" else "exits", "", 1)
                if ev.direction == "in":
                    self.state.add_entries(int(ts // 60), 1)
            self.counter.forget(feet.keys())
            self.buf.sample(ts, "people_visible", "", len(tracks))
            live.update(entries=self.counter.entries, exits=self.counter.exits, occupancy=self.counter.occupancy)
        elif role == "floor":
            if self.heatmap is None:
                self.heatmap = Heatmap(frame.shape[1], frame.shape[0], self.cam.heatmap_cell)
                self._heat_bg = cv2.GaussianBlur(frame, (0, 0), 3)
            self.heatmap.add(feet.values())
            for v in self.dwell.update(ts, feet):
                self._record_visit(v)
            occ = self.dwell.current_occupancy()
            for z, n in occ.items():
                self.buf.sample(ts, "zone_occupancy", z, n)
            if ts - self._last_heat_save >= 300:
                self._save_heatmap(ts)
                self._last_heat_save = ts
            live.update(zone_occupancy=occ)
        elif role == "queue":
            self.queue.update(ts, feet)
            st = self.queue.status(ts)
            for s in st:
                self.buf.sample(ts, "queue_len", s["counter"], s["queue_length"])
            n_arr = sum(len(c.arrivals) for c in self.queue.counters)
            if n_arr > self._arrivals_seen:
                self.buf.add(ts, "checkout_arrivals", "", n_arr - self._arrivals_seen)
                self._arrivals_seen = n_arr
            self._queue_rules(ts, st)
            live.update(counters=st)
        self.state.live[self.cam.id] = live
        return self._draw(frame, tracks)

    def _record_visit(self, v):
        t = v.end
        self.buf.add(t, "zone_visits", v.zone, 1)
        self.buf.add(t, "dwell_s", v.zone, v.seconds)
        if v.seconds >= self.dwell.engaged_s:
            self.buf.add(t, "zone_engaged", v.zone, 1)

    def _save_heatmap(self, ts: float):
        if self.cam.role == "floor" and getattr(self, "heatmap", None) is not None:
            self.db.add_snapshot(ts, self.cfg.store.store_id, "heatmap", self.cam.id,
                                 {"cell": self.heatmap.cell, "w": self.heatmap.w, "h": self.heatmap.h,
                                  "grid": np.round(self.heatmap.grid, 2).tolist()})
            self.db.add_snapshot(ts, self.cfg.store.store_id, "zone_summary", self.cam.id, self.dwell.summary())

    def _queue_rules(self, ts: float, status: List[dict]):
        q = self.cfg.queue
        for s in status:
            if s["queue_length"] > q.max_queue_per_counter:
                self.alerts.raise_alert("warning", "QUEUE_BUILDUP", s["counter"],
                                        f"{s['counter']}: {s['queue_length']} people waiting "
                                        f"(limit {q.max_queue_per_counter}), est. wait "
                                        f"{s['expected_wait_new_customer_s'] / 60:.1f} min", s, ts)

    def _on_queue_minute(self, minute: int, sums: dict, samples: dict):
        """Every minute: append history, run the forecaster, recommend counters."""
        q = self.cfg.queue
        qlen = sum(np.mean(v) for (met, _k), v in samples.items() if met == "queue_len" and v)
        row = {"ts": pd.Timestamp(minute * 60, unit="s", tz="UTC").tz_convert(self.cfg.store.timezone).tz_localize(None),
               "entries": self.state.entries_in(minute),
               "checkout_arrivals": sums.get(("checkout_arrivals", ""), 0.0),
               "queue_len": float(qlen), "counters_open": self.state.counters_open}
        self.history.append(row)
        self.db.add_metrics([(minute * 60.0, self.cfg.store.store_id, self.cam.id, "counters_open", "",
                              float(self.state.counters_open))])
        hist = pd.DataFrame(list(self.history))
        pred = self.forecaster.predict_from_history(hist) if self.forecaster.ready else None
        lam = hist["checkout_arrivals"].tail(15).mean() if len(hist) else 0.0
        svc = [s for c in self.queue.counters for s in c.services]
        avg_service_min = (np.mean(svc) / 60) if len(svc) >= 5 else q.avg_service_minutes
        if pred is not None:
            # forecast queue -> arrival rate via Little's law approximation
            lam = max(lam, pred / max(avg_service_min, 1e-3) / max(self.state.counters_open, 1) * 0.5)
        rec = recommend_counters(float(lam), float(avg_service_min), q.target_wait_minutes, q.counters_total)
        payload = {"minute": minute, "queue_now": round(float(qlen), 2),
                   "forecast_queue": None if pred is None else round(pred, 2),
                   "horizon_min": self.forecaster.horizon, "arrival_rate_per_min": round(float(lam), 3),
                   "avg_service_min": round(float(avg_service_min), 2), "counters_open": self.state.counters_open,
                   **rec}
        self.db.add_snapshot(minute * 60.0, self.cfg.store.store_id, "queue_forecast", self.cam.id, payload)
        limit = q.max_queue_per_counter * max(self.state.counters_open, 1)
        if pred is not None and pred > limit:
            self.alerts.raise_alert("warning", "QUEUE_FORECAST", self.cam.id,
                                    f"Queue predicted to reach {pred:.1f} in {self.forecaster.horizon} min "
                                    f"(limit {limit}). Open {rec['recommended_counters']} counters.", payload,
                                    minute * 60.0)
        # need >= 10 minutes of observations before advising staff (avoids noisy advice at start-up)
        if len(self.history) >= 10 and rec["recommended_counters"] > self.state.counters_open:
            self.alerts.raise_alert("info", "OPEN_COUNTER", self.cam.id,
                                    f"Recommend {rec['recommended_counters']} billing counters "
                                    f"(currently {self.state.counters_open}) to keep wait under "
                                    f"{q.target_wait_minutes:g} min", payload, minute * 60.0)

    def _process_shelf(self, frame: np.ndarray, ts: float) -> Optional[np.ndarray]:
        if ts - self._last_snapshot < self.cam.snapshot_interval_s:
            return None  # shelves change slowly: analyse one snapshot per interval
        self._last_snapshot = ts
        boxes, _conf = self.products.detect(frame)
        h, w = frame.shape[:2]
        vis = frame.copy()
        if self.cfg.privacy.preview_mode == "full":  # shoppers may pass in front of shelf cameras too
            k = max(3, (frame.shape[1] // 30) | 1)
            vis = cv2.GaussianBlur(vis, (k, k), 0)
        sections = self.cam.shelf_sections or []
        if not sections:
            from ..config import ShelfSection
            sections = [ShelfSection(name="shelf")]
        baselines = {r["key"]: r["payload"] for r in self.db.latest_snapshots(self.cfg.store.store_id, "shelf_baseline")}
        for sec in sections:
            # planogram numbers from the config win; otherwise use the "fully stocked" calibration
            # snapshot of this very camera -> systematic counting bias (case vs unit, peg hooks) cancels out
            base = baselines.get(f"{self.cam.id}/{sec.name}", {})
            exp_f = sec.expected_facings or int(base.get("facings", 0))
            exp_r = sec.expected_rows or int(base.get("rows", 0))
            rep = self.shelf.analyze(boxes, (w, h), sec.name, sec.polygon, exp_f, exp_r, image=frame,
                                     baseline_boxes=base.get("boxes"))
            d = rep.to_dict()
            self.db.add_snapshot(ts, self.cfg.store.store_id, "shelf_report", f"{self.cam.id}/{sec.name}", d)
            self.buf.sample(ts, "shelf_fill_ratio", sec.name, rep.fill_ratio)
            self.buf.sample(ts, "shelf_facings", sec.name, rep.facings)
            if rep.status == "OUT_OF_STOCK":
                self.alerts.raise_alert("critical", "OUT_OF_STOCK", f"{self.cam.id}/{sec.name}",
                                        f"{sec.name}: out of stock - replenish now ({rep.facings} facings left)", d, ts)
            elif rep.status == "LOW_STOCK":
                self.alerts.raise_alert("warning", "LOW_STOCK", f"{self.cam.id}/{sec.name}",
                                        f"{sec.name}: low stock, fill {rep.fill_ratio:.0%}, "
                                        f"{len(rep.voids)} empty slot(s)", d, ts)
            if rep.planogram_compliance is not None and rep.planogram_compliance < 0.8:
                self.alerts.raise_alert("warning", "PLANOGRAM", f"{self.cam.id}/{sec.name}",
                                        f"{sec.name}: planogram compliance {rep.planogram_compliance:.0%}", d, ts)
            for v in rep.voids:
                x1, y1, x2, y2 = map(int, v.box)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
            self.state.live[self.cam.id] = {"ts": ts, "section": sec.name, **{k: d[k] for k in
                                            ("facings", "fill_ratio", "status")}, "voids": len(rep.voids)}
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        return vis

    # ---------------------------------------------------------------- drawing
    def _draw(self, frame: np.ndarray, tracks) -> np.ndarray:
        boxes = [b for _t, b, _c in tracks]
        pv = self.cfg.privacy
        vis = frame.copy()
        if pv.preview_mode == "full":
            k = max(3, (frame.shape[1] // 24) | 1)  # odd kernel ~4 % of width: layout visible, identities not
            vis = cv2.GaussianBlur(vis, (k, k), 0)
        if pv.blur_people_in_previews:
            vis = blur_regions(vis, boxes)
        c = self.cam
        for tid, b, _conf in tracks:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(vis, self.anon.anon(c.id, tid)[:6], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 220, 0), 1, cv2.LINE_AA)
        if c.role == "entrance" and c.count_line:
            (ax, ay), (bx, by) = c.count_line
            cv2.line(vis, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 255), 2)
            cv2.putText(vis, f"IN {self.counter.entries}  OUT {self.counter.exits}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        if c.role == "floor":
            for z in c.zones:
                pts = np.array(z.polygon, dtype=np.int32)
                cv2.polylines(vis, [pts], True, (255, 180, 0), 2)
                cv2.putText(vis, z.name, tuple(pts[0] + [5, 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
        if c.role == "queue":
            for cc in c.counters:
                cv2.polylines(vis, [np.array(cc.queue_polygon, np.int32)], True, (255, 0, 255), 2)
                cv2.polylines(vis, [np.array(cc.service_polygon, np.int32)], True, (0, 128, 255), 2)
            st = self.queue.status(self._ts)
            txt = "  ".join(f"{s['counter']}: Q={s['queue_length']}" for s in st)
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"{self.fps():.1f} FPS on-device", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def fps(self) -> float:
        return 1000.0 / np.mean(self.infer_ms) if self.infer_ms else 0.0


class EdgeNode:
    """Runs all cameras of a store + alerting + optional HQ sync."""

    def __init__(self, cfg: AppConfig, realtime: bool = True, max_frames: int = 0,
                 start_ts: Optional[float] = None, video_dir: Optional[str] = None,
                 db: Optional[EdgeDB] = None, cameras: Optional[List[str]] = None, loop_files: bool = False):
        from ..sync import SyncAgent

        self.cfg = cfg
        self.db = db or EdgeDB(cfg.resolve(cfg.storage.db_path))
        self.db.touch_store(cfg.store.store_id, cfg.store.name)
        a = cfg.alerts
        self.alerts = AlertManager(self.db, cfg.store.store_id, a.cooldown_minutes, a.webhook_url, a.notify_levels)
        n_counters = sum(len(c.counters) for c in cfg.cameras if c.role == "queue") or 1
        self.state = StoreState(counters_open=n_counters)
        self.workers = []
        for cam in cfg.cameras:
            if cameras and cam.id not in cameras:
                continue
            out = str(Path(video_dir) / f"{cam.id}.mp4") if video_dir else None
            try:
                self.workers.append(CameraWorker(cam, cfg, self.db, self.alerts, self.state, realtime,
                                                 max_frames, start_ts, out, loop_files))
            except Exception as exc:
                log.error("camera %s disabled: %s", cam.id, exc)
        s = cfg.sync
        self.sync = SyncAgent(self.db, cfg.store.store_id, s.central_url, s.api_key_env, s.batch_size,
                              s.interval_s, cfg.store.name)

    def start(self):
        for w in self.workers:
            w.start()
        self.sync.start()

    def join(self):
        for w in self.workers:
            w.join()

    def stop(self):
        for w in self.workers:
            w.stop_event.set()
        self.sync.stop()

    def health(self) -> dict:
        return {"store_id": self.cfg.store.store_id,
                "cameras": [{"id": w.cam.id, "role": w.cam.role, "alive": w.is_alive(), "fps": round(w.fps(), 2),
                             "frames": w.frames_processed, "error": w.error} for w in self.workers],
                "sync": {"enabled": bool(self.sync.url), "last_success": self.sync.last_success,
                         "last_error": self.sync.last_error}}
