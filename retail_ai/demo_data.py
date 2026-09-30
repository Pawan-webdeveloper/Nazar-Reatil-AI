"""Populate a database with SIMULATED history so the dashboard can be explored
before real cameras have run for weeks. Stores created here are labelled
"(simulated)" so demo numbers are never mistaken for real measurements.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import AppConfig
from .simulation import PROFILES, simulate_store
from .storage.db import EdgeDB

DEMO_STORES = [("supermarket", None), ("neighbourhood", "STORE-MYS-014"), ("hypermarket", "STORE-HYD-002")]


def populate(db: EdgeDB, cfg: AppConfig, days: int = 28, seed: int = 7) -> int:
    rng = np.random.default_rng(seed)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start = today - timedelta(days=days - 1)
    now_ts = time.time()
    total = 0
    for fmt, sid in DEMO_STORES:
        store_id = sid or cfg.store.store_id
        name = f"{cfg.store.name if sid is None else fmt.title() + ' demo store'} (simulated)"
        for table in ("metrics", "alerts", "snapshots", "pos_transactions"):
            db.execute(f"DELETE FROM {table} WHERE store_id=?", (store_id,))
        db.touch_store(store_id, name)
        df = simulate_store(PROFILES[fmt], start=start.strftime("%Y-%m-%d"), days=days, seed=seed + len(store_id))
        df = df[df["ts"].map(lambda t: t.to_pydatetime().timestamp()) <= now_ts]  # naive = local time
        rows, bills = [], []
        zones = ["promo_display", "aisle_snacks", "dairy", "personal_care"]
        for r in df.itertuples():
            t = r.ts.to_pydatetime().timestamp()  # naive local time -> epoch
            if r.entries:
                rows.append((t, store_id, "cam_entrance", "entries", "", float(r.entries)))
            exits = rng.binomial(r.entries, 0.97) if r.entries else 0
            if exits:
                rows.append((t, store_id, "cam_entrance", "exits", "", float(exits)))
            if r.checkout_arrivals:
                rows.append((t, store_id, "cam_checkout", "checkout_arrivals", "", float(r.checkout_arrivals)))
            per = r.queue_len / max(1, r.counters_open)
            rows.append((t, store_id, "cam_checkout", "queue_len", "counter_1", float(per)))
            rows.append((t, store_id, "cam_checkout", "queue_len_max", "counter_1", float(per * 1.3)))
            rows.append((t, store_id, "cam_checkout", "counters_open", "", float(r.counters_open)))
            for z_i, z in enumerate(zones):
                v = rng.poisson(r.entries * (0.5, 0.35, 0.3, 0.2)[z_i])
                if v:
                    dwell = rng.gamma(2.0, (14, 9, 11, 22)[z_i], v)
                    rows.append((t, store_id, "cam_floor", "zone_visits", z, float(v)))
                    rows.append((t, store_id, "cam_floor", "dwell_s", z, float(dwell.sum())))
                    rows.append((t, store_id, "cam_floor", "zone_engaged", z, float((dwell >= 10).sum())))
            for k in range(int(r.checkout_arrivals)):
                items = int(rng.integers(1, 25))
                bills.append((f"{store_id}-{int(t)}-{k}", t + rng.random() * 60, store_id,
                              round(float(items * rng.uniform(40, 180)), 2), items,
                              f"counter_{1 + int(rng.integers(0, r.counters_open))}"))
            if per > cfg.queue.max_queue_per_counter and rng.random() < 0.05:
                db.add_alert(t, store_id, "warning", "QUEUE_BUILDUP", "counter_1",
                             f"counter_1: {per:.0f} people waiting (limit {cfg.queue.max_queue_per_counter})")
        db.add_metrics(rows)
        db.upsert_pos(bills)
        total += len(rows) + len(bills)
        _shelf_and_heatmap(db, store_id, rng, now_ts)
        # latest queue forecast snapshot so the Queues tab has content
        last = df.iloc[-1]
        from .analytics.forecasting import recommend_counters
        lam = float(df["checkout_arrivals"].tail(15).mean())
        rec = recommend_counters(lam, PROFILES[fmt].service_mean_min, cfg.queue.target_wait_minutes, cfg.queue.counters_total)
        db.add_snapshot(now_ts, store_id, "queue_forecast", "cam_checkout",
                        {"queue_now": float(last.queue_len), "forecast_queue": None, "horizon_min": 15,
                         "arrival_rate_per_min": round(lam, 3), "avg_service_min": PROFILES[fmt].service_mean_min,
                         "counters_open": int(last.counters_open), **rec})
    return total


def _shelf_and_heatmap(db: EdgeDB, store_id: str, rng, now_ts: float):
    sections = [("shampoo", 40), ("biscuits", 64), ("detergent", 36), ("snacks", 72), ("dairy_chiller", 30), ("rice_atta", 24)]
    fills = rng.permutation([1.0, 0.95, 0.92, 0.55, 0.2, 0.85])  # 1 out-of-stock, 1 low, rest OK
    for (name, cap), f in zip(sections, fills):
        facings = int(round(cap * f))
        fill = facings / cap
        status = "OUT_OF_STOCK" if fill < 0.25 else ("LOW_STOCK" if fill < 0.6 else "OK")
        n_voids = 0 if fill >= 0.95 else int(1 + (1 - fill) * 6)
        rep = {"section": name, "facings": facings, "rows": 4, "capacity_estimate": cap, "fill_ratio": round(fill, 3),
               "voids": [{"row": int(rng.integers(0, 4)), "box": [0, 0, 0, 0], "missing_facings": 1} for _ in range(n_voids)],
               "stock_ratio_vs_planogram": round(fill, 3), "planogram_compliance": round(min(1, fill + 0.1), 3),
               "status": status, "issues": [f"{n_voids} empty slot(s)"] if n_voids else []}
        db.add_snapshot(now_ts - rng.integers(30, 600), store_id, "shelf_report", f"cam_shelf_A/{name}", rep)
        if status != "OK":
            db.add_alert(now_ts - rng.integers(60, 3000), store_id, "critical" if status == "OUT_OF_STOCK" else "warning",
                         status, f"cam_shelf_A/{name}", f"{name}: {status.replace('_', ' ').lower()} (fill {fill:.0%})")
    # heatmap: two hotspots (promo + aisle) on a 960x540 view
    w, h, cell = 960, 540, 16
    gy, gx = np.mgrid[0:h // cell + 1, 0:w // cell + 1]
    grid = 800 * np.exp(-(((gx - 15) / 6) ** 2 + ((gy - 22) / 5) ** 2)) + 500 * np.exp(-(((gx - 42) / 4) ** 2 + ((gy - 20) / 9) ** 2))
    grid += rng.poisson(20, grid.shape)
    db.add_snapshot(now_ts, store_id, "heatmap", "cam_floor", {"cell": cell, "w": w, "h": h, "grid": np.round(grid, 1).tolist()})
