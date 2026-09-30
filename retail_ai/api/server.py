"""FastAPI service: store operations dashboard + REST API + HQ ingestion.

The same app runs
  * on each store's edge box (reads its local SQLite; works without internet), and
  * at HQ (receives /api/ingest pushes from many stores -> centralised multi-store view).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import reports
from ..analytics.heatmap import Heatmap
from ..config import AppConfig
from ..integrations.pos import build_replenishment_order, load_inventory_master
from ..storage.db import EdgeDB

STATIC = Path(__file__).resolve().parents[1] / "dashboard" / "static"


class PosBill(BaseModel):
    bill_id: str
    timestamp: float
    amount: float
    items: int = 0
    counter: str = ""


class IngestBody(BaseModel):
    store_id: str
    store_name: str = ""
    table: str
    rows: List[dict]


class CountersBody(BaseModel):
    counters_open: int


def create_app(cfg: AppConfig, db: EdgeDB, node=None) -> FastAPI:
    app = FastAPI(title="RetailSense AI", version="1.0.0",
                  description="Privacy-first edge retail intelligence: footfall, dwell, heatmaps, "
                              "inventory, queues, alerts and reports.")
    api_key = os.environ.get("RETAIL_API_KEY", "")

    def require_key(x_api_key: str = Header(default="")):
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "invalid API key")

    def window(day: Optional[str]):
        return reports._day_bounds(day)

    # ------------------------------------------------------------ meta
    @app.get("/api/health")
    def health():
        return {"status": "ok", "time": time.time(), "store_id": cfg.store.store_id,
                "edge": node.health() if node else None}

    @app.get("/api/stores")
    def stores():
        rows = db.query("SELECT store_id, name, last_seen FROM stores ORDER BY store_id")
        t0, t1 = window(None)
        for r in rows:
            k = reports.kpis(db, r["store_id"], t0, t1)
            r["today"] = {x: k[x] for x in ("footfall_in", "bills", "conversion_rate", "avg_queue_len",
                                            "max_queue_len", "on_shelf_availability", "alerts")}
            r["local"] = r["store_id"] == cfg.store.store_id
            r["online"] = r["local"] or (time.time() - r["last_seen"]) < 600
        return rows

    # ------------------------------------------------------------ KPIs / analytics
    @app.get("/api/stores/{sid}/kpis")
    def kpis(sid: str, day: Optional[str] = None):
        t0, t1 = window(day)
        return reports.kpis(db, sid, t0, t1, cfg.privacy.min_group_size_for_reports)

    @app.get("/api/stores/{sid}/series")
    def series(sid: str, metric: str, hours: float = Query(24, gt=0, le=24 * 31), bucket: int = Query(3600, ge=60),
               agg: str = "SUM", key: Optional[str] = None, end: Optional[float] = None):
        t1 = end or time.time()
        try:
            return db.metric_series(sid, metric, t1 - hours * 3600, t1, bucket, key, agg)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/stores/{sid}/footfall_trend")
    def footfall_trend(sid: str, days: int = Query(14, ge=1, le=90), end_day: Optional[str] = None):
        """Daily footfall + day-of-week x hour matrix (for the weekly pattern chart)."""
        t1 = reports._day_bounds(end_day)[1]
        t0 = t1 - days * 86400
        rows = db.metric_series(sid, "entries", t0, t1, 3600)
        daily, matrix = {}, np.zeros((7, 24))
        cnt = np.zeros((7, 24))
        import datetime as _dt
        for r in rows:
            d = _dt.datetime.fromtimestamp(r["bucket"])
            daily[d.strftime("%Y-%m-%d")] = daily.get(d.strftime("%Y-%m-%d"), 0) + r["value"]
            matrix[d.weekday(), d.hour] += r["value"]
            cnt[d.weekday(), d.hour] += 1
        avg = np.divide(matrix, np.maximum(cnt, 1))
        return {"daily": [{"date": k, "entries": int(v)} for k, v in sorted(daily.items())],
                "dow_hour_avg": np.round(avg, 1).tolist()}

    @app.get("/api/stores/{sid}/latest_day")
    def latest_day(sid: str):
        """Most recent local date that has any metric (dashboard opens there if today is still empty)."""
        row = db.query("SELECT MAX(ts) AS t FROM metrics WHERE store_id=?", (sid,))[0]
        if not row["t"]:
            return {"day": None, "ts": None}
        import datetime as _dt
        return {"day": _dt.datetime.fromtimestamp(row["t"]).strftime("%Y-%m-%d"), "ts": row["t"]}

    @app.get("/api/stores/{sid}/daily_kpis")
    def daily_kpis(sid: str, days: int = Query(14, ge=1, le=90), end_day: Optional[str] = None):
        """Per-day KPI series for sparklines and trend charts."""
        import datetime as _dt
        _t0, t_end = reports._day_bounds(end_day)
        out = []
        for i in range(days - 1, -1, -1):
            d1 = t_end - i * 86400
            d0 = d1 - 86400
            k = reports.kpis(db, sid, d0, d1, cfg.privacy.min_group_size_for_reports)
            fill = db.query("SELECT AVG(value) AS v FROM metrics WHERE store_id=? AND metric='shelf_fill_ratio' "
                            "AND ts>=? AND ts<?", (sid, d0, d1))[0]["v"]
            out.append({"date": _dt.datetime.fromtimestamp(d0).strftime("%Y-%m-%d"), "footfall": k["footfall_in"],
                        "exits": k["footfall_out"], "bills": k["bills"], "conversion": k["conversion_rate"],
                        "avg_queue": k["avg_queue_len"], "shelf_fill": None if fill is None else round(fill, 3),
                        "alerts": sum(k["alerts"].values())})
        return out

    @app.get("/api/stores/{sid}/zones")
    def zones(sid: str, day: Optional[str] = None):
        t0, t1 = window(day)
        return reports.kpis(db, sid, t0, t1, cfg.privacy.min_group_size_for_reports)["zones"]

    @app.get("/api/stores/{sid}/heatmap/{camera}.png")
    def heatmap_png(sid: str, camera: str):
        snaps = [s for s in db.latest_snapshots(sid, "heatmap") if s["key"] == camera]
        if not snaps:
            raise HTTPException(404, "no heatmap yet")
        p = snaps[0]["payload"]
        hm = Heatmap(p["w"], p["h"], p["cell"])
        hm.merge(p["grid"])
        import cv2
        ok, png = cv2.imencode(".png", hm.render())
        return Response(png.tobytes(), media_type="image/png")

    @app.get("/api/stores/{sid}/heatmaps")
    def heatmap_list(sid: str):
        return [s["key"] for s in db.latest_snapshots(sid, "heatmap")]

    @app.get("/api/stores/{sid}/queue")
    def queue(sid: str):
        fc = db.latest_snapshots(sid, "queue_forecast")
        live = {k: v for k, v in (node.state.live.items() if node else []) if "counters" in v}
        hist = db.metric_series(sid, "queue_len", time.time() - 6 * 3600, time.time(), 300, agg="AVG")
        return {"forecast": [f["payload"] | {"camera": f["key"], "ts": f["ts"]} for f in fc], "live": live,
                "history_5min": hist}

    @app.get("/api/stores/{sid}/inventory")
    def inventory(sid: str):
        snaps = db.latest_snapshots(sid, "shelf_report")
        return [s["payload"] | {"key": s["key"], "ts": s["ts"]} for s in snaps]

    @app.post("/api/stores/{sid}/inventory/baseline", dependencies=[Depends(require_key)])
    def mark_full(sid: str, key: str):
        """Staff pressed 'Mark as full' after restocking: the latest snapshot becomes the 100 % reference."""
        snap = [s for s in db.latest_snapshots(sid, "shelf_report") if s["key"] == key]
        if not snap:
            raise HTTPException(404, "no shelf snapshot for this section yet")
        pl = snap[0]["payload"]
        base = {"facings": pl["facings"], "rows": pl["rows"], "boxes": pl.get("boxes", []), "from_ts": snap[0]["ts"]}
        db.add_snapshot(time.time(), sid, "shelf_baseline", key, base)
        return {"key": key, "baseline": base}

    @app.get("/api/stores/{sid}/inventory/baselines")
    def baselines(sid: str):
        return {s["key"]: s["payload"] for s in db.latest_snapshots(sid, "shelf_baseline")}

    @app.get("/api/stores/{sid}/replenishment")
    def replenishment(sid: str):
        snaps = [s["payload"] | {"section": s["key"]} for s in db.latest_snapshots(sid, "shelf_report")]
        return build_replenishment_order(sid, snaps, load_inventory_master(cfg.resolve(cfg.integrations.inventory_csv)))

    # ------------------------------------------------------------ alerts
    @app.get("/api/stores/{sid}/alerts")
    def alerts(sid: str, limit: int = Query(50, le=500), unacked: bool = False):
        sql = "SELECT id, ts, level, type, key, message, acknowledged FROM alerts WHERE store_id=?"
        if unacked:
            sql += " AND acknowledged=0"
        return db.query(sql + " ORDER BY ts DESC LIMIT ?", (sid, limit))

    @app.post("/api/alerts/{alert_id}/ack", dependencies=[Depends(require_key)])
    def ack(alert_id: int):
        if not db.acknowledge_alert(alert_id):
            raise HTTPException(404, "alert not found")
        return {"ok": True}

    # ------------------------------------------------------------ reports
    @app.get("/api/stores/{sid}/reports/daily")
    def daily(sid: str, day: Optional[str] = None):
        return reports.daily_report(db, sid, day)

    @app.get("/api/stores/{sid}/reports/weekly")
    def weekly(sid: str, end_day: Optional[str] = None):
        return reports.weekly_report(db, sid, end_day)

    # ------------------------------------------------------------ live edge
    @app.get("/api/cameras")
    def cameras():
        return [{"id": c.id, "role": c.role, "live": node.state.live.get(c.id) if node else None}
                for c in cfg.cameras]

    @app.get("/api/cameras/{cam}/preview.jpg")
    def preview(cam: str):
        if not node or cam not in node.state.previews:
            raise HTTPException(404, "no live preview (edge node not running in this process)")
        return Response(node.state.previews[cam], media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/counters_open", dependencies=[Depends(require_key)])
    def set_counters(body: CountersBody):
        if not node:
            raise HTTPException(409, "edge node not running")
        node.state.counters_open = max(1, body.counters_open)
        return {"counters_open": node.state.counters_open}

    # ------------------------------------------------------------ integrations
    @app.post("/api/pos/transactions", dependencies=[Depends(require_key)])
    def pos(bills: List[PosBill], store_id: Optional[str] = None):
        sid = store_id or cfg.store.store_id
        n = db.upsert_pos([(b.bill_id, b.timestamp, sid, b.amount, b.items, b.counter) for b in bills])
        return {"received": n}

    @app.post("/api/ingest")
    def ingest(body: IngestBody, x_api_key: str = Header(default="")):
        key = os.environ.get("RETAIL_INGEST_KEY", "")
        if key and x_api_key != key:
            raise HTTPException(401, "invalid ingest key")
        db.touch_store(body.store_id, body.store_name)
        r = body.rows
        if body.table == "metrics":
            db.add_metrics((x["ts"], body.store_id, x["camera_id"], x["metric"], x.get("key", ""), x["value"]) for x in r)
        elif body.table == "alerts":
            for x in r:
                db.add_alert(x["ts"], body.store_id, x["level"], x["type"], x.get("key", ""), x["message"])
        elif body.table == "snapshots":
            import json as _j
            for x in r:
                payload = x["payload"] if isinstance(x["payload"], (dict, list)) else _j.loads(x["payload"])
                db.add_snapshot(x["ts"], body.store_id, x["kind"], x.get("key", ""), payload)
        else:
            raise HTTPException(400, "unknown table")
        return {"accepted": len(r)}

    # ------------------------------------------------------------ dashboard
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config_public():
        return {"store_id": cfg.store.store_id, "store_name": cfg.store.name,
                "cameras": [{"id": c.id, "role": c.role} for c in cfg.cameras],
                "queue": cfg.queue.model_dump(), "privacy": cfg.privacy.model_dump()}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def revalidate_dashboard(request, call_next):
        """Dashboard files must be revalidated so an update is visible without a hard refresh."""
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.exception_handler(Exception)
    async def unhandled(_req, exc):  # never leak stack traces to the browser
        return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)[:300]})

    return app
