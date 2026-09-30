"""KPI computation and daily / weekly reports (used by API, dashboard and CLI)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Optional

from .storage.db import EdgeDB


def _day_bounds(day: Optional[str] = None):
    d = datetime.strptime(day, "%Y-%m-%d") if day else datetime.now()
    start = datetime(d.year, d.month, d.day)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _sum(db: EdgeDB, store: str, metric: str, t0: float, t1: float, key: Optional[str] = None) -> float:
    sql = "SELECT COALESCE(SUM(value),0) AS v FROM metrics WHERE store_id=? AND metric=? AND ts>=? AND ts<?"
    params = [store, metric, t0, t1]
    if key is not None:
        sql += " AND key=?"
        params.append(key)
    return float(db.query(sql, tuple(params))[0]["v"])


def kpis(db: EdgeDB, store: str, t0: float, t1: float, k_anon: int = 3) -> Dict:
    entries = _sum(db, store, "entries", t0, t1)
    exits = _sum(db, store, "exits", t0, t1)
    bills = db.query("SELECT COUNT(*) AS n, COALESCE(SUM(amount),0) AS rev FROM pos_transactions "
                     "WHERE store_id=? AND ts>=? AND ts<?", (store, t0, t1))[0]
    zones = db.query("SELECT key, SUM(CASE WHEN metric='zone_visits' THEN value ELSE 0 END) AS visits, "
                     "SUM(CASE WHEN metric='dwell_s' THEN value ELSE 0 END) AS dwell, "
                     "SUM(CASE WHEN metric='zone_engaged' THEN value ELSE 0 END) AS engaged "
                     "FROM metrics WHERE store_id=? AND ts>=? AND ts<? AND metric IN "
                     "('zone_visits','dwell_s','zone_engaged') GROUP BY key", (store, t0, t1))
    zone_out = {}
    for z in zones:
        v = z["visits"] or 0
        zone_out[z["key"]] = {"visits": int(v),
                              "avg_dwell_s": round(z["dwell"] / v, 1) if v >= k_anon else None,
                              "engagement_rate": round(z["engaged"] / v, 3) if v >= k_anon else None}
    q = db.query("SELECT AVG(value) AS avg, MAX(value) AS mx FROM metrics WHERE store_id=? AND metric='queue_len' "
                 "AND ts>=? AND ts<?", (store, t0, t1))[0]
    qmax = db.query("SELECT MAX(value) AS mx FROM metrics WHERE store_id=? AND metric='queue_len_max' "
                    "AND ts>=? AND ts<?", (store, t0, t1))[0]
    shelf = db.latest_snapshots(store, "shelf_report")
    shelf_status = {"OK": 0, "LOW_STOCK": 0, "OUT_OF_STOCK": 0}
    fills = []
    for s in shelf:
        shelf_status[s["payload"]["status"]] = shelf_status.get(s["payload"]["status"], 0) + 1
        fills.append(s["payload"]["fill_ratio"])
    alerts = db.query("SELECT level, COUNT(*) AS n FROM alerts WHERE store_id=? AND ts>=? AND ts<? GROUP BY level",
                      (store, t0, t1))
    hours = max(1e-9, (t1 - t0) / 3600)
    staff_eff = db.query("SELECT counter, COUNT(*) AS bills FROM pos_transactions WHERE store_id=? AND ts>=? AND ts<? "
                         "GROUP BY counter", (store, t0, t1))
    return {
        "footfall_in": int(entries), "footfall_out": int(exits),
        "bills": int(bills["n"]), "revenue": round(float(bills["rev"]), 2),
        "conversion_rate": round(bills["n"] / entries, 3) if entries >= k_anon and bills["n"] else None,
        "avg_basket_value": round(bills["rev"] / bills["n"], 2) if bills["n"] else None,
        "avg_queue_len": round(q["avg"], 2) if q["avg"] is not None else None,
        "max_queue_len": round(qmax["mx"], 1) if qmax["mx"] is not None else None,
        "zones": zone_out,
        "shelf_sections": shelf_status,
        "avg_shelf_fill": round(sum(fills) / len(fills), 3) if fills else None,
        "on_shelf_availability": round(1 - shelf_status["OUT_OF_STOCK"] / len(shelf), 3) if shelf else None,
        "alerts": {a["level"]: a["n"] for a in alerts},
        "bills_per_counter_hour": {s["counter"] or "unknown": round(s["bills"] / hours, 2) for s in staff_eff},
    }


def hourly_footfall(db: EdgeDB, store: str, t0: float, t1: float):
    rows = db.metric_series(store, "entries", t0, t1, 3600)
    return [{"hour": datetime.fromtimestamp(r["bucket"]).strftime("%Y-%m-%d %H:00"), "entries": int(r["value"])}
            for r in rows]


def daily_report(db: EdgeDB, store: str, day: Optional[str] = None) -> Dict:
    t0, t1 = _day_bounds(day)
    rep = kpis(db, store, t0, t1)
    rep["date"] = datetime.fromtimestamp(t0).strftime("%Y-%m-%d")
    rep["hourly_footfall"] = hourly_footfall(db, store, t0, t1)
    peak = max(rep["hourly_footfall"], key=lambda r: r["entries"], default=None)
    rep["peak_hour"] = peak["hour"][-5:] if peak else None
    rep["recommendations"] = recommendations(rep)
    return rep


def weekly_report(db: EdgeDB, store: str, end_day: Optional[str] = None) -> Dict:
    _, t1 = _day_bounds(end_day)
    t0 = t1 - 7 * 86400
    days = []
    for i in range(7):
        d0 = t0 + i * 86400
        k = kpis(db, store, d0, d0 + 86400)
        days.append({"date": datetime.fromtimestamp(d0).strftime("%Y-%m-%d (%a)"), "footfall": k["footfall_in"],
                     "bills": k["bills"], "conversion_rate": k["conversion_rate"], "avg_queue_len": k["avg_queue_len"],
                     "max_queue_len": k["max_queue_len"]})
    total = kpis(db, store, t0, t1)
    busiest = max(days, key=lambda d: d["footfall"]) if days else None
    return {"period": f"{days[0]['date']} .. {days[-1]['date']}", "days": days, "totals": total,
            "busiest_day": busiest["date"] if busiest else None, "recommendations": recommendations(total)}


def recommendations(k: Dict) -> list:
    rec = []
    if k.get("max_queue_len") and k["max_queue_len"] >= 5:
        rec.append("Checkout queues reached {:.0f} people - schedule an extra cashier for the peak hour.".format(
            k["max_queue_len"]))
    if k.get("conversion_rate") is not None and k["conversion_rate"] < 0.5:
        rec.append("Conversion below 50 % - check availability of high-demand items and staff assistance on the floor.")
    if k.get("shelf_sections", {}).get("OUT_OF_STOCK"):
        rec.append(f"{k['shelf_sections']['OUT_OF_STOCK']} shelf section(s) out of stock - raise replenishment order.")
    for z, v in (k.get("zones") or {}).items():
        if v.get("engagement_rate") is not None and v["engagement_rate"] < 0.15 and v["visits"] > 20:
            rec.append(f"Zone '{z}' has low engagement ({v['engagement_rate']:.0%}) - consider moving the promotion.")
    return rec or ["Operations within targets."]


def now() -> float:
    return time.time()
