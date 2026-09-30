"""Storage, API, POS integration, HQ multi-store sync and offline behaviour."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from retail_ai.api.server import create_app
from retail_ai.config import load_config
from retail_ai.integrations.pos import build_replenishment_order, import_pos_csv
from retail_ai.storage.db import EdgeDB
from retail_ai.sync import SyncAgent


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def db(tmp_path):
    d = EdgeDB(tmp_path / "edge.db")
    yield d
    d.close()


def _seed(db, store="S1"):
    t = time.time() - 3600
    db.touch_store(store, "Test store")
    db.add_metrics([(t, store, "cam_entrance", "entries", "", 10.0), (t + 60, store, "cam_entrance", "entries", "", 5.0),
                    (t, store, "cam_floor", "zone_visits", "promo", 4.0), (t, store, "cam_floor", "dwell_s", "promo", 80.0),
                    (t, store, "cam_floor", "zone_engaged", "promo", 2.0),
                    (t, store, "cam_checkout", "queue_len", "counter_1", 3.0),
                    (t, store, "cam_checkout", "queue_len_max", "counter_1", 6.0)])
    db.upsert_pos([("b1", t, store, 250.0, 3, "counter_1"), ("b2", t + 10, store, 100.0, 1, "counter_1")])
    db.add_snapshot(t, store, "shelf_report", "cam_shelf_A/shampoo",
                    {"section": "shampoo", "facings": 5, "rows": 2, "voids": [], "capacity_estimate": 40,
                     "fill_ratio": 0.125, "stock_ratio_vs_planogram": 0.125, "planogram_compliance": 0.3,
                     "status": "OUT_OF_STOCK", "issues": []})
    return db.add_alert(t, store, "critical", "OUT_OF_STOCK", "cam_shelf_A/shampoo", "shampoo out of stock")


def test_kpis_and_endpoints(cfg, db):
    alert_id = _seed(db)
    c = TestClient(create_app(cfg, db))
    assert c.get("/api/health").json()["status"] == "ok"
    k = c.get("/api/stores/S1/kpis").json()
    assert k["footfall_in"] == 15 and k["bills"] == 2
    assert k["conversion_rate"] == pytest.approx(2 / 15, abs=1e-3)
    assert k["zones"]["promo"]["avg_dwell_s"] == 20.0
    assert k["shelf_sections"]["OUT_OF_STOCK"] == 1 and k["on_shelf_availability"] == 0.0
    assert c.get("/api/stores/S1/alerts").json()[0]["type"] == "OUT_OF_STOCK"
    assert c.post(f"/api/alerts/{alert_id}/ack").json() == {"ok": True}
    assert c.get("/api/stores/S1/alerts?unacked=true").json() == []
    d = c.get("/api/stores/S1/reports/daily").json()
    assert d["footfall_in"] == 15 and d["recommendations"]
    w = c.get("/api/stores/S1/reports/weekly").json()
    assert len(w["days"]) == 7
    assert c.get("/api/stores/S1/footfall_trend?days=7").json()["daily"][0]["entries"] == 15
    assert c.get("/api/stores/S1/replenishment").json()["lines"][0]["priority"] == "HIGH"
    assert c.get("/").status_code == 200
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/api/stores/S1/series?metric=entries&agg=DROP").status_code == 400


def test_k_anonymity_suppresses_small_groups(cfg, db):
    t = time.time() - 100
    db.add_metrics([(t, "S2", "cam_floor", "zone_visits", "tiny", 2.0), (t, "S2", "cam_floor", "dwell_s", "tiny", 30.0)])
    k = TestClient(create_app(cfg, db)).get("/api/stores/S2/kpis").json()
    assert k["zones"]["tiny"]["avg_dwell_s"] is None


def test_pos_push_and_csv_import(cfg, db, tmp_path):
    c = TestClient(create_app(cfg, db))
    r = c.post("/api/pos/transactions?store_id=S3", json=[{"bill_id": "x1", "timestamp": time.time(), "amount": 99.5}])
    assert r.json() == {"received": 1}
    folder = tmp_path / "pos"
    folder.mkdir()
    (folder / "bills.csv").write_text("timestamp,bill_id,amount,items,counter\n2026-09-29 10:00:00,z1,120,2,c1\n"
                                      "bad,z2,1,1,c1\n")
    assert import_pos_csv(db, "S3", folder) == 1
    assert (folder / "bills.imported").exists()


def test_api_key_protects_writes(cfg, db, monkeypatch):
    monkeypatch.setenv("RETAIL_API_KEY", "secret")
    c = TestClient(create_app(cfg, db))
    assert c.post("/api/pos/transactions", json=[]).status_code == 401
    assert c.post("/api/pos/transactions", json=[], headers={"X-API-Key": "secret"}).status_code == 200


def test_sync_store_and_forward_to_hq(cfg, db, tmp_path, monkeypatch):
    """Edge pushes to HQ; while HQ is down nothing is lost; afterwards everything arrives once."""
    _seed(db, "EDGE-1")
    hq_db = EdgeDB(tmp_path / "hq.db")
    hq = TestClient(create_app(cfg, hq_db))
    online = {"up": False}

    class Resp:
        def __init__(self, r):
            self.r = r

        def raise_for_status(self):
            if self.r.status_code >= 400:
                raise RuntimeError(self.r.status_code)

    def fake_post(url, data=None, timeout=None, headers=None):
        if not online["up"]:
            raise ConnectionError("network down")
        return Resp(hq.post("/api/ingest", content=data, headers=headers))

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    agent = SyncAgent(db, "EDGE-1", "http://hq.example", batch_size=1000)
    with pytest.raises(ConnectionError):
        agent.sync_once()
    assert len(db.unsynced("metrics", 100)) == 7  # still queued locally
    online["up"] = True
    assert agent.sync_once() > 0
    assert db.unsynced("metrics", 100) == []
    assert agent.sync_once() == 0  # idempotent: nothing re-sent
    stores = hq.get("/api/stores").json()
    assert stores[0]["store_id"] == "EDGE-1" and stores[0]["today"]["footfall_in"] == 15
    hq_db.close()


def test_replenishment_order_uses_master_data():
    order = build_replenishment_order("S", [{"section": "cam/shampoo", "status": "LOW_STOCK", "facings": 10,
                                             "capacity_estimate": 30}],
                                      {"shampoo": {"erp_code": "SKU-1", "supplier": "ACME", "reorder_qty": "24"}})
    line = order["lines"][0]
    assert line["erp_code"] == "SKU-1" and line["suggested_qty"] == 24 and line["priority"] == "NORMAL"


def test_config_validation_rejects_bad_role(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("store: {store_id: X}\ncameras:\n  - {id: a, role: teleport, source: x}\n")
    with pytest.raises(Exception):
        load_config(p)


def test_hourly_buckets_follow_local_clock(db):
    # 10:15 and 10:45 IST (UTC+5:30) must land in the same local 10:00 bucket
    ist = 19800
    base = 1_790_000_000 - (1_790_000_000 + ist) % 86400  # local midnight (IST) as epoch
    t1015, t1045 = base + 10 * 3600 + 900, base + 10 * 3600 + 2700
    db.add_metrics([(t1015, "S", "c", "entries", "", 1.0), (t1045, "S", "c", "entries", "", 1.0)])
    rows = db.metric_series("S", "entries", base, base + 86400, 3600, tz_offset_s=ist)
    assert len(rows) == 1 and rows[0]["value"] == 2.0
    assert rows[0]["bucket"] == base + 10 * 3600


def test_forecaster_delta_mode(tmp_path):
    import joblib
    import pandas as pd
    from retail_ai.analytics.forecasting import QueueForecaster, build_features

    from sklearn.dummy import DummyRegressor

    hist = pd.DataFrame({"ts": pd.date_range("2026-09-29 10:00", periods=30, freq="min"), "entries": 3,
                         "checkout_arrivals": 2, "queue_len": 5.0, "counters_open": 2})
    feats = list(build_features(hist).columns)
    const = DummyRegressor(strategy="constant", constant=2.0).fit(build_features(hist)[feats].fillna(0), [2.0] * 30)
    joblib.dump({"model": const, "features": feats, "horizon": 15, "target_mode": "delta"}, tmp_path / "m.joblib")
    f = QueueForecaster(tmp_path / "m.joblib")
    assert f.ready and f.predict_from_history(hist) == 7.0  # current 5 + predicted change 2


def test_mark_shelf_as_full_sets_baseline(cfg, db):
    _seed(db)
    c = TestClient(create_app(cfg, db))
    r = c.post("/api/stores/S1/inventory/baseline?key=cam_shelf_A/shampoo")
    assert r.status_code == 200 and r.json()["baseline"]["facings"] == 5
    assert c.get("/api/stores/S1/inventory/baselines").json()["cam_shelf_A/shampoo"]["facings"] == 5
    assert c.post("/api/stores/S1/inventory/baseline?key=nope").status_code == 404


def test_latest_day_and_daily_kpis(cfg, db):
    _seed(db)
    c = TestClient(create_app(cfg, db))
    day = c.get("/api/stores/S1/latest_day").json()["day"]
    assert day is not None
    rows = c.get(f"/api/stores/S1/daily_kpis?days=3&end_day={day}").json()
    assert len(rows) == 3 and rows[-1]["footfall"] == 15 and rows[-1]["bills"] == 2
    assert c.get("/api/stores/NOPE/latest_day").json()["day"] is None
