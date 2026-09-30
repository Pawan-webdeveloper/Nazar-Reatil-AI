"""Offline-first local storage (SQLite, WAL mode).

Only aggregated, anonymous analytics are stored - never frames or images.
Every row has a `synced` flag so the SyncAgent can forward data to HQ when the
network is available (store-and-forward), and the store keeps working offline.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,              -- unix seconds, start of the minute bucket
    store_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    metric TEXT NOT NULL,          -- entries, exits, zone_visits, dwell_s, queue_len, ...
    key TEXT NOT NULL DEFAULT '',  -- zone / counter / section name
    value REAL NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_metrics_ts ON metrics(store_id, metric, ts);
CREATE INDEX IF NOT EXISTS ix_metrics_sync ON metrics(synced);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    store_id TEXT NOT NULL,
    level TEXT NOT NULL,           -- critical | warning | info
    type TEXT NOT NULL,            -- OUT_OF_STOCK, LOW_STOCK, QUEUE_BUILDUP, QUEUE_FORECAST, ...
    key TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    acknowledged INTEGER NOT NULL DEFAULT 0,
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_alerts_ts ON alerts(store_id, ts);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    store_id TEXT NOT NULL,
    kind TEXT NOT NULL,            -- shelf_report, queue_status, heatmap, zone_summary
    key TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL,
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_snap ON snapshots(store_id, kind, key, ts);

CREATE TABLE IF NOT EXISTS pos_transactions (
    bill_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    store_id TEXT NOT NULL,
    amount REAL NOT NULL,
    items INTEGER NOT NULL DEFAULT 0,
    counter TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_pos_ts ON pos_transactions(store_id, ts);

CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    last_seen REAL NOT NULL DEFAULT 0
);
"""


class EdgeDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ writes
    def add_metrics(self, rows: Iterable[tuple]) -> None:
        """rows: (ts, store_id, camera_id, metric, key, value)"""
        rows = list(rows)
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO metrics(ts,store_id,camera_id,metric,key,value) VALUES (?,?,?,?,?,?)", rows)
            self._conn.commit()

    def add_alert(self, ts: float, store_id: str, level: str, type_: str, key: str, message: str,
                  payload: Optional[dict] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO alerts(ts,store_id,level,type,key,message,payload) VALUES (?,?,?,?,?,?,?)",
                (ts, store_id, level, type_, key, message, json.dumps(payload or {})))
            self._conn.commit()
            return int(cur.lastrowid)

    def add_snapshot(self, ts: float, store_id: str, kind: str, key: str, payload: Any) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO snapshots(ts,store_id,kind,key,payload) VALUES (?,?,?,?,?)",
                               (ts, store_id, kind, key, json.dumps(payload)))
            self._conn.commit()

    def upsert_pos(self, rows: Iterable[tuple]) -> int:
        """rows: (bill_id, ts, store_id, amount, items, counter)"""
        rows = list(rows)
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO pos_transactions(bill_id,ts,store_id,amount,items,counter) VALUES (?,?,?,?,?,?)",
                rows)
            self._conn.commit()
        return len(rows)

    def touch_store(self, store_id: str, name: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO stores(store_id,name,last_seen) VALUES (?,?,?) "
                "ON CONFLICT(store_id) DO UPDATE SET last_seen=excluded.last_seen, "
                "name=CASE WHEN excluded.name!='' THEN excluded.name ELSE stores.name END",
                (store_id, name, time.time()))
            self._conn.commit()

    def acknowledge_alert(self, alert_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------ reads
    def query(self, sql: str, params: tuple = ()) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def metric_series(self, store_id: str, metric: str, t0: float, t1: float, bucket_s: int = 3600,
                      key: Optional[str] = None, agg: str = "SUM", tz_offset_s: Optional[int] = None) -> List[Dict]:
        """Time-bucketed series. Buckets are aligned to LOCAL time (e.g. IST is UTC+05:30),
        so an hourly bucket really is 10:00-11:00 on the store clock."""
        agg = agg.upper()
        if agg not in {"SUM", "AVG", "MAX", "MIN"}:
            raise ValueError("bad aggregation")
        off = int(time.localtime().tm_gmtoff if tz_offset_s is None else tz_offset_s)
        b = int(bucket_s)
        sql = (f"SELECT CAST((ts+{off})/{b} AS INTEGER)*{b}-{off} AS bucket, key, {agg}(value) AS value "
               "FROM metrics WHERE store_id=? AND metric=? AND ts>=? AND ts<?")
        params: list = [store_id, metric, t0, t1]
        if key is not None:
            sql += " AND key=?"
            params.append(key)
        sql += " GROUP BY bucket, key ORDER BY bucket"
        return self.query(sql, tuple(params))

    def latest_snapshots(self, store_id: str, kind: str) -> List[Dict]:
        rows = self.query(
            "SELECT s.* FROM snapshots s JOIN (SELECT key, MAX(ts) AS mts FROM snapshots "
            "WHERE store_id=? AND kind=? GROUP BY key) m ON s.key=m.key AND s.ts=m.mts "
            "WHERE s.store_id=? AND s.kind=?", (store_id, kind, store_id, kind))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    # ------------------------------------------------------------------ sync
    def unsynced(self, table: str, limit: int) -> List[Dict]:
        if table not in {"metrics", "alerts", "snapshots"}:
            raise ValueError(table)
        return self.query(f"SELECT * FROM {table} WHERE synced=0 ORDER BY id LIMIT ?", (limit,))

    def mark_synced(self, table: str, ids: List[int]) -> None:
        if table not in {"metrics", "alerts", "snapshots"} or not ids:
            return
        with self._lock:
            self._conn.executemany(f"UPDATE {table} SET synced=1 WHERE id=?", [(i,) for i in ids])
            self._conn.commit()

    def purge_older_than(self, days: int) -> None:
        """Data retention: delete detailed rows older than N days (aggregates stay at HQ)."""
        cutoff = time.time() - days * 86400
        with self._lock:
            for t in ("metrics", "snapshots"):
                self._conn.execute(f"DELETE FROM {t} WHERE ts<? AND synced=1", (cutoff,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
