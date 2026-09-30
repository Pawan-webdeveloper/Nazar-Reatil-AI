"""POS / inventory / ERP connectors.

Most Indian POS systems (Tally, Marg, GoFrugal, Ginesys, custom billing) can export
bills as CSV or call a webhook. Both are supported:

* `import_pos_csv(folder)`   - watches a folder for CSV exports
      columns: timestamp, bill_id, amount, items, counter
* `POST /api/pos/transactions` (see api/server.py) - real-time push, same fields
* `load_inventory_master(csv)` - sku_group -> reorder info used to enrich shelf alerts
      columns: sku_group, section, supplier, reorder_qty, erp_code
* `build_replenishment_order(...)` - converts shelf alerts into an ERP-ready purchase /
      replenishment request (JSON) that can be POSTed to SAP / Oracle / Odoo / Zoho APIs.

Combining POS bills with camera footfall gives the conversion rate (bills / entries).
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..storage.db import EdgeDB

log = logging.getLogger("retail_ai.pos")


def _parse_ts(v: str) -> float:
    v = v.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(v, fmt).timestamp()
        except ValueError:
            continue
    return float(v)  # unix seconds


def import_pos_csv(db: EdgeDB, store_id: str, folder: str | Path) -> int:
    folder = Path(folder)
    if not folder.exists():
        return 0
    total = 0
    for f in sorted(folder.glob("*.csv")):
        rows = []
        with f.open(newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    rows.append((str(r["bill_id"]), _parse_ts(r["timestamp"]), store_id, float(r["amount"]),
                                 int(float(r.get("items", 0) or 0)), str(r.get("counter", ""))))
                except (KeyError, ValueError) as exc:
                    log.warning("skip bad POS row in %s: %s", f.name, exc)
        total += db.upsert_pos(rows)
        f.rename(f.with_suffix(".imported"))
    return total


def load_inventory_master(path: str | Path) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8-sig") as fh:
        return {r["section"]: r for r in csv.DictReader(fh)}


def build_replenishment_order(store_id: str, shelf_reports: List[dict], master: Dict[str, dict]) -> dict:
    lines = []
    for rep in shelf_reports:
        if rep.get("status") not in ("LOW_STOCK", "OUT_OF_STOCK"):
            continue
        sec = rep["section"].split("/")[-1]
        info = master.get(sec, {})
        missing = max(0, int(rep.get("capacity_estimate", 0)) - int(rep.get("facings", 0)))
        lines.append({"section": sec, "erp_code": info.get("erp_code", ""), "supplier": info.get("supplier", ""),
                      "priority": "HIGH" if rep["status"] == "OUT_OF_STOCK" else "NORMAL",
                      "facings_missing": missing,
                      "suggested_qty": max(missing, int(info.get("reorder_qty", 0) or 0))})
    return {"store_id": store_id, "created": datetime.now().isoformat(timespec="seconds"),
            "type": "shelf_replenishment", "lines": lines}
