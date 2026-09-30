"""Turn the edge node's minute metrics into the training table used by the forecaster."""
from __future__ import annotations

import pandas as pd

from ..storage.db import EdgeDB


def minute_history_from_db(db_path: str, store_id: str | None = None, tz: str = "Asia/Kolkata") -> pd.DataFrame:
    db = EdgeDB(db_path)
    where = "WHERE metric IN ('entries','checkout_arrivals','queue_len','counters_open')"
    params: tuple = ()
    if store_id:
        where += " AND store_id=?"
        params = (store_id,)
    rows = db.query(f"SELECT store_id, ts, metric, SUM(value) AS value FROM metrics {where} "
                    "GROUP BY store_id, ts, metric", params)
    db.close()
    if not rows:
        raise ValueError("no minute metrics in database yet - run the edge node for a few weeks first")
    df = pd.DataFrame(rows).pivot_table(index=["store_id", "ts"], columns="metric", values="value",
                                        aggfunc="sum").reset_index()
    for col in ("entries", "checkout_arrivals", "queue_len"):
        if col not in df:
            df[col] = 0.0
    if "counters_open" not in df:
        df["counters_open"] = 1
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df = df.rename(columns={"store_id": "store_type"}).fillna(0.0)
    df["is_holiday"] = 0
    return df
