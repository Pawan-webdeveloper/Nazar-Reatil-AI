"""Queue congestion forecasting + staffing recommendation.

Two complementary pieces:

1. `QueueForecaster` - a trained ML model (LightGBM by default, chosen by
   training/train_queue_forecaster.py) that predicts the total checkout queue
   length `horizon` minutes ahead from minute-level history. Key leading signal:
   entrance footfall 10-30 minutes ago (shoppers reach the checkout after shopping).

2. Queueing theory (Erlang-C, M/M/c) - converts a predicted arrival rate into an
   expected waiting time for c open counters and recommends the minimum number of
   counters that keeps the expected wait under the target.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = []  # filled by build_features()

LAGS_QUEUE = [0, 1, 2, 5, 10, 15]
LAGS_ENTRY_SUM = [(0, 5), (5, 15), (15, 30), (30, 45)]
LAGS_ARRIVAL_SUM = [(0, 5), (5, 15)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from a minute-indexed frame.

    Required columns: ts (datetime64), entries, checkout_arrivals, queue_len, counters_open.
    Optional: is_holiday. Only past/present information is used (no leakage).
    """
    d = df.sort_values("ts").reset_index(drop=True)
    f = pd.DataFrame(index=d.index)
    for lag in LAGS_QUEUE:
        f[f"queue_lag{lag}"] = d["queue_len"].shift(lag)
    for a, b in LAGS_ENTRY_SUM:
        f[f"entries_{a}_{b}"] = d["entries"].shift(a).rolling(b - a, min_periods=1).sum()
    for a, b in LAGS_ARRIVAL_SUM:
        f[f"arrivals_{a}_{b}"] = d["checkout_arrivals"].shift(a).rolling(b - a, min_periods=1).sum()
    f["queue_trend_5"] = d["queue_len"] - d["queue_len"].shift(5)
    f["counters_open"] = d["counters_open"]
    f["queue_per_counter"] = d["queue_len"] / d["counters_open"].clip(lower=1)
    minute = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    f["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    f["hour"] = d["ts"].dt.hour
    f["dow"] = d["ts"].dt.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["is_holiday"] = d["is_holiday"].astype(int) if "is_holiday" in d else 0
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = list(f.columns)
    return f


SCALED_PREFIXES = ("queue_lag", "entries_", "arrivals_", "queue_trend")


def per_counter(features: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features: divide volume features by open counters so one model
    transfers between a 1-counter kirana and a 10-counter hypermarket."""
    f = features.copy()
    c = f["counters_open"].clip(lower=1)
    for col in f.columns:
        if col.startswith(SCALED_PREFIXES):
            f[col] = f[col] / c
    return f


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    d = df.sort_values("ts").reset_index(drop=True)
    return d["queue_len"].shift(-horizon)


# ---------------------------------------------------------------- Erlang C
def erlang_c_prob_wait(lam: float, mu: float, c: int) -> float:
    """Probability an arriving customer must wait (M/M/c). lam, mu per minute."""
    if c <= 0:
        return 1.0
    a = lam / mu  # offered load (Erlangs)
    rho = a / c
    if rho >= 1:
        return 1.0
    # numerically stable iterative computation of Erlang-B then convert to Erlang-C
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b / (1 - rho + rho * b)


def expected_wait_minutes(lam: float, mu: float, c: int) -> float:
    if lam <= 0:
        return 0.0
    if c <= 0 or lam / (c * mu) >= 1:
        return math.inf
    return erlang_c_prob_wait(lam, mu, c) / (c * mu - lam)


def recommend_counters(lam_per_min: float, avg_service_min: float, target_wait_min: float,
                       max_counters: int, min_counters: int = 1) -> Dict:
    mu = 1.0 / max(avg_service_min, 1e-6)
    for c in range(max(1, min_counters), max_counters + 1):
        w = expected_wait_minutes(lam_per_min, mu, c)
        if w <= target_wait_min:
            return {"recommended_counters": c, "expected_wait_min": round(w, 2),
                    "utilization": round(lam_per_min / (c * mu), 3)}
    w = expected_wait_minutes(lam_per_min, mu, max_counters)
    return {"recommended_counters": max_counters, "expected_wait_min": (round(w, 2) if math.isfinite(w) else None),
            "utilization": round(lam_per_min / (max_counters * mu), 3), "note": "demand exceeds capacity"}


# ---------------------------------------------------------------- model wrapper
class QueueForecaster:
    def __init__(self, path: Optional[str | Path] = None):
        self.bundle = None
        if path and Path(path).exists():
            import joblib
            self.bundle = joblib.load(path)

    @property
    def ready(self) -> bool:
        return self.bundle is not None

    @property
    def horizon(self) -> int:
        return int(self.bundle["horizon"]) if self.bundle else 15

    def predict_from_history(self, history: pd.DataFrame) -> Optional[float]:
        """history: minute rows (>= 45 rows recommended) with the columns of build_features()."""
        if not self.ready or len(history) < 2:
            return None
        raw = build_features(history).iloc[[-1]].fillna(0.0)
        mode = self.bundle.get("target_mode", "level")
        X = per_counter(raw) if mode == "per_counter_delta" else raw
        pred = float(self.bundle["model"].predict(X[self.bundle["features"]])[0])
        q0 = float(raw["queue_lag0"].iloc[0])
        if mode == "delta":
            pred += q0  # model learned the change vs. the current queue
        elif mode == "per_counter_delta":
            pred = pred * max(1.0, float(raw["counters_open"].iloc[0])) + q0
        return max(0.0, pred)
