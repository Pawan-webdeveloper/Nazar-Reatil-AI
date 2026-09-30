"""Discrete-event simulator of store traffic and checkout queues.

Why: there is no public minute-level dataset of Indian retail footfall + checkout
queues. The simulator encodes well-known retail dynamics so the forecasting method
can be developed and validated end-to-end; in production the SAME features are logged
by the edge node, and `training/train_queue_forecaster.py --from-db` retrains on the
store's own data after 2-4 weeks.

Dynamics modelled
  * time-of-day profile with late-morning and evening peaks
  * weekday / weekend effect, salary-week (1st-5th of month) uplift, festival days
  * day-level random demand shocks (weather, local events)
  * shopping duration ~ log-normal (median ~18 min) -> checkout arrival lag
  * conversion probability (visitors who buy)
  * multi-server FIFO checkout with gamma service times and a fixed, imperfect
    staffing roster (so real congestion episodes occur)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class StoreProfile:
    name: str
    base_entries_per_min: float      # average entry rate at the busiest hour on a weekday
    conversion: float = 0.72
    shop_median_min: float = 18.0
    shop_sigma: float = 0.5
    service_mean_min: float = 2.2
    roster: tuple = ((8, 11, 2), (11, 17, 3), (17, 22, 4))  # (from_hour, to_hour, counters)
    weekend_extra_counters: int = 1
    open_hour: int = 8
    close_hour: int = 22


PROFILES = {
    "neighbourhood": StoreProfile("neighbourhood", 0.9, conversion=0.8, shop_median_min=9, service_mean_min=1.6,
                                  roster=((8, 12, 1), (12, 17, 1), (17, 22, 2)), weekend_extra_counters=0),
    "supermarket": StoreProfile("supermarket", 2.2),
    "hypermarket": StoreProfile("hypermarket", 3.8, shop_median_min=26, service_mean_min=2.6,
                                roster=((8, 11, 5), (11, 17, 7), (17, 22, 9)), weekend_extra_counters=2),
}

FESTIVALS_2026 = ["2026-01-14", "2026-03-04", "2026-03-20", "2026-08-15", "2026-08-28", "2026-09-14",
                  "2026-10-20", "2026-11-08", "2026-11-09", "2026-11-10", "2026-12-25"]


def _daily_profile(minutes: np.ndarray) -> np.ndarray:
    h = minutes / 60.0
    morning = 0.55 * np.exp(-0.5 * ((h - 11.0) / 1.4) ** 2)
    evening = 1.00 * np.exp(-0.5 * ((h - 19.0) / 1.6) ** 2)
    return 0.18 + morning + evening


def simulate_store(profile: StoreProfile, start: str = "2026-06-01", days: int = 84, seed: int = 0,
                   dt_min: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    festivals = set(pd.to_datetime(FESTIVALS_2026).date)
    for d in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        open_m, close_m = profile.open_hour * 60, profile.close_hour * 60
        mins = np.arange(open_m, close_m)
        mult = 1.0
        dow = day.dayofweek
        if dow == 5:
            mult *= 1.30
        elif dow == 6:
            mult *= 1.45
        if day.day <= 5:
            mult *= 1.18  # salary week
        is_holiday = day.date() in festivals
        if is_holiday:
            mult *= rng.uniform(1.6, 2.0)
        mult *= rng.lognormal(0, 0.12)  # day shock
        lam = profile.base_entries_per_min * mult * _daily_profile(mins) / 1.18
        lam[mins >= close_m - 15] = 0.0  # doors close for new customers 15 min before closing
        # intra-day autocorrelated noise (bursts: buses, school pick-up, rain)
        noise = np.exp(np.convolve(rng.normal(0, 0.25, len(mins)), np.ones(20) / 20 ** 0.5, mode="same") * 0.35)
        entries = rng.poisson(lam * noise)

        # checkout arrival times (continuous, minutes of day)
        arr = []
        for m, n in zip(mins, entries):
            for _ in range(n):
                if rng.random() < profile.conversion:
                    arr.append(m + rng.random() + rng.lognormal(np.log(profile.shop_median_min), profile.shop_sigma))
        arr = np.sort(np.array(arr))

        def counters_open(minute: float) -> int:
            hour = minute / 60.0
            for a, b, c in profile.roster:
                if a <= hour < b:
                    return c + (profile.weekend_extra_counters if dow >= 5 else 0)
            return profile.roster[-1][2]

        # multi-server FIFO simulation, time step dt
        queue: list = []
        ai = 0
        n_steps = int((close_m + 60 - open_m) / dt_min)
        q_samples = np.zeros(len(mins))
        q_counts = np.zeros(len(mins))
        waits_sum = np.zeros(len(mins))
        waits_n = np.zeros(len(mins))
        arrivals_per_min = np.zeros(len(mins))
        counters_per_min = np.array([counters_open(m) for m in mins])
        max_c = max(c for _a, _b, c in profile.roster) + profile.weekend_extra_counters
        busy_until = [0.0] * max_c
        for s in range(n_steps):
            t = open_m + s * dt_min
            while ai < len(arr) and arr[ai] <= t:
                queue.append(arr[ai])
                mi = int(arr[ai]) - open_m
                if 0 <= mi < len(mins):
                    arrivals_per_min[mi] += 1
                ai += 1
            c_open = counters_open(min(t, close_m - 1))
            for k in range(c_open):
                if busy_until[k] <= t and queue:
                    a_t = queue.pop(0)
                    busy_until[k] = t + rng.gamma(2.0, profile.service_mean_min / 2.0)
                    mi = int(t) - open_m
                    if 0 <= mi < len(mins):
                        waits_sum[mi] += t - a_t
                        waits_n[mi] += 1
            mi = int(t) - open_m
            if 0 <= mi < len(mins):
                q_samples[mi] += len(queue)
                q_counts[mi] += 1
        queue_len = q_samples / np.maximum(q_counts, 1)
        for i, m in enumerate(mins):
            rows.append({
                "ts": day + pd.Timedelta(minutes=int(m)),
                "store_type": profile.name,
                "entries": int(entries[i]),
                "checkout_arrivals": int(arrivals_per_min[i]),
                "queue_len": float(round(queue_len[i], 3)),
                "counters_open": int(counters_per_min[i]),
                "avg_wait_min": float(waits_sum[i] / waits_n[i]) if waits_n[i] else np.nan,
                "is_holiday": int(is_holiday),
            })
    return pd.DataFrame(rows)


def simulate_all(days: int = 84, start: str = "2026-06-01", seed: int = 0,
                 profiles: Optional[list] = None) -> pd.DataFrame:
    frames = []
    for i, name in enumerate(profiles or list(PROFILES)):
        frames.append(simulate_store(PROFILES[name], start, days, seed + 17 * i))
    return pd.concat(frames, ignore_index=True)
