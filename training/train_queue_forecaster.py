"""Train and compare checkout-queue forecasting models (predict queue length H minutes ahead).

Data
  default : simulated 12 weeks x 3 store formats (see retail_ai/simulation.py)
  --from-db: minute metrics logged by a real edge node (retrain on the store's own data)

Protocol (time-based, no leakage)
  weeks 1-8  train | weeks 9-10 validation (model selection) | weeks 11-12 test (reported once)
  Also a "new store" check: train on 2 store formats, test on the unseen third format.

Candidates
  persistence (queue now), seasonal naive (same minute last week), moving average,
  Ridge regression, Random Forest, LightGBM

Metrics
  MAE / RMSE / R^2 on queue length, congestion early-warning precision / recall / F1
  (queue per open counter > threshold at t+H), and error slices (hour band, weekend,
  holiday, store format) to check that errors are not concentrated in one segment (bias).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retail_ai.analytics.forecasting import build_features, make_target, per_counter  # noqa: E402
from retail_ai.simulation import simulate_all  # noqa: E402

OUT = ROOT / "outputs" / "queue_forecaster"


def prepare(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    parts = []
    for (store, day), g in df.groupby(["store_type", df["ts"].dt.date]):
        g = g.sort_values("ts").reset_index(drop=True)
        f = build_features(g)
        f["target"] = make_target(g, horizon)
        f["ts"] = g["ts"]
        f["store_type"] = store
        f["queue_now"] = g["queue_len"]
        f["counters_open_future"] = g["counters_open"].shift(-horizon)
        parts.append(f)
    data = pd.concat(parts, ignore_index=True)
    data = data.dropna(subset=["target"]).reset_index(drop=True)
    # seasonal naive: forecast for t+H = observed queue at (t+H) one week earlier
    lookup = pd.DataFrame({"store_type": data["store_type"],
                           "ts": data["ts"] + pd.Timedelta(days=7) - pd.Timedelta(minutes=horizon),
                           "seasonal_naive": data["queue_now"]})
    data = data.merge(lookup, on=["store_type", "ts"], how="left")
    return data


def congestion_scores(y_true, y_pred, counters, thr_per_counter: float):
    t = y_true > thr_per_counter * counters
    p = y_pred > thr_per_counter * counters
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "events": int(t.sum())}


def regression_scores(y, p):
    err = p - y
    ss_res = float((err ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"MAE": round(float(np.abs(err).mean()), 3), "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
            "R2": round(1 - ss_res / ss_tot, 4) if ss_tot else None}


def fit_models(Xtr, ytr, seed=0):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=4, random_state=seed),
        "lightgbm": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                  random_state=seed, verbose=-1, n_jobs=4),
    }
    fitted = {}
    for name, m in models.items():
        t = time.time()
        m.fit(Xtr, ytr)
        fitted[name] = m
        print(f"  fitted {name} in {time.time() - t:.1f}s", flush=True)
    return fitted


def predict_all(fitted, X, data, to_queue):
    preds = {"persistence": data["queue_now"].to_numpy(),
             "seasonal_naive": data["seasonal_naive"].fillna(data["queue_now"]).to_numpy(),
             "moving_avg_15": data["ma15"].to_numpy()}
    for name, m in fitted.items():
        preds[name] = np.clip(to_queue(m.predict(X)), 0, None)
    return preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--days", type=int, default=259, help="37 weeks: festivals fall in train, validation AND test")
    ap.add_argument("--start", default="2026-01-05")
    ap.add_argument("--target-mode", choices=["per_counter_delta", "delta", "level"], default="per_counter_delta",
                    help="delta = learn the change vs. the current queue (extrapolates to unseen peak levels); "
                         "per_counter_delta = same, with volume features/target divided by open counters "
                         "(transfers to new store formats, see outputs/queue_forecaster/loo_experiment.json)")
    ap.add_argument("--threshold-per-counter", type=float, default=4.0)
    ap.add_argument("--from-db", default="", help="path to an edge SQLite db to retrain on real data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-sim", action="store_true", help="reuse outputs/queue_forecaster/simulated_minutes.csv.gz")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if a.from_db:
        from retail_ai.integrations.history import minute_history_from_db
        df = minute_history_from_db(a.from_db)
    elif a.reuse_sim and (OUT / "simulated_minutes.csv.gz").exists():
        df = pd.read_csv(OUT / "simulated_minutes.csv.gz", parse_dates=["ts"])
    else:
        df = simulate_all(days=a.days, start=a.start, seed=a.seed)
        df.to_csv(OUT / "simulated_minutes.csv.gz", index=False, compression="gzip")
    print(f"data: {len(df):,} minute rows, {df['store_type'].nunique()} store formats ({time.time() - t0:.0f}s)")

    data = prepare(df, a.horizon)
    feats = [c for c in data.columns if c not in {"target", "fit_target", "ts", "store_type", "queue_now",
                                                  "seasonal_naive", "counters_open_future"}]
    data[feats] = data[feats].fillna(0.0)
    start = data["ts"].min().normalize()
    week = ((data["ts"] - start).dt.days // 7)
    n_weeks = int(week.max()) + 1
    tr, va, te = week < n_weeks - 4, (week >= n_weeks - 4) & (week < n_weeks - 2), week >= n_weeks - 2
    print(f"split rows: train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")

    counters = data["counters_open"].clip(lower=1)
    data["ma15"] = data[["queue_lag0", "queue_lag1", "queue_lag2", "queue_lag5", "queue_lag10", "queue_lag15"]].mean(axis=1)
    if a.target_mode == "per_counter_delta":
        data[feats] = per_counter(data[feats])
        data["fit_target"] = (data["target"] - data["queue_now"]) / counters
    elif a.target_mode == "delta":
        data["fit_target"] = data["target"] - data["queue_now"]
    else:
        data["fit_target"] = data["target"]

    def to_queue(pred, rows):
        """Convert raw model output back to 'people waiting in H minutes'."""
        if a.target_mode == "per_counter_delta":
            return pred * counters[rows].to_numpy() + data.loc[rows, "queue_now"].to_numpy()
        if a.target_mode == "delta":
            return pred + data.loc[rows, "queue_now"].to_numpy()
        return pred
    fitted = fit_models(data.loc[tr, feats], data.loc[tr, "fit_target"], a.seed)
    results = {"horizon_min": a.horizon, "features": feats, "splits": {}}
    for split_name, mask in (("validation", va), ("test", te)):
        d = data[mask]
        preds = predict_all(fitted, d[feats], d, lambda p, m=mask: to_queue(p, m))
        res = {}
        for name, p in preds.items():
            res[name] = {**regression_scores(d["target"].to_numpy(), p),
                         "congestion": congestion_scores(d["target"].to_numpy(), p,
                                                         d["counters_open_future"].fillna(d["counters_open"]).to_numpy(),
                                                         a.threshold_per_counter)}
        results["splits"][split_name] = res

    val = results["splits"]["validation"]
    best = min((k for k in fitted), key=lambda k: val[k]["MAE"])
    results["selected_model"] = best
    print("\nVALIDATION  (model selection)")
    for k, v in val.items():
        print(f"  {k:15s} MAE {v['MAE']:.3f}  RMSE {v['RMSE']:.3f}  R2 {v['R2']}  congestion-F1 {v['congestion']['f1']}")
    print(f"selected: {best}")
    print("\nTEST  (held-out last 2 weeks)")
    for k, v in results["splits"]["test"].items():
        print(f"  {k:15s} MAE {v['MAE']:.3f}  RMSE {v['RMSE']:.3f}  R2 {v['R2']}  congestion {v['congestion']}")

    # ---- bias / robustness slices on the test period for the selected model
    d = data[te].copy()
    d["pred"] = np.clip(to_queue(fitted[best].predict(d[feats]), te), 0, None)
    d["abs_err"] = (d["pred"] - d["target"]).abs()
    d["hour_band"] = pd.cut(d["ts"].dt.hour, [0, 11, 16, 20, 24], right=False,
                            labels=["08-11", "11-16", "16-20", "20-22"])
    slices = {}
    for col in ("store_type", "hour_band", "is_weekend", "is_holiday"):
        g = d.groupby(col, observed=True).agg(MAE=("abs_err", "mean"), mean_queue=("target", "mean"), n=("target", "size"))
        g["MAE_per_mean_queue"] = g["MAE"] / g["mean_queue"].clip(lower=0.1)
        slices[col] = {str(k): {kk: round(float(vv), 3) for kk, vv in row.items()} for k, row in g.iterrows()}
    results["test_slices_selected_model"] = slices
    print("\nTEST error slices:", json.dumps(slices, indent=1))

    # ---- unseen store format generalisation (leave-one-format-out)
    loo = {}
    for held in sorted(data["store_type"].unique()):
        m_tr = tr & (data["store_type"] != held)
        m_te = te & (data["store_type"] == held)
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=a.seed, verbose=-1, n_jobs=4)
        m.fit(data.loc[m_tr, feats], data.loc[m_tr, "fit_target"])
        p = np.clip(to_queue(m.predict(data.loc[m_te, feats]), m_te), 0, None)
        y = data.loc[m_te, "target"].to_numpy()
        loo[held] = {**regression_scores(y, p), "persistence_MAE": round(float(np.abs(data.loc[m_te, "queue_now"] - y).mean()), 3)}
    results["unseen_store_format"] = loo
    print("\nUnseen store format (LightGBM trained without it):", json.dumps(loo, indent=1))

    # ---- refit selected model on train+val and save
    final = fit_models(data.loc[tr | va, feats], data.loc[tr | va, "fit_target"], a.seed)[best]
    bundle = {"model": final, "features": feats, "horizon": a.horizon, "name": best, "target_mode": a.target_mode,
              "trained_rows": int((tr | va).sum()), "created": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "queue_forecaster.joblib")
    if hasattr(final, "feature_importances_"):
        imp = sorted(zip(feats, final.feature_importances_.tolist()), key=lambda x: -x[1])[:12]
        results["top_features"] = [[f, int(v)] for f, v in imp]
    (OUT / "queue_forecaster_results.json").write_text(json.dumps(results, indent=2, default=str))
    _plot(d, best, a.horizon)
    print(f"\nsaved models/queue_forecaster.joblib and {OUT / 'queue_forecaster_results.json'}")
    return 0


def _plot(d: pd.DataFrame, best: str, horizon: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    for ax, store in zip(axes, sorted(d["store_type"].unique())):
        g = d[d["store_type"] == store]
        day = g["ts"].dt.date.unique()[3]
        g = g[g["ts"].dt.date == day]
        tt = g["ts"] + pd.Timedelta(minutes=horizon)
        ax.plot(tt, g["target"], label="actual queue", color="#1f77b4")
        ax.plot(tt, g["pred"], label=f"{best} forecast ({horizon} min ahead)", color="#d62728", alpha=0.8)
        ax.plot(tt, g["queue_now"], label="persistence baseline", color="#999999", alpha=0.6, lw=1)
        ax.set_title(f"{store} - {day}")
        ax.set_ylabel("people waiting")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_example.png", dpi=110)


if __name__ == "__main__":
    sys.exit(main())
