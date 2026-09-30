"""Experiment: do per-counter (scale-free) features generalise better to an unseen store format?"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from lightgbm import LGBMRegressor
from training.train_queue_forecaster import prepare

df = pd.read_csv("outputs/queue_forecaster/simulated_minutes.csv.gz", parse_dates=["ts"])
data = prepare(df, 15)
drop = {"target", "ts", "store_type", "queue_now", "seasonal_naive", "counters_open_future"}
feats = [c for c in data.columns if c not in drop]
data[feats] = data[feats].fillna(0.0)
c = data["counters_open"].clip(lower=1)
sf = data[feats].copy()
for col in feats:
    if col.startswith(("queue_lag", "entries_", "arrivals_", "queue_trend")):
        sf[col] = data[col] / c
week = (data["ts"] - data["ts"].min().normalize()).dt.days // 7
tr, te = week < week.max() - 3, week >= week.max() - 1
res = {}
for held in sorted(data["store_type"].unique()):
    mtr, mte = tr & (data.store_type != held), te & (data.store_type == held)
    y = data.loc[mte, "target"].to_numpy(); q0 = data.loc[mte, "queue_now"].to_numpy()
    out = {"persistence_MAE": round(float(np.abs(q0 - y).mean()), 3)}
    for name, X, tgt, back in [("absolute_delta", data[feats], data["target"] - data["queue_now"], lambda p, m: p + data.loc[m, "queue_now"].to_numpy()),
                               ("per_counter_delta", sf, (data["target"] - data["queue_now"]) / c, lambda p, m: p * c[m].to_numpy() + data.loc[m, "queue_now"].to_numpy())]:
        m = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=4)
        m.fit(X[mtr], tgt[mtr])
        p = np.clip(back(m.predict(X[mte]), mte), 0, None)
        out[name + "_MAE"] = round(float(np.abs(p - y).mean()), 3)
    res[held] = out; print(held, out, flush=True)
json.dump(res, open("outputs/queue_forecaster/loo_experiment.json", "w"), indent=2)
