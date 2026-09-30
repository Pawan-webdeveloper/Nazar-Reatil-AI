"""Verify shopper analytics against human ground truth (MOT17 benchmark).

MOT17 (MOTChallenge) sequences have every pedestrian boxed with a persistent id in every frame.
From that ground truth we derive the TRUE answers and compare them with what the edge pipeline
(YOLO person detector + ByteTrack + the same LineCounter code) reports:

  * line-crossing counts in both directions (= footfall in / out)
  * people visible per frame (= occupancy)
  * person detection precision / recall at IoU 0.5 on the processed frames

Each sequence gets a horizontal AND a vertical counting line through the image centre, so both
walking directions are tested. Models and frame rates are varied to show the accuracy / cost trade-off.

Usage: python training/evaluate_people_counting.py --root C:/Users/itanm/retail_ai_data/MOT17/train
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retail_ai.analytics.footfall import LineCounter  # noqa: E402
from retail_ai.geometry import greedy_match, iou_matrix  # noqa: E402

OUT = ROOT / "outputs" / "people_counting"


def load_seq(d: Path):
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    s = ini["Sequence"]
    info = {"name": s["name"], "fps": float(s["frameRate"]), "n": int(s["seqLength"]),
            "w": int(s["imWidth"]), "h": int(s["imHeight"])}
    gt = np.loadtxt(d / "gt" / "gt.txt", delimiter=",")
    # MOT17 gt columns: frame, id, x, y, w, h, consider_flag, class, visibility ; class 1 = pedestrian
    gt = gt[(gt[:, 7] == 1) & (gt[:, 6] == 1)]
    return info, gt, sorted((d / "img1").glob("*.jpg"))


def lines_for(info):
    w, h = info["w"], info["h"]
    return {"horizontal": (((0, h * 0.55), (w, h * 0.55)), "down"),
            "vertical": (((w * 0.5, 0), (w * 0.5, h)), "right")}


def gt_counts(info, gt, lines):
    res = {}
    for name, (line, direction) in lines.items():
        lc = LineCounter(line, direction, recount_s=1e9, stitch_px=0)  # GT ids never switch
        events = []
        for f in range(1, info["n"] + 1):
            rows = gt[gt[:, 0] == f]
            vis = {int(r[1]): float(r[8]) for r in rows}
            for ev in lc.update(f / info["fps"], {int(r[1]): (r[2] + r[4] / 2, r[3] + r[5]) for r in rows}):
                events.append({"ts": ev.ts, "dir": ev.direction, "pt": ev.point, "visible": vis[ev.track_id] >= 0.3})
        res[name] = {"in": lc.entries, "out": lc.exits, "events": events}
    return res


def match_events(gt_ev, sys_ev, max_dt=1.0, max_dist=150.0):
    """Greedy one-to-one matching of crossing events by direction, time and position."""
    used, tp, tp_vis = set(), 0, 0
    for g in sorted(gt_ev, key=lambda e: e["ts"]):
        best, best_dt = None, max_dt
        for i, e in enumerate(sys_ev):
            if i in used or e["dir"] != g["dir"]:
                continue
            dt = abs(e["ts"] - g["ts"])
            dist = ((e["pt"][0] - g["pt"][0]) ** 2 + (e["pt"][1] - g["pt"][1]) ** 2) ** 0.5
            if dt <= best_dt and dist <= max_dist:
                best, best_dt = i, dt
        if best is not None:
            used.add(best)
            tp += 1
            tp_vis += int(g["visible"])
    return tp, tp_vis


def run_system(info, gt, frames, lines, weights, stride, process_width, conf, imgsz=640, margin=8.0, stitch_px=0.0, tracker="bytetrack.yaml"):
    from ultralytics import YOLO

    model = YOLO(weights)
    scale = process_width / info["w"]
    counters = {n: LineCounter(((l[0][0] * scale, l[0][1] * scale), (l[1][0] * scale, l[1][1] * scale)), d,
                               recount_s=1e9, margin=margin, stitch_px=stitch_px) for n, (l, d) in lines.items()}
    occ_err, occ_gt, tp = [], [], 0
    n_pred = n_gt = 0
    t_inf = []
    for idx in range(0, len(frames), stride):
        f = idx + 1
        img = cv2.imread(str(frames[idx]))
        img = cv2.resize(img, (process_width, int(round(info["h"] * scale))))
        t = time.perf_counter()
        r = model.track(img, persist=True, classes=[0], conf=conf, tracker=tracker, verbose=False, imgsz=imgsz,
                        device="cpu")[0]
        t_inf.append((time.perf_counter() - t) * 1000)
        feet = {}
        if r.boxes is not None and r.boxes.id is not None:
            for tid, b in zip(r.boxes.id.int().tolist(), r.boxes.xyxy.tolist()):
                feet[tid] = ((b[0] + b[2]) / 2, b[3])
        for c in counters.values():
            c.update(f / info["fps"], feet)
        # detection quality against visible ground truth (visibility >= 0.3)
        g = gt[(gt[:, 0] == f) & (gt[:, 8] >= 0.3)]
        gb = np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], 1) * scale if len(g) else np.zeros((0, 4))
        pb = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        tp += len(greedy_match(iou_matrix(gb, pb), 0.5))
        n_pred += len(pb)
        n_gt += len(gb)
        occ_err.append(abs(len(feet) - len(gb)))
        occ_gt.append(len(gb))
    counts = {n: {"in": c.entries, "out": c.exits,
                  "events": [{"ts": e.ts, "dir": e.direction, "pt": (e.point[0] / scale, e.point[1] / scale)}
                             for e in c.events]} for n, c in counters.items()}
    return counts, {
        "detection_precision": round(tp / max(1, n_pred), 3), "detection_recall": round(tp / max(1, n_gt), 3),
        "occupancy_MAE": round(float(np.mean(occ_err)), 2), "mean_people_visible": round(float(np.mean(occ_gt)), 1),
        "occupancy_accuracy_%": round(100 * (1 - float(np.sum(occ_err)) / max(1, float(np.sum(occ_gt)))), 1),
        "ms_per_frame": round(float(np.median(t_inf)), 1), "frames_processed": len(t_inf)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seqs", nargs="+", default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"])
    ap.add_argument("--configs", nargs="+", default=["yolo11n.pt:3:640", "yolo11n.pt:6:640", "yolo11s.pt:3:640",
                                                         "yolo11s.pt:6:640"], help="model:fps:imgsz")
    ap.add_argument("--margin", type=float, default=8.0, help="counting dead band in processed pixels")
    ap.add_argument("--tag", default="")
    ap.add_argument("--stitch-px", type=float, default=0.0)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--process-width", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"settings": vars(a), "sequences": {}}
    total = defaultdict(lambda: {"gt": 0, "abs_err": 0})
    for seq in a.seqs:
        info, gt, frames = load_seq(Path(a.root) / seq)
        lines = lines_for(info)
        truth = gt_counts(info, gt, lines)
        entry = {"info": info, "ground_truth_crossings": {n: {k: v for k, v in t.items() if k != "events"}
                                                           for n, t in truth.items()}, "runs": {}}
        truth_counts = entry["ground_truth_crossings"]
        print(seq, "ground truth crossings:", truth_counts, flush=True)
        for spec in a.configs:
            m, fps, imgsz = spec.split(":"); fps = float(fps); imgsz = int(imgsz)
            if True:
                stride = max(1, int(round(info["fps"] / fps)))
                counts, quality = run_system(info, gt, frames, lines, str(ROOT / "models" / m), stride,
                                             a.process_width, a.conf, imgsz, a.margin, a.stitch_px, a.tracker)
                err = {n: {k: counts[n][k] - truth_counts[n][k] for k in ("in", "out")} for n in truth_counts}
                key = f"{m}@{fps:g}fps/{imgsz}px"
                ev = {"gt": 0, "gt_visible": 0, "sys": 0, "tp": 0, "tp_visible": 0}
                for n in truth:
                    tp, tpv = match_events(truth[n]["events"], counts[n]["events"], max_dist=0.08 * info["w"])
                    ev["gt"] += len(truth[n]["events"]); ev["sys"] += len(counts[n]["events"])
                    ev["gt_visible"] += sum(e["visible"] for e in truth[n]["events"])
                    ev["tp"] += tp; ev["tp_visible"] += tpv
                for k2 in ev:
                    total[key][k2] = total[key].get(k2, 0) + ev[k2]
                slim = {n: {k: v for k, v in c.items() if k != "events"} for n, c in counts.items()}
                entry["runs"][key] = {"counts": slim, "count_error": err, "events": ev, **quality}
                counts = slim
                gt_sum = sum(truth_counts[n][k] for n in truth_counts for k in ("in", "out"))
                abs_err = sum(abs(err[n][k]) for n in truth_counts for k in ("in", "out"))
                total[key]["count_gt"] = total[key].get("count_gt", 0) + gt_sum
                total[key]["abs_err"] += abs_err
                print(f"  {key}: counts {counts} | err {err} | {quality}", flush=True)
        results["sequences"][seq] = entry
    results["summary"] = {k: {"true_crossings": v["gt"], "visible_true_crossings": v["gt_visible"],
                              "absolute_count_error": v["abs_err"],
                              "counting_accuracy_%": round(100 * (1 - v["abs_err"] / max(1, v["count_gt"])), 1),
                              "event_precision": round(v["tp"] / max(1, v["sys"]), 3),
                              "event_recall_all": round(v["tp"] / max(1, v["gt"]), 3),
                              "event_recall_visible": round(v["tp_visible"] / max(1, v["gt_visible"]), 3)}
                          for k, v in total.items()}
    (OUT / f"mot17_counting_eval{a.tag}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
