"""Why are line crossings missed? For each ground-truth crossing, look at which system track ids
covered that person before and after the line (MOT17-09, vertical line, yolo11n 960 px, 6 fps)."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from evaluate_people_counting import load_seq  # noqa: E402
from retail_ai.geometry import greedy_match, iou_matrix  # noqa: E402
from ultralytics import YOLO  # noqa: E402

seq = sys.argv[1] if len(sys.argv) > 1 else "MOT17-09-FRCNN"
info, gt, frames = load_seq(Path("C:/Users/itanm/retail_ai_data/MOT17/train") / seq)
X = info["w"] * 0.5
m = YOLO(str(ROOT / "models" / "yolo11n.pt"))
scale = 960 / info["w"]
stride = 5
match = {}  # (frame, gt_id) -> system id
for idx in range(0, len(frames), stride):
    f = idx + 1
    img = cv2.resize(cv2.imread(str(frames[idx])), (960, int(info["h"] * scale)))
    r = m.track(img, persist=True, classes=[0], conf=0.25, imgsz=960, tracker="bytetrack.yaml", verbose=False, device="cpu")[0]
    g = gt[gt[:, 0] == f]
    if r.boxes is None or r.boxes.id is None or not len(g):
        continue
    gb = np.stack([g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]], 1) * scale
    pb = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.int().tolist()
    for i, j in greedy_match(iou_matrix(gb, pb), 0.4):
        match[(f, int(g[i, 1]))] = ids[j]

for tid in np.unique(gt[:, 1]):
    t = gt[gt[:, 1] == tid]
    t = t[np.argsort(t[:, 0])]
    fx = t[:, 2] + t[:, 4] / 2
    side = np.sign(fx - X)
    if len(np.unique(side[np.abs(fx - X) > 16])) < 2:
        continue
    seq_ids = []
    for f, s, vis in zip(t[:, 0].astype(int), side, t[:, 8]):
        if (f - 1) % stride:
            continue
        sid = match.get((f, int(tid)))
        seq_ids.append(f"{'L' if s < 0 else 'R'}{'' if sid is None else sid}{'' if vis > 0.3 else '(occ)'}")
    print(f"gt {int(tid)}:", " ".join(seq_ids))
