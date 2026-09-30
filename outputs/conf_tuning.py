"""Pick the counting confidence threshold on VALIDATION, report it on TEST (no test-set tuning)."""
import glob, json, numpy as np
from ultralytics import YOLO
m = YOLO('models/shelf_detector.pt')
TH = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

def counts(split):
    out = []
    for p in sorted(glob.glob(f'C:/Users/itanm/retail_ai_data/sku110k_yolo/images/{split}/*.jpg')):
        gt = sum(1 for _ in open(p.replace('images', 'labels').replace('.jpg', '.txt')))
        c = m.predict(p, imgsz=640, conf=min(TH), max_det=1000, verbose=False, device='cpu')[0].boxes.conf.numpy()
        out.append((gt, [int((c >= t).sum()) for t in TH]))
    return out

def score(rows):
    gt = np.array([g for g, _ in rows], float); pr = np.array([p for _, p in rows], float)
    ape = np.abs(pr - gt[:, None]) / gt[:, None]
    res = {}
    for k, t in enumerate(TH):
        res[t] = {'count_accuracy_%': round(100 * (1 - ape[:, k].mean()), 2),
                  'within_10pct_%': round(100 * (ape[:, k] <= 0.1).mean(), 1),
                  'sparse_acc_%': round(100 * (1 - ape[gt < 100, k].mean()), 1) if (gt < 100).any() else None}
    return res

val = score(counts('val'))
best = max(TH, key=lambda t: val[t]['count_accuracy_%'])
test = score(counts('test'))
out = {'validation': val, 'selected_conf': best, 'test_at_selected': test[best], 'test_at_0.25': test[0.25],
       'note': 'local test = first test shard (420 images); Colab reported all 2935 test images at conf 0.25'}
json.dump(out, open('outputs/shelf_conf_tuning.json', 'w'), indent=2)
print(json.dumps(out, indent=1))
