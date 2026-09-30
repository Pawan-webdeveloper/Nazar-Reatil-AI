"""Recover the GPU-trained weights that inventory.ipynb embedded in its output.

Usage:  python scripts/extract_weights_from_notebook.py [--nb inventory.ipynb] [--out models/shelf_detector_colab.pt]
"""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def collect_text(nb: dict) -> str:
    parts = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            txt = out.get("text")
            if txt is None and "data" in out:
                txt = out["data"].get("text/plain")
            if txt:
                parts.append("".join(txt) if isinstance(txt, list) else txt)
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=str(ROOT / "inventory.ipynb"))
    ap.add_argument("--out", default=str(ROOT / "models" / "shelf_detector_colab.pt"))
    a = ap.parse_args(argv)
    nb = json.loads(Path(a.nb).read_text(encoding="utf-8"))
    text = collect_text(nb)
    chunks, sha, results = {}, None, None
    for line in text.splitlines():
        if line.startswith("WEIGHTS_CHUNK::"):
            _, idx, data = line.split("::", 2)
            chunks[int(idx)] = data.strip()
        elif line.startswith("WEIGHTS_SHA256::"):
            sha = line.split("::", 1)[1].strip()
        elif line.startswith("RESULTS_JSON::"):
            results = line.split("::", 1)[1]
    if not chunks:
        print("No embedded weights found. Did the last cell finish and did you save the notebook (Ctrl+S)?")
        return 1
    missing = [i for i in range(max(chunks) + 1) if i not in chunks]
    if missing:
        print("Missing chunks", missing, "- output may have been truncated. Use the Google Drive copy instead.")
        return 1
    raw = base64.b64decode("".join(chunks[i] for i in sorted(chunks)))
    if sha and hashlib.sha256(raw).hexdigest() != sha:
        print("Checksum mismatch - file corrupted, not writing.")
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"wrote {out} ({len(raw) / 1e6:.1f} MB), sha256 verified")
    if results:
        (ROOT / "outputs" / "colab_results.json").write_text(json.dumps(json.loads(results), indent=2))
        print("saved outputs/colab_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
