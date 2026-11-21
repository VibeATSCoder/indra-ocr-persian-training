"""The real receipts that may train the detector, as a private Kaggle dataset folder: train and validation splits only.

Pages are decoded and written again as PNG, which drops photo metadata and applies EXIF rotation as inference does. The
folder holds no transcriptions. Held-out test receipts stay on this machine, also when another file has the same bytes.

    .control-venv/bin/python receipts/det/build_pages.py     # writes receipts/real/pages-dataset/
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor/indra"))
from indra_ocr import pipeline as P

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "holdout"))
from guard import Holdout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "real"
R = Path("/var/services/chatbot_x/receipt-tests")
OUT = ROOT / "pages-dataset"
OUT.mkdir(exist_ok=True)
rows = json.loads((ROOT / "real-lines.json").read_text(encoding="utf8"))
split_of = {r["file"]: r["split"] for r in rows}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
held_out = {sha(R / f) for f, s in split_of.items() if s == "test"}
meta, skewed, skipped = [], 0, 0
sealed = Holdout()
for i, name in enumerate(sorted(f for f, s in split_of.items() if s != "test")):
    src = R / name
    assert sha(src) not in held_out, f"{name} has the bytes of a held-out receipt"
    if src in sealed:  # the sealed evaluation set never reaches training
        skipped += 1
        continue
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    skew = P.estimate_skew(P.preview(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    skewed += abs(skew) >= P.DESKEW_MIN
    dest = f"real-{i:03d}.png"  # no original file names
    cv2.imwrite(str(OUT / dest), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    meta.append({"file": dest, "split": split_of[name], "skew": round(float(skew), 2)})
(OUT / "real-pages.json").write_text(json.dumps(meta), encoding="utf8")
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "Persian Receipt Real Pages", "id": "sharifi76/persian-receipt-real-pages", "licenses": [{"name": "other"}]}), encoding="utf8")
print("sealed skipped:", skipped)
print(len(meta), "pages;", sum(m["split"] == "validation" for m in meta), "validation;", skewed, "skewed;", round(sum(p.stat().st_size for p in OUT.iterdir()) / 1e6, 1), "MB;", "held-out in dataset:", sum(s == "test" for s in [m["split"] for m in meta]))
