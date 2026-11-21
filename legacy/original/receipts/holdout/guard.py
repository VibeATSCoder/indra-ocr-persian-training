"""receipts-eval is sealed: never train, tune, select a model or adjust a parser on it.

Three checks enforce this. Kernel builders refuse it as a source, training scripts refuse to run if it is mounted, and data
builders skip any image that matches its fingerprints. The 76 t-cs files are the approved tuning set (receipts/real/dev76), not sealed.
receipts-eval.json holds only hashes of the images, no contents.
"""
import hashlib
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
# Only the sealed f-* receipts are blocked. The t-cs receipts are the approved tuning set, and the older real-receipt datasets hold none of the sealed ones.
SLUGS = ("receipts-eval",)
MOUNT_NAMES = ("receipts-eval.tar", "labels-r2")
MAX_DISTANCE = 10  # bits of 256: resized or recompressed copies of an image stay within this


def check_sources(meta):
    """Refuse a training or data kernel that attaches the sealed set."""
    sources = [*meta.get("dataset_sources", []), *meta.get("kernel_sources", []), *meta.get("model_sources", [])]
    bad = [s for s in sources if any(slug in s for slug in SLUGS)]
    if bad:
        raise SystemExit(f"Refusing to build: {bad} holds the sealed evaluation set and must never reach a training kernel")


def fingerprint(path):
    """(sha256, 256-bit difference hash) of an image."""
    data = Path(path).read_bytes()
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA) if gray is not None else None
    return hashlib.sha256(data).hexdigest(), None if small is None else "".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten())


class Holdout:
    def __init__(self):
        rows = json.loads((HERE / "receipts-eval.json").read_text(encoding="utf8"))["files"]
        self.sha = {r["sha256"] for r in rows}
        self.dhash = [r["dhash"] for r in rows if r["dhash"]]

    def __contains__(self, path):
        sha, dhash = fingerprint(path)
        if sha in self.sha:
            return True
        return dhash is not None and any(sum(a != b for a, b in zip(dhash, d)) <= MAX_DISTANCE for d in self.dhash)
