"""Does a result hold on fresh damage? The held-out receipts through the whole pipeline, on copies made here and not used
for any choice: lowq.py's levels with new noise, and damage kinds the fixed test does not contain.

    PYTHONPATH=vendor/indra:v3:receipts:receipts/real .control-venv/bin/python receipts/det/robust.py v3=<bundle> v4=<bundle> [--out f.json]
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np

from enhance_test import damage
from lowq import FIELDS, LEVELS, R, Gold, GoldPage, field_value, lower_quality, norm

HERE = Path(__file__).resolve().parents[1] / "real"
SEED = 20261121  # not used for anything else
KINDS = ("light again", "medium again", "heavy again", "noise", "dirty", "lowres", "photo")


def damaged(name, kind):
    rs = np.random.default_rng((SEED, int(hashlib.sha256(f"{name}/{kind}".encode()).hexdigest()[:8], 16)))
    if kind.endswith(" again"):
        return cv2.cvtColor(lower_quality(cv2.imread(str(R / name), cv2.IMREAD_COLOR), *LEVELS[kind.split()[0]], rs), cv2.COLOR_BGR2GRAY)
    return damage(cv2.imread(str(R / name), cv2.IMREAD_GRAYSCALE), kind, rs)


def job(args):
    model, bundle, name, kind, items = args
    cv2.setNumThreads(1)
    import indra_ocr
    from receipt_ocr import parse, read_image

    ocr = indra_ocr.load(path=bundle, threads=4, mode="ocr")
    res = read_image(ocr, damaged(name, kind))
    truth = parse(GoldPage([Gold(g["box"], g["text"]) for g in items]))["fields"]
    flagged = {c["field"] for c in res["checks"]}
    row = {"model": model, "file": name, "kind": kind, "fields": 0, "right": 0, "silent": 0, "per_field": {}}
    for key in FIELDS:
        want = field_value(truth, key)
        if want is None:
            continue
        got = field_value(res["fields"], key)
        row["fields"] += 1
        row["right"] += got == want
        row["per_field"][key] = got == want
        row["silent"] += got is not None and got != want and not ({key, "page"} & flagged)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=bundle")
    ap.add_argument("--out")
    a = ap.parse_args()
    gold = collections.defaultdict(list)
    for r in json.loads((HERE / "real-lines.json").read_text(encoding="utf8")):
        if r["split"] == "test":
            gold[r["file"]].append(r)
    models = dict(m.split("=", 1) for m in a.models)
    jobs = [(m, b, f, k, items) for m, b in models.items() for k in KINDS for f, items in gold.items()]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(job, jobs, chunksize=4)
    print(f"{len(gold)} receipts")
    print(f"{'damage':14s}" + "".join(f"{m + ' fields':>12s}{'amounts':>9s}{'silent':>7s}" for m in models))
    for k in KINDS:
        cells = ""
        for m in models:
            xs = [r for r in rows if r["model"] == m and r["kind"] == k]
            am = [r["per_field"]["amount"] for r in xs if "amount" in r["per_field"]]
            cells += f"{100 * sum(r['right'] for r in xs) / max(1, sum(r['fields'] for r in xs)):11.1f}%{100 * sum(am) / max(1, len(am)):8.1f}%{sum(r['silent'] for r in xs):7d}"
        print(f"{k:14s}{cells}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
