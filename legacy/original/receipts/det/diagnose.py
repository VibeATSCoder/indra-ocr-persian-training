"""What can a better detector gain? Local diagnostics on the real receipts, before any GPU is spent.

raw: compares det.onnx's raw boxes with the hand-checked line boxes at each quality level, because final lines are
also cut and dropped by the recognizer's confidence. A box counts as found at IoU >= .5.
oracle: the whole pipeline with the hand-checked boxes in place of the detector, the most any detector could gain.

    PYTHONPATH=vendor/indra:v3:receipts:receipts/real .control-venv/bin/python receipts/det/diagnose.py raw <bundle> [--splits test,validation,train]
    PYTHONPATH=vendor/indra:v3:receipts:receipts/real .control-venv/bin/python receipts/det/diagnose.py oracle <bundle> --out f.json
"""
import argparse
import collections
import json
import multiprocessing as mp
from pathlib import Path

import cv2
import indra_ocr
import numpy as np
from indra_ocr import pipeline as P

from lowq import FIELDS, LEVELS, OUT, R, Gold, GoldPage, field_value, norm

HERE = Path(__file__).resolve().parents[1] / "real"
DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ALL_LEVELS = ("original", *LEVELS)


def gold_lines(splits):
    by = collections.defaultdict(list)
    for r in json.loads((HERE / "real-lines.json").read_text(encoding="utf8")):
        if r["split"] in splits:
            by[r["file"]].append(r)
    return by


def page_path(name, level):
    return R / name if level == "original" else OUT / level / (Path(name).stem + ".jpg")


def scale_of(level):
    return 1.0 if level == "original" else LEVELS[level][0]


def clean_skew(name):
    """The angle the gold boxes were measured at: the original page's, as build_lines.py used."""
    return P.estimate_skew(P.preview(cv2.imread(str(R / name), cv2.IMREAD_GRAYSCALE)))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def covered(a, b):
    """Share of box a inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def classify(g, boxes):
    """found, merged (inside a much larger box), split (only pieces of it), or missed."""
    if not len(boxes):
        return "missed"
    if max(iou(g, b) for b in boxes) >= .5:
        return "found"
    cover = [covered(g, b) for b in boxes]
    if max(cover) >= .6:
        return "merged"
    return "split" if sum(cover) >= .3 else "missed"


def raw_job(args):
    bundle, name, level, items, thresh, side = args
    cv2.setNumThreads(1)
    ocr = indra_ocr.load(path=bundle, threads=2, mode="ocr").ocr
    gray = cv2.imread(str(page_path(name, level)), cv2.IMREAD_GRAYSCALE)
    a = clean_skew(name)
    if abs(a) >= P.DESKEW_MIN:
        gray = P.rotate_page(gray, a)
    s = scale_of(level)
    x, f = P.page_to_input(gray, side)
    boxes = P.decode(ocr.det.run(None, {"x": x[None, None]})[0][0], thresh=thresh)
    boxes = boxes[:, :4] / f if len(boxes) else boxes
    out = collections.Counter()
    for g in items:
        kind = "digit" if any(c in DIGITS for c in g["text"]) else "text"
        out[kind, classify([v * s for v in g["box"]], boxes)] += 1
    golds = [[v * s for v in g["box"]] for g in items]
    out["extra", "boxes"] += sum(max((covered(b, g) for g in golds), default=0) < .3 for b in boxes)
    out["all", "boxes"] += len(boxes)
    return level, thresh, side, name, out


def raw(bundle, splits):
    gold = gold_lines(splits)
    settings = [(.5, 1024), (.35, 1024), (.5, 1536)]
    jobs = [(bundle, f, level, items, t, side) for level in ALL_LEVELS for t, side in settings for f, items in gold.items()]
    totals = collections.defaultdict(collections.Counter)
    with mp.get_context("spawn").Pool(10) as pool:
        for level, t, side, _, out in pool.imap_unordered(raw_job, jobs, chunksize=4):
            totals[level, t, side].update(out)
    print(f"{len(gold)} receipts, splits {splits}; shares of gold lines")
    print(f"{'level':9s}{'thresh':>7s}{'side':>6s}  {'digit found':>11s}{'merged':>8s}{'split':>7s}{'missed':>8s}  {'text found':>10s}{'merged':>8s}{'split':>7s}{'missed':>8s}  {'boxes':>6s}{'extra':>6s}")
    for level in ALL_LEVELS:
        for t, side in settings:
            c = totals[level, t, side]
            cells = ""
            for kind in ("digit", "text"):
                n = max(1, sum(c[kind, k] for k in ("found", "merged", "split", "missed")))
                cells += "  " + "".join(f"{100 * c[kind, k] / n:{w}.1f}%" for k, w in (("found", 10), ("merged", 7), ("split", 6), ("missed", 7)))
            print(f"{level:9s}{t:7.2f}{side:6d}{cells}  {c['all', 'boxes']:6d}{c['extra', 'boxes']:6d}", flush=True)


def oracle_job(args):
    bundle, name, level, items, use_gold = args
    cv2.setNumThreads(1)
    from receipt_ocr import parse, read

    conv = indra_ocr.load(path=bundle, threads=4, mode="ocr")
    if use_gold:
        s = scale_of(level)
        boxes = np.array([[*(v * s for v in g["box"]), 1.0] for g in items], np.float32)
        conv.ocr.detect_all = lambda gray: (boxes, np.zeros((0, 4), np.float32))
    res = read(conv, page_path(name, level))
    truth = parse(GoldPage([Gold(g["box"], g["text"]) for g in items]))["fields"]
    pred = collections.Counter(norm(l["text"]) for l in res["lines"])
    row = {"file": name, "level": level, "gold_boxes": use_gold, "digit_lines": 0, "digit_hit": 0, "fields": 0, "fields_right": 0, "wrong_silent": 0, "per_field": {}}
    for g in items:
        if any(c in DIGITS for c in g["text"]):
            n = norm(g["text"])
            ok = pred[n] > 0
            pred[n] -= ok
            row["digit_lines"] += 1
            row["digit_hit"] += ok
    flagged = {c["field"] for c in res["checks"]}
    for key in FIELDS:
        want = field_value(truth, key)
        if want is None:
            continue
        got = field_value(res["fields"], key)
        row["fields"] += 1
        row["fields_right"] += got == want
        row["per_field"][key] = got == want
        row["wrong_silent"] += got is not None and got != want and not ({key, "page"} & flagged)
    return row


def oracle(bundle, out):
    gold = gold_lines(("test",))
    jobs = [(bundle, f, level, items, use) for use in (False, True) for level in ALL_LEVELS for f, items in gold.items()]
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(oracle_job, jobs, chunksize=4)
    print(f"{'level':9s}{'boxes':>10s}{'digit lines':>13s}{'fields':>9s}{'amounts':>9s}{'wrong, unflagged':>18s}")
    for level in ALL_LEVELS:
        for use in (False, True):
            xs = [r for r in rows if r["level"] == level and r["gold_boxes"] == use]
            am = [r["per_field"]["amount"] for r in xs if "amount" in r["per_field"]]
            print(f"{level:9s}{'perfect' if use else 'detector':>10s}{100 * sum(r['digit_hit'] for r in xs) / max(1, sum(r['digit_lines'] for r in xs)):12.1f}%"
                  f"{100 * sum(r['fields_right'] for r in xs) / max(1, sum(r['fields'] for r in xs)):8.1f}%{100 * sum(am) / max(1, len(am)):8.1f}%{sum(r['wrong_silent'] for r in xs):>18d}", flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["raw", "oracle"])
    ap.add_argument("bundle")
    ap.add_argument("--splits", default="test,validation,train")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.action == "raw":
        raw(a.bundle, tuple(a.splits.split(",")))
    else:
        oracle(a.bundle, a.out)
