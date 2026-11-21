"""ID card lines for the recognizer (Kaggle CPU, no GPU quota).

MIDV-UP gives mock Iranian documents in real photos, scans and video frames, with typed values and the document's corners.
Each view is flattened onto the template, so a value's box is known, and its lines are cropped as the recognizer sees them;
copies are damaged at page level as domain id_midv_noisy. Synthetic card lines in the training fonts add variety in
fonts and names, as domain id_synth. Test cards are never read here. Output shards use the v3 parquet schema.

MIDV-UP: Chernyshova, Ilyukhin, Arlazarov, ICDAR 2025, CC BY-SA 2.5.
"""
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import midv
from common import clean, crop_image, fits
from enhance_test import damage
from indra_ocr.rtl import visual_label
from lowq import LEVELS, lower_quality

SEED = 20261123
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
ZIP = Path(os.environ.get("MIDV_ZIP", "/tmp/midv-up.zip"))
DOCS = ("irn_id", "irn_drvlic", "irn_bc")
SYNTH_LINES = int(os.environ.get("SYNTH_LINES", 120000))
CARD_LIMIT = int(os.environ.get("CARD_LIMIT", 0))  # small local runs
# The ID card's printed labels and their fixed boxes on the template, checked on an enlarged card: pale text on a pattern defeats a threshold.
ID_LABELS = {"id_number": "شماره ملی :", "name": "نام :", "last_name": "نام خانوادگی :", "birth_date": "تاریخ تولد :", "father_name": "نام پدر :", "expiry_date": "پایان اعتبار :"}
ID_LABEL_BOXES = {"id_number": (823, 176, 970, 207), "name": (823, 244, 862, 272), "last_name": (823, 311, 963, 338), "birth_date": (823, 372, 947, 403), "father_name": (823, 430, 906, 460), "expiry_date": (823, 484, 949, 515)}
PLAN = [("level:light", 1.0), ("level:medium", 1.0), ("level:medium", 1.0), ("level:heavy", 1.0), ("level:heavy", 1.0), ("noise", .6), ("noise", 1.0), ("dirty", .8), ("lowres", 1.0), ("photo", .8), ("photo", 1.15)]
VALID_PLAN = [("level:light", 1.0), ("level:medium", 1.0), ("level:heavy", 1.0), ("noise", 1.0), ("photo", 1.0)]
schema = pa.schema([("ink", pa.binary()), ("width", pa.int32()), ("text", pa.string()), ("visual", pa.string()), ("source", pa.string()),
                    ("domain", pa.string()), ("doc_id", pa.string()), ("group", pa.string()), ("image_sha256", pa.string())])


def tight(gray, rect):
    """Ink box inside a rectangle of a template; annotated rectangles are looser than the boxes the detector gives."""
    x0, y0, x1, y1 = [int(v) for v in rect]
    h, w = gray.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    part = gray[y0:y1, x0:x1]
    if part.size == 0:
        return None
    _, ink = cv2.threshold(part, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.where(ink > 0)
    if len(xs) < 8:
        return None
    return float(x0 + xs.min()), float(y0 + ys.min()), float(x0 + xs.max() + 1), float(y0 + ys.max() + 1)


def template_lines(doc, gray, regions):
    """(field, text, box on the template) of each text line."""
    out = []
    for r in regions:
        a, field, value = r["shape_attributes"], r["region_attributes"].get("field_name"), r["region_attributes"].get("value")
        if a.get("name") != "rect" or not value or not value.strip():
            continue
        rect = (a["x"], a["y"], a["x"] + a["width"], a["y"] + a["height"])
        box = tight(gray, rect)
        if box:
            out.append((field, value, box))
        if doc == "irn_id" and field in ID_LABELS:
            out.append(("label/" + field, ID_LABELS[field], tuple(float(v) for v in ID_LABEL_BOXES[field])))
    return out


def flatten(image, quad, tw, th):
    """The document as a flat image in the template's frame, at about its resolution in this view. Returns (image, scale)."""
    q = np.float32(quad)
    width = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    s = float(np.clip(width / tw, .3, 1.0))
    H = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [tw * s, 0], [tw * s, th * s], [0, th * s]]))
    return cv2.warpPerspective(image, H, (round(tw * s), round(th * s)), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), s


def rows_from(flat, s, lines, card_id, split, view, kind, domain, rs, stats):
    out = []
    for field, text, box in lines:
        text = clean(text)
        j = rs.integers(-2, 3, 4)  # the detector's boxes are not exact either
        b = (box[0] * s + j[0], box[1] * s + j[1], box[2] * s + j[2], box[3] * s + j[3])
        if b[2] - b[0] < 6 or b[3] - b[1] < 4:
            continue
        crop = crop_image(flat, b)
        if not fits(crop, text):
            stats["too_short"] += 1
            continue
        data = crop.tobytes()
        out.append({"ink": data, "width": crop.shape[1], "text": text, "visual": visual_label(text), "source": f"{domain}/{view}/{field.split('/')[0]}", "domain": domain, "doc_id": card_id,
                    "group": hashlib.sha256(f"{split}/{card_id}/{view}/{field}/{kind}".encode()).hexdigest(), "image_sha256": hashlib.sha256(data).hexdigest()})
    return out


def card_job(job):
    doc, card, split, members = job
    cv2.setNumThreads(1)
    rs = np.random.default_rng((SEED, DOCS.index(doc), int(card)))
    stats, rows = collections.Counter(), []
    with zipfile.ZipFile(ZIP) as z:
        tmpl_bytes = z.read(members["template"])
        template = cv2.cvtColor(cv2.imdecode(np.frombuffer(tmpl_bytes, np.uint8), 1), cv2.COLOR_BGR2GRAY)
        th, tw = template.shape
        lines = template_lines(doc, template, members["regions"])
        if not lines:
            return rows, stats
        plan = PLAN if split == "train" else VALID_PLAN
        for view, member, quad in members["views"]:
            if quad is None:  # video frames come without the document's corners, so their lines cannot be located
                stats["no_corners"] += 1
                continue
            raw = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                stats["unusable_view"] += 1
                continue
            flat, s = flatten(raw, quad, tw, th)
            base = f"{view}/{Path(member).stem}"
            rows += rows_from(flat, s, lines, f"{doc}-{card}", split, base, "plain", "id_midv", rs, stats)
            stats[f"{split}/{view}"] += 1
            for v, (kind, strength) in enumerate(plan):
                bad = lower_quality(flat, *LEVELS[kind[6:]], rs) if kind.startswith("level:") else damage(flat, kind, rs, strength)
                f = bad.shape[1] / flat.shape[1]
                rows += rows_from(bad, s * f, lines, f"{doc}-{card}", split, base, f"{kind}/{v}", "id_midv_noisy", rs, stats)
    return rows, stats


class Shards:
    def __init__(self, prefix):
        self.prefix, self.buffers, self.files, self.counts = prefix, collections.defaultdict(list), [], collections.Counter()

    def add(self, split, row):
        self.buffers[split].append(row)
        self.counts[f"{split}/{row['domain']}"] += 1
        if len(self.buffers[split]) >= 20000:
            self.flush(split)

    def flush(self, split):
        if self.buffers[split]:
            name = f"{self.prefix}-{split}-{sum(f['split'] == split for f in self.files):04d}.parquet"
            pq.write_table(pa.Table.from_pylist(self.buffers[split], schema=schema), OUT / name, compression="zstd", compression_level=3, row_group_size=256)
            self.files.append({"file": name, "split": split, "rows": len(self.buffers[split]), "bytes": (OUT / name).stat().st_size})
            self.buffers[split] = []

    def close(self):
        for split in list(self.buffers):
            self.flush(split)


def midv_jobs():
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        name_set = set(names)
        cache = {}

        def meta(member):
            if member not in cache:
                cache[member] = json.load(z.open(member)) if member in name_set else {"_via_img_metadata": {}}
            return cache[member]

        by_card = collections.defaultdict(list)
        for n in names:  # video frames, grouped once: searching all names per card would take long
            p = n.split("/")
            if "video_frames" in p and "images" in p:
                by_card[p[-3], p[-2]].append(n)
        jobs = []
        for doc in DOCS:
            for i in range(1, 251):
                card = f"{i:03d}"
                split = midv.split_of(card)
                if split == "test":
                    continue
                views = midv.views(card, doc, by_card[f"{doc}_{midv.camera_of(card)}", card])
                template_regions = midv.regions(meta(views[0][2]), views[0][3])
                out = []
                for view, member, ann, fname in views:
                    if member not in name_set:
                        continue
                    if view == "template":
                        im = cv2.imdecode(np.frombuffer(z.read(member), np.uint8), cv2.IMREAD_GRAYSCALE)
                        h, w = im.shape
                        out.append((view, member, [[0, 0], [w, 0], [w, h], [0, h]]))
                        continue
                    quad = next(([list(p) for p in zip(r["shape_attributes"]["all_points_x"], r["shape_attributes"]["all_points_y"])] for r in midv.regions(meta(ann), fname) if r["region_attributes"].get("field_name") == "doc_quad"), None)
                    out.append((view, member, quad))
                jobs.append((doc, card, split, {"template": views[0][1], "regions": template_regions, "views": out}))
                if CARD_LIMIT and len(jobs) >= CARD_LIMIT:
                    return jobs
    return jobs


FIRST_EXTRA = """پانیذ راحله نسترن نیلوفر مهناز فریبا ناهید شهلا سوسن کتایون مرجان بیتا آتنا رویا ترانه مژگان نوشین هستی آیدا الناز باران دریا رها ستاره
اکبر اصغر جواد تقی نقی کاظم صادق جعفر موسی عیسی ابراهیم اسماعیل یعقوب غلامرضا غلامحسین محمدحسین محمدعلی محمدمهدی محمدامین علی‌اکبر علی‌اصغر سیدعلی سیدمحمد عمر عثمان بهرام رستم سهراب""".split()
LAST_EXTRA = """میرلاشاری بلوچی کردی لرستانی نظریان اسلامی کمالی جمالی فاضلی صالحی ناصری منصوری مظفری قنبری عسگری یزدانی رحمانی سبحانی جوادی مهدوی نعمتی شکری سعیدی وکیلی حیدریان مرادیان
احمدزاده حسن‌زاده علی‌پور رضاپور محمدپور نوروزی بیات افشار شاهرخی آذری ترکمان سیستانی بندری جنوبی گیلانی مازندرانی خراسانی""".split()


def id_line(r, st=None):
    """(kind, text) of one line of a card, for render.py's line renderer."""
    import fields as f

    first, last = f.FIRST + FIRST_EXTRA, f.LAST + LAST_EXTRA
    kind = r.choices(["id", "date", "first", "last", "two", "label"], [.25, .25, .15, .15, .1, .1])[0]
    if kind == "id":
        d = "".join(r.choice("0123456789") for _ in range(10))
        return kind, (d if r.random() < .8 else f"{d[:3]}-{d[3:9]}-{d[9]}").translate(f.FA)
    if kind == "date":
        return kind, f"{r.randint(1300, 1420)}/{r.randint(1, 12):02d}/{r.randint(1, 31 if r.random() < .5 else 30):02d}".translate(f.FA)
    if kind == "first":
        return kind, r.choice(first)
    if kind == "last":
        return kind, r.choice(last)
    if kind == "two":
        return kind, f"{r.choice(first)} {r.choice(last)}" if r.random() < .6 else f"{r.choice(last)} {r.choice(last)}"
    return kind, r.choice(list(ID_LABELS.values())).replace(" :", "" if r.random() < .3 else " :")


def synth_rows(shards):
    import fields
    import render

    fields.sample_line = id_line
    by = collections.defaultdict(list)
    for family, (split, _) in render.FAMILIES.items():
        if family in render.fonts and split != "test":
            by[split].append(family)
    print("FONTS", {s: len(v) for s, v in by.items()}, flush=True)
    r = random.Random(SEED)
    bold = lambda fam: 3 if any(x in fam.lower() for x in ("bd", "bold", "titr", "black")) else 1  # card values are heavy
    jobs = [(f, SEED * 1000 + k) for k, f in enumerate(r.choices(by["train"], [bold(f) for f in by["train"]], k=SYNTH_LINES))]
    jobs += [(f, SEED * 1000 + SYNTH_LINES + k) for k, f in enumerate(r.choices(by["validation"], k=max(100, SYNTH_LINES // 20)))]
    seen = set()
    with mp.Pool(mp.cpu_count()) as pool:
        for n, row in enumerate(pool.imap(render.render, jobs, chunksize=64), 1):
            if row is None or row["image_sha256"] in seen:
                continue
            seen.add(row["image_sha256"])
            family = jobs[n - 1][0]
            row.pop("look")
            row["domain"], row["source"] = "id_synth", row["source"].replace("receipt/", "id_synth/")
            shards.add(render.FAMILIES[family][0], row)
            if n % 20000 == 0:
                print("SYNTH", n, "/", len(jobs), flush=True)


def main():
    t0 = time.monotonic()
    if not ZIP.exists() or ZIP.stat().st_size != midv.SIZE:
        proc = subprocess.Popen(["curl", "-s", "-C", "-", "--retry", "30", "--retry-delay", "20", "--retry-all-errors", midv.URL, "-o", str(ZIP)])
        while proc.poll() is None:
            time.sleep(60)
            print("DOWNLOAD", round(ZIP.stat().st_size / 1e9, 2) if ZIP.exists() else 0, "GB", round(time.monotonic() - t0), "s", flush=True)
        assert proc.returncode == 0, proc.returncode
    assert ZIP.stat().st_size == midv.SIZE, "download incomplete"
    print("DOWNLOADED", round(time.monotonic() - t0), "s", flush=True)
    jobs = midv_jobs()
    assert not any(midv.split_of(j[1]) == "test" for j in jobs), "a test card reached training"
    print("CARDS", collections.Counter((j[0], j[2]) for j in jobs), flush=True)
    shards, stats = Shards("id-lines"), collections.Counter()
    with mp.Pool(mp.cpu_count()) as pool:
        for n, (rows, s) in enumerate(pool.imap_unordered(card_job, jobs), 1):
            stats.update(s)
            for row in rows:
                split = "train" if midv.split_of(row["doc_id"].split("-")[1]) == "train" else "validation"
                shards.add(split, row)
            if n % 25 == 0:
                print("MIDV", n, "/", len(jobs), dict(shards.counts), round(time.monotonic() - t0), "s", flush=True)
    synth_rows(shards)
    shards.close()
    (OUT / "id-lines-manifest.json").write_text(json.dumps({"seed": SEED, "docs": DOCS, "counts": dict(shards.counts), "stats": dict(stats), "shards": shards.files, "levels": LEVELS,
                                                            "source": "MIDV-UP, Chernyshova, Ilyukhin, Arlazarov, ICDAR 2025, CC BY-SA 2.5"}, indent=1), encoding="utf8")
    print("DONE", dict(shards.counts), dict(stats), round(time.monotonic() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
