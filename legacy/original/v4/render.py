"""Render book-style Persian lines with scan degradations (Kaggle CPU, no GPU quota).

Text comes from the existing v3 corpus splits, so a train/validation/test text never moves
across splits. Font families are disjoint per split: Nahid = validation, Nazli = test.
Output shards use the v3 parquet schema with domain 'bookfont'.
"""
import collections
import hashlib
import json
import multiprocessing as mp
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, features

from common import clean, crop_image, fits, key
from indra_ocr.rtl import visual_label

assert features.check_feature("raqm"), "Arabic shaping (raqm) required"
SEED = 20261103
import os
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
SCALE = float(os.environ.get("RENDER_SCALE", "1"))
v3 = next(INPUT.rglob("manifest.json")).parent
manifest = json.loads((v3 / "manifest.json").read_text())
allowed = set(manifest["characters"])
fonts = {p.stem.lower(): p for p in INPUT.rglob("*.ttf")}
print("FONTS", sorted(fonts), flush=True)

FAMILIES = {  # family -> (split, lines)
    "samim": ("train", 15000), "shabnam": ("train", 15000), "tanha": ("train", 15000),
    "parastoo": ("train", 15000), "gandom": ("train", 15000), "homa": ("train", 15000),
    "notokufiarabic": ("train", 15000), "estedad": ("train", 15000), "titr": ("train", 6000),
    "nahid": ("validation", 2500), "nazli": ("test", 3000), "nazlib": ("test", 1500),
}
missing = [f for f in FAMILIES if f not in fonts]
assert not missing, f"missing fonts {missing}"

# ---- texts per split from v3 (natural book lines first: arshasb, then 230k, garshasp)
texts = collections.defaultdict(list)
for shard in manifest["shards"]:
    t = pq.read_table(v3 / shard["file"], columns=["text", "domain"])
    for text, domain in zip(t.column("text").to_pylist(), t.column("domain").to_pylist()):
        if domain in ("arshasb", "persian230k", "garshasp") and 12 <= len(text) <= 110 and not set(text) - allowed:
            texts[shard["split"]].append(text)
rng = random.Random(SEED)
for s in texts:
    texts[s] = list(dict.fromkeys(texts[s]))
    rng.shuffle(texts[s])
print("TEXTS", {s: len(v) for s, v in texts.items()}, flush=True)


def line_texts(split, n, r):
    pool, out = texts[split], []
    while len(out) < n:
        a = pool[r.randrange(len(pool))]
        if len(a) < 45 and r.random() < .6:  # join short items into book-length lines
            b = pool[r.randrange(len(pool))]
            if len(a) + len(b) + 1 <= 110:
                a = a + " " + b
        out.append(a)
    return out


_font_cache = {}


def font(family, size):
    k = (family, size)
    if k not in _font_cache:
        _font_cache[k] = ImageFont.truetype(str(fonts[family]), size)
    return _font_cache[k]


def degrade(img, r):
    """img: uint8 gray, dark text on light paper. Randomised scanner/photocopy chain."""
    h, w = img.shape
    out = img.astype(np.float32)
    if r.random() < .5:  # uneven illumination
        g = np.linspace(r.uniform(-25, 0), r.uniform(-25, 0), w)[None, :] + np.linspace(r.uniform(-15, 0), r.uniform(-15, 0), h)[:, None]
        out = out + g
    if r.random() < .5:  # paper texture
        out = out + cv2.GaussianBlur(np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(2, 9), (h, w)).astype(np.float32), (0, 0), r.uniform(.5, 2))
    out = np.clip(out, 0, 255).astype(np.uint8)
    if r.random() < .6:  # low resolution scan
        f = r.uniform(.3, .85)
        small = cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=r.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST]))
    if r.random() < .3:  # ink spread / thinning
        k = np.ones((2, 2), np.uint8)
        out = cv2.erode(out, k) if r.random() < .6 else cv2.dilate(out, k)
    if r.random() < .5:
        out = cv2.GaussianBlur(out, (0, 0), r.uniform(.3, 1.3))
    if r.random() < .5:
        out = np.clip(out.astype(np.float32) + np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(2, 14), out.shape), 0, 255).astype(np.uint8)
    if r.random() < .2:  # binarised photocopy / fax
        out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] if r.random() < .5 else \
            cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, r.randint(5, 15))
    if r.random() < .6:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, r.randint(25, 92)])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def render(job):
    text, family, split, seed = job
    r = random.Random(seed)
    text = clean(text)
    if not 3 <= len(text) <= 240 or set(visual_label(text)) - allowed:
        return None
    size = r.randint(24, 46)
    f = font(family, size)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    stroke = 1 if r.random() < .12 else 0
    box = probe.textbbox((0, 0), text, font=f, direction="rtl", language="fa", stroke_width=stroke)
    pad = r.randint(6, 18)
    W, H = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    paper, ink = r.randint(200, 255), r.randint(0, 70)
    im = Image.new("L", (W, H), paper)
    d = ImageDraw.Draw(im)
    if r.random() < .15:  # faint bleed-through from the reverse side
        other = texts[split][r.randrange(len(texts[split]))]
        d.text((pad - box[0] + r.randint(-20, 20), pad - box[1] + r.randint(-6, 6)), other, font=f, fill=int(paper - r.uniform(15, 40)), direction="rtl", language="fa")
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        d = ImageDraw.Draw(im)
    d.text((pad - box[0], pad - box[1]), text, font=f, fill=ink, direction="rtl", language="fa", stroke_width=stroke, stroke_fill=ink)
    gray = np.asarray(im)
    angle = r.uniform(-.6, .6) if r.random() < .5 else 0
    if angle:  # rotate before measuring the ink box so the crop keeps both line ends
        m = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1)
        gray = cv2.warpAffine(gray, m, (W, H), borderValue=paper)
    ys, xs = np.where(gray < (paper + ink) / 2)
    if len(xs) < 3:
        return None
    gray = degrade(gray, r)
    j = lambda: r.randint(-2, 3)
    bx = (max(0, xs.min() - 2 + j()), max(0, ys.min() - 2 + j()), min(W, xs.max() + 3 + j()), min(H, ys.max() + 3 + j()))
    if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 6:
        return None
    try:
        crop = crop_image(gray, bx)
    except Exception:
        return None
    if not fits(crop, text):
        return None
    data = crop.tobytes()
    return {"ink": data, "width": crop.shape[1], "text": text, "visual": visual_label(text), "source": "bookfont/" + family,
            "domain": "bookfont", "doc_id": family, "group": key(text), "image_sha256": hashlib.sha256(data).hexdigest()}


schema = pa.schema([("ink", pa.binary()), ("width", pa.int32()), ("text", pa.string()), ("visual", pa.string()), ("source", pa.string()),
                    ("domain", pa.string()), ("doc_id", pa.string()), ("group", pa.string()), ("image_sha256", pa.string())])
jobs = []
for i, (family, (split, n)) in enumerate(FAMILIES.items()):
    r = random.Random(SEED + i)
    jobs += [(t, family, split, SEED * 1000 + i * 100000 + k) for k, t in enumerate(line_texts(split, max(1, int(n * SCALE)), r))]
rng.shuffle(jobs)
buffers, shards, counts, rejected, seen = collections.defaultdict(list), [], collections.Counter(), collections.Counter(), set()


def flush(split):
    if not buffers[split]:
        return
    name = f"bookfont-{split}-{sum(s['split'] == split for s in shards):04d}.parquet"
    pq.write_table(pa.Table.from_pylist(buffers[split], schema=schema), OUT / name, compression="zstd", compression_level=3, row_group_size=256)
    shards.append({"file": name, "split": split, "rows": len(buffers[split]), "bytes": (OUT / name).stat().st_size})
    buffers[split] = []


split_of = {f: s for f, (s, _) in FAMILIES.items()}
with mp.Pool(mp.cpu_count()) as pool:
    for n, row in enumerate(pool.imap(render, jobs, chunksize=64)):
        fam = jobs[n][1]
        if row is None:
            rejected["render_or_fit"] += 1
            continue
        if row["image_sha256"] in seen:
            rejected["duplicate_image"] += 1
            continue
        seen.add(row["image_sha256"])
        s = split_of[fam]
        buffers[s].append(row)
        counts[f"{s}:{fam}"] += 1
        if len(buffers[s]) >= 20000:
            flush(s)
        if (n + 1) % 10000 == 0:
            print("RENDERED", n + 1, "/", len(jobs), flush=True)
for s in list(buffers):
    flush(s)
# A few previews for visual inspection
prev = OUT / "previews"
prev.mkdir(exist_ok=True)
for s in shards[:3]:
    for k, row in enumerate(pq.read_table(OUT / s["file"]).slice(0, 8).to_pylist()):
        Image.fromarray(np.frombuffer(row["ink"], np.uint8).reshape(32, row["width"])).save(prev / f"{s['split']}-{k}-{row['doc_id']}.png")
(OUT / "bookfont-manifest.json").write_text(json.dumps({"seed": SEED, "families": {f: s for f, (s, _) in FAMILIES.items()}, "counts": dict(counts),
    "rejected": dict(rejected), "shards": shards, "text_source": "v3 corpus splits (arshasb, persian230k, garshasp labels)",
    "degradations": "illumination, paper texture, low-res resample, erode/dilate, blur, noise, Otsu/adaptive binarisation, JPEG 25-92, bleed-through, +-1 deg rotation"}, indent=1))
print("DONE", json.dumps(dict(counts)), json.dumps(dict(rejected)), flush=True)
