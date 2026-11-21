"""Render synthetic Persian receipt lines (Kaggle CPU, no GPU quota).

Screen, photo, scan and thermal-print looks on receipt fields: masked cards, amounts, dates, tracking numbers,
IBANs, names, labels and notices. Look and wording follow the real receipts: dark mode, inline bank icons, small
asterisks and units, dotted leaders, watermarks. Font families are disjoint per split. Output shards use the
v3 parquet schema with domain 'receipt' and source 'receipt/<kind>'.

Many Persian fonts draw ASCII digits with Persian shapes, so Latin text uses a fallback font wherever
fonts-meta.json says the font cannot draw it: the picture always matches its label.
"""
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont, features

import fields
from common import clean, crop_image, fits
from indra_ocr.rtl import flip, visual_label
from layout import runs

assert features.check_feature("raqm"), "Arabic shaping (raqm) required"
SEED = 20261115
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
SCALE = float(os.environ.get("RENDER_SCALE", "1"))
allowed = set(json.loads(next(INPUT.rglob("manifest.json")).read_text())["characters"])
fonts = {p.stem.lower(): p for p in INPUT.rglob("*") if p.suffix.lower() == ".ttf"}
print("FONTS", sorted(fonts), flush=True)

META = {k.lower(): v for k, v in json.loads(next(INPUT.rglob("fonts-meta.json")).read_text(encoding="utf8")).items()}
FAMILIES = {f: (m["split"], m["lines"]) for f, m in META.items()}
missing = [f for f in FAMILIES if f not in fonts]
assert not missing, f"missing fonts {missing}"
FALLBACK = "vazirmatn"
LATIN = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
LOOKS = {"screen": .40, "photo": .25, "scan": .20, "thermal": .15}
SS = 3  # supersampling factor

_font_cache = {}


def font(family, size, weight):
    k = (family, size, weight)
    if k not in _font_cache:
        f = ImageFont.truetype(str(fonts[family]), size)
        try:
            axes = f.get_variation_axes()
            f.set_variation_by_axes([weight if a.get("name") in (b"Weight", "Weight") else a["default"] for a in axes])
        except (OSError, AttributeError):  # static font
            pass
        _font_cache[k] = f
    return _font_cache[k]


def text_parts(text, family, size, weight, unit_scale=1.0, mask_scale=1.0):
    """Drawable pieces of a line, left to right: (text, drawing args, font)."""
    meta = META.get(family, {})
    latin_ok, missing_chars = meta.get("latin_ok", True), set(meta.get("missing_punct", ""))
    f = font(family, size * SS, weight)
    fb = font(FALLBACK, size * SS, 400)
    parts = []
    for rtl, run in runs(visual_label(text)):
        if rtl:
            logical = flip(run, base_rtl=True)
            bare = logical.strip()
            use = font(family, max(8, int(size * unit_scale)) * SS, weight) if bare in ("ریال", "ريال", "تومان") else f
            parts.append((logical, {"direction": "rtl", "language": "fa"}, fb if set(logical) & missing_chars else use))
            continue
        start = 0  # same-font stretches of a left-to-right run
        kind = lambda c: "mask" if c in "*•" else "fallback" if (c in LATIN and not latin_ok) or c in missing_chars else "own"
        for i in range(1, len(run) + 1):
            if i == len(run) or kind(run[i]) != kind(run[start]):
                k = kind(run[start])
                use = fb if k == "fallback" else font(family, max(8, int(size * mask_scale)) * SS, weight) if k == "mask" else f
                parts.append((run[start:i], {"direction": "ltr", "features": ["-liga"]}, use))
                start = i
    return parts


def draw_parts(parts, size, r):
    """Ink mask (float 0-1) of pieces on one baseline, at SS x scale. A piece may be ("gap", width, dotted)."""
    width = sum(p[1] if p[0] == "gap" else p[2].getlength(p[0], **p[1]) for p in parts)
    W, H = int(width) + 8 * size * SS, 4 * size * SS
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    x, base = 4 * size * SS // 2, int(H * .62)
    for p in parts:
        if p[0] == "gap":
            if p[2]:  # dotted leader
                step, dot = int(size * SS * r.uniform(.22, .4)), max(1, int(size * SS * .035))
                for dx in range(0, int(p[1]), max(2, step)):
                    d.ellipse((x + dx - dot, base - dot, x + dx + dot, base + dot), fill=r.randint(140, 255))
            x += p[1]
            continue
        d.text((x, base), p[0], font=p[2], fill=255, anchor="ls", **p[1])
        x += p[2].getlength(p[0], **p[1])
    return im


def add_icon(im, size, r):
    """A small bank-logo or status blob beside the text; part of real line boxes, not of the label."""
    a = np.asarray(im)
    xs = np.where(a.max(0) > 128)[0]
    ys = np.where(a.max(1) > 128)[0]
    if not len(xs):
        return im
    s = int(size * SS * r.uniform(.8, 1.3))
    gap = int(size * SS * r.uniform(.25, .9))
    cy = (ys.min() + ys.max()) // 2
    left = r.random() < .8
    x0 = xs.min() - gap - s if left else xs.max() + gap
    if x0 < 0 or x0 + s >= im.width:
        return im
    d = ImageDraw.Draw(im)
    fill = r.randint(90, 230)
    box = (x0, cy - s // 2, x0 + s, cy + s // 2)
    if r.random() < .6:
        d.ellipse(box, fill=fill)
    else:
        d.rounded_rectangle(box, radius=s // 4, fill=fill)
    inner = s // r.randint(3, 5)  # a mark inside, as logos and check marks have
    d.ellipse((x0 + inner, cy - s // 2 + inner, x0 + s - inner, cy + s // 2 - inner), fill=r.randint(0, 120))
    return im


def draw_line(text, family, size, weight, r, kind):
    """Ink mask (float 0-1, H x W) of a line at SS x scale. Returns None when nothing was drawn."""
    unit = r.uniform(.55, .75) if r.random() < .25 else 1.0
    mask = r.uniform(.65, .85) if r.random() < .3 else 1.0
    gap_text = "\uE000" in text
    plain = text.replace("\uE000", " ")
    parts = None
    if gap_text and r.random() < .5:
        label_text, value_text = text.split("\uE000", 1)
        if visual_label(clean(plain)) == visual_label(clean(value_text)) + " " + visual_label(clean(label_text)):  # same order as the label
            gap = int(size * SS * r.uniform(1.5, 7))
            parts = (text_parts(clean(value_text), family, size, weight, unit, mask) + [("gap", gap, r.random() < .6)]
                     + text_parts(clean(label_text), family, size, weight, unit, mask))
    im = draw_parts(parts or text_parts(clean(plain), family, size, weight, unit, mask), size, r)
    if kind in ("card", "person", "bank", "iban", "pair") and r.random() < .22:
        im = add_icon(im, size, r)
    a = np.asarray(im, dtype=np.float32) / 255
    ys, xs = np.where(a > .5)
    if not len(xs):
        return None
    return a[max(0, ys.min() - 2 * size):ys.max() + 2 * size, max(0, xs.min() - size):xs.max() + size]


def watermark(gray, paper, r):
    """Faint diagonal stamp across the line, as some banks print over their receipts."""
    h, w = gray.shape
    mark = Image.new("L", (w, h), 0)
    f = font(FALLBACK, max(10, int(h * r.uniform(.5, 1.1))), 400)
    ImageDraw.Draw(mark).text((r.randint(-w // 4, w // 2), r.randint(-h // 4, h // 3)), r.choice(["تراکنش موفق", "موفق", "نسخه مشتری"]), font=f, fill=255, direction="rtl", language="fa")
    m = cv2.warpAffine(np.asarray(mark, np.float32) / 255, cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-40, 40), 1), (w, h))
    strength = r.uniform(12, 55) * (1 if paper > 128 else -1)
    return np.clip(gray.astype(np.float32) - strength * m, 0, 255).astype(np.uint8)


def degrade_scan(out, r):
    h, w = out.shape
    out = out.astype(np.float32)
    if r.random() < .5:
        out = out + np.linspace(r.uniform(-25, 0), r.uniform(-25, 0), w)[None, :]
    if r.random() < .5:
        out = out + cv2.GaussianBlur(np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(2, 9), (h, w)).astype(np.float32), (0, 0), r.uniform(.5, 2))
    out = np.clip(out, 0, 255).astype(np.uint8)
    if r.random() < .6:
        f = r.uniform(.4, .85)
        small = cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=r.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if r.random() < .25:
        out = (cv2.erode if r.random() < .6 else cv2.dilate)(out, np.ones((2, 2), np.uint8))
    if r.random() < .5:
        out = cv2.GaussianBlur(out, (0, 0), r.uniform(.3, 1.1))
    if r.random() < .2:
        out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return out


def degrade(gray, look, r):
    h, w = gray.shape
    rs = np.random.default_rng(r.randrange(1 << 30))
    if look == "screen":  # app screenshot, sometimes re-encoded by a messenger
        out = gray
        if r.random() < .4:
            f = r.uniform(.6, 1)
            small = cv2.resize(out, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif look == "photo":  # phone photo of a screen or slip
        out = gray.astype(np.float32) + np.linspace(r.uniform(-35, 5), r.uniform(-35, 5), w)[None, :] + rs.normal(0, r.uniform(1, 7), (h, w))
        out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), r.uniform(.4, 1.2))
    elif look == "scan":
        out = degrade_scan(gray, r)
    else:  # thermal POS slip: faded, banded print with dropouts
        out = gray.astype(np.float32)
        out = out + np.interp(np.arange(h), [0, h], [r.uniform(-8, 8), r.uniform(-8, 8)])[:, None]
        ink = out < np.median(out) - 25
        out[ink & (rs.random((h, w)) < r.uniform(.02, .16))] = np.median(out)
        out = np.clip(out + rs.normal(0, r.uniform(1, 5), (h, w)), 0, 255).astype(np.uint8)
        if r.random() < .3:
            out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        out = cv2.GaussianBlur(out, (0, 0), r.uniform(.25, .8))
    if r.random() < (.5 if look == "screen" else .7):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, r.randint(35, 95)])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def render(job):
    family, seed = job
    r = random.Random(seed)
    kind, raw = fields.sample_line(r)
    meta = META.get(family, {})
    if not meta.get("persian_forms", True):
        raw = raw.translate(fields.ARABIC_FORMS)
    text = clean(raw.replace("\uE000", " "))
    label = visual_label(text)
    if not 3 <= len(label) <= 240 or set(label) - allowed:
        return None
    look = r.choices(list(LOOKS), list(LOOKS.values()))[0]
    size = r.randint(16, 44)
    weight = r.choice([300, 400, 400, 500, 700])
    mask = draw_line(raw, family, size, weight, r, kind)
    if mask is None:
        return None
    small = cv2.resize(mask, (max(8, mask.shape[1] // SS), max(8, mask.shape[0] // SS)), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    if look == "thermal":
        paper, ink = r.randint(205, 245), r.randint(60, 140)
    elif look in ("screen", "photo") and r.random() < .33:  # dark mode: about a third of the real receipts
        paper, ink = r.randint(15, 70), r.choice([r.randint(215, 255), r.randint(215, 255), r.randint(125, 190)])
    elif look == "screen" and r.random() < .1:  # text on a coloured bar or status pill
        paper, ink = r.randint(90, 200), r.choice([r.randint(240, 255), r.randint(0, 40)])
    else:  # receipts mix black, grey and coloured text; only its luminance survives
        paper, ink = r.randint(232, 255), r.choice([r.randint(0, 45), r.randint(0, 45), r.randint(60, 130), r.randint(125, 170)])
    bg = np.full((h, w), paper, np.float32)
    if r.random() < .1:  # row separator close to the text
        y = r.choice([r.randint(0, 2), h - 1 - r.randint(0, 2)])
        bg[y:y + r.randint(1, 2), :] = paper - r.randint(12, 40)
    gray = np.clip(bg + (ink - bg) * small, 0, 255).astype(np.uint8)
    if r.random() < .06:
        gray = watermark(gray, paper, r)
    ys, xs = np.where(small > .5)
    if len(xs) < 3:
        return None
    if look in ("photo", "scan") and r.random() < .5:  # rotate before measuring so the crop keeps both ends
        m = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-1, 1), 1)
        gray = cv2.warpAffine(gray, m, (w, h), borderValue=paper)
        small = cv2.warpAffine(small, m, (w, h))
        ys, xs = np.where(small > .5)
        if len(xs) < 3:
            return None
    gray = degrade(gray, look, r)
    j = lambda: r.randint(-2, 3)
    bx = (max(0, xs.min() - 2 + j()), max(0, ys.min() - 2 + j()), min(w, xs.max() + 3 + j()), min(h, ys.max() + 3 + j()))
    if bx[2] - bx[0] < 8 or bx[3] - bx[1] < 6:
        return None
    try:
        crop = crop_image(gray, bx)
    except Exception:
        return None
    if not fits(crop, text):
        return None
    data = crop.tobytes()
    split = FAMILIES[family][0]
    # Fields repeat by design (labels, bank names); font families, not texts, separate the splits.
    group = hashlib.sha256(f"receipt/{split}/{family}/{seed}".encode()).hexdigest()
    return {"ink": data, "width": crop.shape[1], "text": text, "visual": label, "source": f"receipt/{kind}", "domain": "receipt",
            "doc_id": family, "group": group, "image_sha256": hashlib.sha256(data).hexdigest(), "look": look}


schema = pa.schema([("ink", pa.binary()), ("width", pa.int32()), ("text", pa.string()), ("visual", pa.string()), ("source", pa.string()),
                    ("domain", pa.string()), ("doc_id", pa.string()), ("group", pa.string()), ("image_sha256", pa.string())])


def main():
    jobs = []
    for i, (family, (split, n)) in enumerate(FAMILIES.items()):
        jobs += [(family, SEED * 1000 + i * 1000000 + k) for k in range(max(1, int(n * SCALE)))]
    random.Random(SEED).shuffle(jobs)
    buffers, shards, counts, rejected, seen = collections.defaultdict(list), [], collections.Counter(), collections.Counter(), set()

    def flush(split):
        if buffers[split]:
            name = f"receipt-{split}-{sum(s['split'] == split for s in shards):04d}.parquet"
            pq.write_table(pa.Table.from_pylist(buffers[split], schema=schema), OUT / name, compression="zstd", compression_level=3, row_group_size=256)
            shards.append({"file": name, "split": split, "rows": len(buffers[split]), "bytes": (OUT / name).stat().st_size})
            buffers[split] = []

    prev = OUT / "previews"
    prev.mkdir(exist_ok=True)
    with mp.Pool(mp.cpu_count()) as pool:
        for n, row in enumerate(pool.imap(render, jobs, chunksize=64)):
            family = jobs[n][0]
            if row is None:
                rejected["render_or_fit"] += 1
                continue
            if row["image_sha256"] in seen:
                rejected["duplicate_image"] += 1
                continue
            seen.add(row["image_sha256"])
            split = FAMILIES[family][0]
            look = row.pop("look")
            buffers[split].append(row)
            counts[f"{split}:{family}"] += 1
            counts[f"{split}/{row['source']}"] += 1
            counts[f"{split}/look:{look}"] += 1
            if n < 60:
                Image.fromarray(np.frombuffer(row["ink"], np.uint8).reshape(32, row["width"])).save(prev / f"{n:03d}-{look}-{family}.png")
            if len(buffers[split]) >= 20000:
                flush(split)
            if (n + 1) % 20000 == 0:
                print("RENDERED", n + 1, "/", len(jobs), flush=True)
    for s in list(buffers):
        flush(s)
    (OUT / "receipt-manifest.json").write_text(json.dumps({"seed": SEED, "families": {f: s for f, (s, _) in FAMILIES.items()}, "looks": LOOKS,
                                                           "line_kinds": fields.LINE_KINDS, "counts": dict(counts), "rejected": dict(rejected), "shards": shards}, indent=1))
    print("DONE", json.dumps({k: v for k, v in counts.items() if "/" in k}), json.dumps(dict(rejected)), flush=True)


if __name__ == "__main__":
    main()
