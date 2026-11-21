"""Full-page receipt test (local CPU): synthetic receipts through the whole Indra pipeline.

Pages use fonts no receipt model trains on: Sahel, Nazli, B Yas and Bassam (held out on Kaggle) plus Tahoma, Arial,
Segoe UI and Times from this machine (never uploaded). Selection never looks at this test.

Scores: value = the field's value was read exactly on some line; pair = the parser matched it to the right field.
Comparison folds ی/ک and digit sets, and ignores spaces.

    PYTHONPATH=vendor/indra:receipts .control-venv/bin/python receipts/eval_pages.py \
        --model v5=v5/results/indra-persian-v5-fast-loose --model receipts=receipts/results/model --out receipts/eval-pages.json
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein

import fields
import indra_ocr
from indra_ocr.rtl import flip, visual_label
from layout import runs
from receipt_ocr import group_rows, parse, pair_rows

ROOT = Path(__file__).resolve().parents[1]
FREE = ROOT / "receipts/fonts-dataset/fonts"
LOCAL = Path("/var/services/chatbot_x/parser_bench/fonts/local")
FONTS = {"tahoma": LOCAL / "tahoma.ttf", "arial": LOCAL / "arial.ttf", "segoeui": LOCAL / "segoeui.ttf", "times": LOCAL / "times.ttf",
         "byas": ROOT / "receipts/fonts384-dataset/fonts/BYas.ttf", "bassam": ROOT / "receipts/fonts384-dataset/fonts/Bassam_MRT.ttf",
         "sahel": FREE / "Sahel.ttf", "nazli": ROOT / "v4/fonts-dataset/fonts/nazli.ttf"}
LAYOUTS = ("rows", "slip", "card")
LOOKS = ("screen", "messenger", "photo")
SEED = 20261117


def text_width(text, font):
    w = 0
    for rtl, s in runs(visual_label(text)):
        w += font.getlength(flip(s, base_rtl=True), direction="rtl", language="fa") if rtl else font.getlength(s, direction="ltr", features=["-liga"])
    return int(w)


def draw_text(d, text, font, fill, y, left=None, right=None):
    x = left if left is not None else right - text_width(text, font)
    for rtl, s in runs(visual_label(text)):
        if rtl:
            s = flip(s, base_rtl=True)
            d.text((x, y), s, font=font, fill=fill, direction="rtl", language="fa")
            x += font.getlength(s, direction="rtl", language="fa")
        else:
            d.text((x, y), s, font=font, fill=fill, direction="ltr", features=["-liga"])
            x += font.getlength(s, direction="ltr", features=["-liga"])


def page(rec, path, layout, r):
    """RGB receipt image. Returns the image."""
    width = {"rows": r.randint(760, 1080), "slip": r.randint(520, 680), "card": r.randint(720, 1000)}[layout]
    body = int(width * (.030 if layout != "slip" else .036) * r.uniform(.9, 1.1))
    f, big = ImageFont.truetype(str(path), body), ImageFont.truetype(str(path), int(body * 1.3))
    row_h = int(body * r.uniform(1.9, 2.3))
    H = int(body * 15 + row_h * (2 * len(rec["rows"]) + 2))
    paper = (255, 255, 255) if layout != "slip" else tuple([r.randint(238, 250)] * 3)
    im = Image.new("RGB", (width, H), paper)
    d = ImageDraw.Draw(im)
    y = int(body * .8)
    if layout == "card":
        color = r.choice([(150, 20, 40), (20, 70, 150), (10, 110, 90), (90, 40, 130)])
        d.rectangle((0, 0, width, int(body * 3)), fill=color)
        draw_text(d, rec["title"], big, (255, 255, 255), int(body * .55), right=width - body)
        y = int(body * 3.6)
    else:  # logo, bank, status, title
        s = int(body * 1.6)
        c = width // 2
        d.polygon([(c, y), (c + s, y + s // 2), (c + s, y + 3 * s // 2), (c, y + 2 * s), (c - s, y + 3 * s // 2), (c - s, y + s // 2)], fill=(200, 30, 50))
        d.ellipse((c - s // 3, y + 2 * s // 3, c + s // 3, y + 4 * s // 3), fill=(245, 170, 60))
        y += 2 * s + body
        draw_text(d, rec["bank"], big, (15, 15, 15), y, right=(width + text_width(rec["bank"], big)) // 2)
        y += int(body * 2)
        if rec["status"]:
            w = text_width(rec["status"], f)
            d.rounded_rectangle((c - w // 2 - body, y - body // 4, c + w // 2 + body, y + int(body * 1.5)), radius=body // 2, fill=(236, 236, 236))
            draw_text(d, rec["status"], f, (10, 120, 30), y, right=c + w // 2)
            y += int(body * 2.2)
        draw_text(d, rec["title"], big, (190, 30, 60), y, right=(width + text_width(rec["title"], big)) // 2)
        y += int(body * 2.4)
    ink, soft = (35, 35, 35), r.choice([(70, 70, 70), (100, 100, 100), (35, 35, 35)])
    for i, (_, label, value) in enumerate(rec["rows"]):
        if layout == "card" and i % 2:
            d.rectangle((0, y - body // 4, width, y + row_h - body // 2), fill=(243, 244, 247))
        lw, vw = text_width(label, f), text_width(value, f)
        if lw + vw + 3 * body > width:  # too wide for one row: the value goes under its label
            draw_text(d, label, f, ink, y, right=width - body)
            y += row_h
            draw_text(d, value, f, soft, y, left=body)
        elif layout == "slip" and r.random() < .4:  # short gap, read as one line
            draw_text(d, f"{label} {value}", f, ink, y, right=width - body)
        else:
            draw_text(d, label, f, ink, y, right=width - body)
            draw_text(d, value, f, soft, y, left=body)
        y += row_h
        if layout == "rows":
            d.line((0, y - body // 2, width, y - body // 2), fill=(226, 226, 226), width=max(1, body // 12))
    if rec["footer"]:
        draw_text(d, rec["footer"], f, (90, 90, 90), y + body // 2, right=(width + text_width(rec["footer"], f)) // 2)
        y += int(body * 2)
    return np.asarray(im.crop((0, 0, width, min(H, y + body))))


def apply_look(a, look, r):
    rs = np.random.default_rng(r.randrange(1 << 30))
    if look == "screen":
        return a
    if look == "messenger":
        f = r.uniform(.72, .95)
        a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, r.randint(45, 82)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    h, w = a.shape[:2]
    m = int(.04 * w)  # phone photo: small perspective, uneven light, blur
    canvas = cv2.copyMakeBorder(a, m, m, m, m, cv2.BORDER_CONSTANT, value=(120, 120, 120))
    H, W = canvas.shape[:2]
    j = lambda: r.uniform(-.012, .012) * W
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[j(), j()], [W + j(), j()], [W + j(), H + j()], [j(), H + j()]])
    out = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(120, 120, 120)).astype(np.float32)
    out += np.linspace(r.uniform(-30, 5), r.uniform(-30, 5), W)[None, :, None] + rs.normal(0, r.uniform(1, 4), out.shape)
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (0, 0), r.uniform(.4, .9))
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, r.randint(60, 88)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def norm(s):
    return "".join(s.translate(fields.FA).translate(str.maketrans({"ي": "ی", "ك": "ک", "٬": ",", "‌": "", "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳",
                                                                  "٤": "۴", "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹"})).split())


def build(n, out):
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    r = random.Random(SEED)
    for font in FONTS:
        for layout in LAYOUTS:
            for look in LOOKS:
                for k in range(n):
                    rec = fields.sample_receipt(r)
                    a = apply_look(page(rec, FONTS[font], layout, r), look, r)
                    name = f"{font}-{layout}-{look}-{k}.jpg"
                    cv2.imwrite(str(out / name), a[:, :, ::-1] if look == "screen" else a, [cv2.IMWRITE_JPEG_QUALITY, 98])
                    pages.append({"file": name, "font": font, "layout": layout, "look": look, "rec": rec})
    return pages


def evaluate(model_dir, pages, out, threads):
    conv = indra_ocr.load(path=str(model_dir), threads=threads, mode="ocr")
    res = []
    for p in pages:
        _, pg = next(iter(conv.convert(out / p["file"])))
        parsed = parse(pg)
        line_texts = {norm(l.text) for l in pg.lines}
        all_text = norm(" ".join(l.text for l in pg.lines))
        got = {}
        for row in parsed["rows"]:
            if row["key"] and row["key"] not in got:
                got[row["key"]] = norm(row["value"])
        items = []
        for key, label, value in p["rec"]["rows"]:
            v = norm(value)
            items.append({"key": key, "value": v in line_texts or v in all_text, "pair": got.get(key) == v})
        ref = norm(" ".join(f"{l} {v}" for _, l, v in p["rec"]["rows"]))
        rows_text = norm(" ".join(f"{a} {b}" for a, b, _ in pair_rows(group_rows(pg.lines))[0]))
        res.append({**{k: p[k] for k in ("file", "font", "layout", "look")}, "items": items, "cer": Levenshtein.distance(ref, rows_text) / max(1, len(ref)),
                    "text": [l.text for l in pg.lines]})
    return res


def summarize(res):
    def rate(rs, field, keep=lambda i: True):
        xs = [i[field] for r in rs for i in r["items"] if keep(i)]
        return {"rate": sum(xs) / max(1, len(xs)), "n": len(xs)}
    out = {"value": rate(res, "value"), "pair": rate(res, "pair"), "cer": sum(r["cer"] for r in res) / max(1, len(res)), "pages": len(res),
           "perfect_pages": sum(all(i["pair"] for i in r["items"]) for r in res) / max(1, len(res))}
    for dim in ("font", "layout", "look"):
        out[dim] = {v: {"value": rate([r for r in res if r[dim] == v], "value")["rate"], "pair": rate([r for r in res if r[dim] == v], "pair")["rate"]}
                    for v in sorted({r[dim] for r in res})}
    keys = sorted({i["key"] for r in res for i in r["items"]})
    out["key"] = {k: {"value": rate(res, "value", lambda i: i["key"] == k)["rate"], "pair": rate(res, "pair", lambda i: i["key"] == k)["rate"]} for k in keys}
    return out


def real_receipts(model_dir, threads):
    """Hand-transcribed real receipts kept on this machine only."""
    conv = indra_ocr.load(path=str(model_dir), threads=threads, mode="ocr")
    rows = []
    for gt_path in sorted((ROOT / "receipts/local-eval").glob("*.gt.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))["lines"]
        image = gt_path.with_name(gt_path.name.replace(".gt.json", ".jpg"))
        _, pg = next(iter(conv.convert(image)))
        pred = [norm(l.text) for l in pg.lines]
        rows.append({"file": image.name, "exact_lines": sum(norm(g) in pred for g in gt), "lines": len(gt),
                     "cer": Levenshtein.distance(norm("".join(sorted(gt))), norm("".join(sorted(pred)))) / max(1, len(norm("".join(gt))))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages-per-combo", type=int, default=3)
    ap.add_argument("--work", default=str(ROOT / "receipts/local-eval/pages"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    work = Path(a.work)
    pages = build(a.pages_per_combo, work)
    print("pages", len(pages), file=sys.stderr, flush=True)
    report, detail = {"pages": len(pages)}, {}
    for m in a.model:
        name, d = m.split("=", 1)
        res = evaluate(d, pages, work, a.threads)
        report[name] = summarize(res)
        report[name]["real"] = real_receipts(d, a.threads)
        detail[name] = res
        s = report[name]
        print(f"{name:14s} value {s['value']['rate']:.3f} pair {s['pair']['rate']:.3f} cer {s['cer']:.3f} perfect-pages {s['perfect_pages']:.3f} real {s['real']}", flush=True)
    Path(a.out).write_text(json.dumps({"summary": report, "detail": detail}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
