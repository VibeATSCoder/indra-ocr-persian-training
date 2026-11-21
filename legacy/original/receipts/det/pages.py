"""Whole synthetic receipt pages for detector training (Kaggle CPU, no GPU quota).

Layouts follow the real receipts: status bars, bank headers, status pills, label and value rows, stacked rows, cards,
dark mode, grey labels, buttons and small notes. Font families keep their split. Line boxes come later from the original
detector on the clean page, so these pages only need to look right.

    KAGGLE_INPUT=<fonts> KAGGLE_OUT=<out> python pages.py
"""
import json
import multiprocessing as mp
import os
import random
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

import fields
from indra_ocr.rtl import flip, visual_label
from layout import runs

SEED = 20261119
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("PAGES_TMP", "/tmp/det-pages"))
TRAIN_PAGES = int(os.environ.get("TRAIN_PAGES", 4000))
VALID_PAGES = int(os.environ.get("VALID_PAGES", 240))
TINTS = [(20, 90, 170), (150, 25, 45), (10, 120, 90), (95, 45, 140), (205, 110, 20), (25, 130, 170), (60, 60, 70)]


def text_width(text, font):
    w = 0
    for rtl, s in runs(visual_label(text)):
        w += font.getlength(flip(s, base_rtl=True), direction="rtl", language="fa") if rtl else font.getlength(s, direction="ltr", features=["-liga"])
    return int(w)


def draw_text(d, text, font, fill, y, left=None, right=None, center=None):
    w = text_width(text, font)
    x = left if left is not None else right - w if right is not None else center - w // 2
    for rtl, s in runs(visual_label(text)):
        if rtl:
            s = flip(s, base_rtl=True)
            d.text((x, y), s, font=font, fill=fill, direction="rtl", language="fa")
            x += font.getlength(s, direction="rtl", language="fa")
        else:
            d.text((x, y), s, font=font, fill=fill, direction="ltr", features=["-liga"])
            x += font.getlength(s, direction="ltr", features=["-liga"])
    return w


def theme(r):
    mode = r.choices(["light", "dark", "tint"], [.58, .3, .12])[0]
    accent = r.choice(TINTS)
    grey = lambda a, b: (lambda v: (v, v, v))(r.randint(a, b))
    if mode == "dark":  # about a third of the real receipts
        page, card, ink = grey(10, 45), grey(28, 70), grey(215, 255)
        soft, line = r.choice([grey(120, 190), ink]), grey(55, 90)
        accent = tuple(min(255, c + 90) for c in accent)
    elif mode == "tint":
        base = r.choice(TINTS)
        page = tuple(235 + (c * 20) // 255 for c in base)
        card, ink, soft, line = (255, 255, 255), grey(10, 50), grey(70, 140), grey(205, 230)
    else:
        page, card, ink = grey(238, 255), grey(248, 255), grey(0, 60)
        soft, line = r.choice([grey(120, 175), grey(70, 120), ink]), grey(210, 236)
    return {"mode": mode, "page": page, "card": card, "ink": ink, "soft": soft, "line": line, "accent": accent,
            "on_accent": (255, 255, 255) if sum(accent) < 480 else (10, 10, 10)}


def status_bar(d, W, body, t, r, st):
    size = ImageFont.load_default(max(9, int(body * .8)))
    clock = f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}"
    d.text((int(W * .04), int(body * .3)), clock, font=size, fill=t["ink"])
    x = W - int(W * .05)
    for _ in range(r.randint(2, 4)):
        s = int(body * .7)
        d.rounded_rectangle((x - s, int(body * .4), x, int(body * .4) + s // 2 + 2), radius=2, outline=t["ink"], fill=t["ink"] if r.random() < .5 else None)
        x -= int(s * 1.6)


def logo(d, cx, y, s, t, r):
    color = r.choice(TINTS)
    if r.random() < .5:
        d.ellipse((cx - s, y, cx + s, y + 2 * s), fill=color)
    else:
        d.polygon([(cx, y), (cx + s, y + s // 2), (cx + s, y + 3 * s // 2), (cx, y + 2 * s), (cx - s, y + 3 * s // 2), (cx - s, y + s // 2)], fill=color)
    d.ellipse((cx - s // 3, y + 2 * s // 3, cx + s // 3, y + 4 * s // 3), fill=(245, 170, 60))
    return 2 * s


def dotted(d, x0, x1, y, fill, step):
    for x in range(x0, x1, max(3, step)):
        d.ellipse((x - 1, y - 1, x + 1, y + 1), fill=fill)


def row(d, label, value, kind, x0, x1, y, f, fv, t, body, r, row_h):
    """One label and value. Returns the height it used."""
    lw, vw = text_width(label, f), text_width(value, fv)
    if lw + vw + 2 * body > x1 - x0 and kind != "stacked":
        kind = "stacked"
    if kind == "stacked":
        draw_text(d, label, f, t["soft"], y, right=x1)
        draw_text(d, value, fv, t["ink"], y + int(body * 1.5), right=x1 if r.random() < .7 else x0 + vw)
        return int(body * 1.5) + row_h
    if kind == "inline":
        draw_text(d, f"{label} {value}", f, t["ink"], y, right=x1)
    elif kind == "reverse":
        draw_text(d, label, f, t["soft"], y, left=x0)
        draw_text(d, value, fv, t["ink"], y, right=x1)
    else:
        draw_text(d, label, f, t["soft"], y, right=x1)
        draw_text(d, value, fv, t["ink"], y, left=x0)
        if kind == "leader":
            dotted(d, x0 + vw + body // 2, x1 - lw - body // 2, y + int(body * .8), t["soft"], int(body * r.uniform(.25, .5)))
    return row_h


def page(font_path, r, persian_forms=True):
    """A BGR receipt page, as cv2 reads images."""
    W = r.choice([r.randint(480, 620), r.randint(620, 820), r.randint(820, 1200)])
    body = int(W * r.choice([r.uniform(.026, .034), r.uniform(.034, .046), r.uniform(.046, .058)]))
    f = ImageFont.truetype(str(font_path), body)
    fv = ImageFont.truetype(str(font_path), int(body * r.choice([1, 1, 1.08, .92])))
    big = ImageFont.truetype(str(font_path), int(body * r.uniform(1.2, 1.7)))
    small = ImageFont.truetype(str(font_path), max(9, int(body * r.uniform(.72, .9))))
    t = theme(r)
    rec = fields.sample_receipt(r)
    st = fields.Style(r)
    T = (lambda s: s) if persian_forms else (lambda s: s.translate(fields.ARABIC_FORMS))
    H = int(body * 60 + body * 7 * len(rec["rows"]))
    im = Image.new("RGB", (W, H), t["page"])
    d = ImageDraw.Draw(im)
    y = 0
    if r.random() < .4:
        status_bar(d, W, body, t, r, st)
        y += int(body * 1.6)
    if r.random() < .5:  # app bar
        h = int(body * r.uniform(2.6, 3.4))
        d.rectangle((0, y, W, y + h), fill=t["accent"])
        draw_text(d, T(rec["title"]), big if r.random() < .5 else f, t["on_accent"], y + int(h * .22), center=W // 2)
        y += h
    y += int(body * r.uniform(.6, 1.6))
    if r.random() < .6:
        y += logo(d, W // 2, y, int(body * r.uniform(1, 1.8)), t, r) + int(body * .6)
        draw_text(d, T(rec["bank"]), f if r.random() < .5 else big, t["ink"], y, center=W // 2)
        y += int(body * 2.2)
    if rec["status"] and r.random() < .7:
        w = text_width(T(rec["status"]), f)
        if r.random() < .5:
            d.rounded_rectangle((W // 2 - w // 2 - body, y - body // 4, W // 2 + w // 2 + body, y + int(body * 1.5)), radius=body // 2, fill=(236, 236, 236) if t["mode"] != "dark" else (50, 70, 55))
        draw_text(d, T(rec["status"]), f, (10, 130, 40) if t["mode"] != "dark" else (110, 225, 130), y, center=W // 2)
        y += int(body * 2.2)
    amount = next((v for k, _, v in rec["rows"] if k == "amount"), None)
    if amount and r.random() < .4:
        draw_text(d, T(amount), big, t["ink"], y, center=W // 2)
        y += int(body * 2.6)
    margin = int(W * r.uniform(.03, .08))
    x0, x1 = margin, W - margin
    on_card = r.random() < .6
    card_top = y
    pad = int(body * r.uniform(.6, 1.2)) if on_card else 0
    if on_card:
        y += pad
        layer = Image.new("RGB", (W, H), t["card"])  # the card's height is known only after its rows: draw them on their own layer
        d2 = ImageDraw.Draw(layer)
    else:
        layer, d2 = im, d
    kind = r.choices(["sides", "stacked", "inline", "reverse", "leader"], [.55, .15, .1, .08, .12])[0]
    row_h = int(body * r.uniform(1.8, 2.8))
    rule = r.choice(["none", "none", "line", "dash", "zebra"])
    for i, (_, label, value) in enumerate(rec["rows"]):
        if rule == "zebra" and i % 2:
            d2.rectangle((x0 - pad, y - body // 4, x1 + pad, y + row_h - body // 2), fill=t["line"])
        y += row(d2, T(label), T(value), kind, x0 + pad, x1 - pad, y, f, fv, t, body, r, row_h)
        if rule in ("line", "dash"):
            sep = y - body // 2
            for x in range(x0, x1, body // 2 if rule == "dash" else x1 - x0):
                d2.line((x, sep, min(x1, x + (body // 4 if rule == "dash" else x1 - x0)), sep), fill=t["line"], width=max(1, body // 14))
    if on_card:
        y += pad
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((x0 - pad, card_top, x1 + pad, y), radius=int(body * r.uniform(.3, 1.2)), fill=255)
        im.paste(layer, (0, 0), mask)
        d = ImageDraw.Draw(im)
    y += int(body * r.uniform(.8, 1.6))
    if r.random() < .4:  # small notice, wrapped by hand into short lines
        for _ in range(r.randint(1, 3)):
            s = T(r.choice(fields.SENTENCES))
            if text_width(s, small) < x1 - x0:
                draw_text(d, s, small, t["soft"], y, right=x1)
                y += int(body * 1.3)
    if rec["footer"] and r.random() < .6:
        draw_text(d, T(rec["footer"]), small, t["soft"], y, center=W // 2)
        y += int(body * 1.6)
    if r.random() < .5:  # buttons
        y += body // 2
        names = r.sample(fields.BUTTONS, r.randint(1, 2))
        w = (x1 - x0 - body * (len(names) - 1)) // len(names)
        for k, name in enumerate(names):
            a = x0 + k * (w + body)
            solid = r.random() < .6
            d.rounded_rectangle((a, y, a + w, y + int(body * 2.4)), radius=body // 2, fill=t["accent"] if solid else None, outline=t["accent"])
            if text_width(T(name), f) < w:
                draw_text(d, T(name), f, t["on_accent"] if solid else t["accent"], y + int(body * .5), center=a + w // 2)
        y += int(body * 3)
    y += int(body * r.uniform(.6, 3))
    im = im.crop((0, 0, W, min(H, y)))
    if r.random() < .8:  # phone screens are tall: the real receipts' median is 2.06 times their width
        tall = int(W * r.uniform(1.4, 2.25))
        if tall > im.height:
            screen = Image.new("RGB", (W, tall), t["page"])
            screen.paste(im, (0, int((tall - im.height) * r.choice([0, 0, .3, .5]))))
            im = screen
    a = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    if r.random() < .1:  # faint diagonal stamp over the page
        h, w = a.shape[:2]
        mark = Image.new("L", (w, h), 0)
        draw_text(ImageDraw.Draw(mark), r.choice(["رسید معتبر", "تراکنش موفق", "نسخه مشتری"]), ImageFont.truetype(str(font_path), int(body * r.uniform(2.5, 5))), 255, h // 2, center=w // 2)
        m = cv2.warpAffine(np.asarray(mark, np.float32) / 255, cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-45, 45), 1), (w, h))
        a = np.clip(a.astype(np.float32) - r.uniform(8, 30) * (1 if t["mode"] != "dark" else -1) * m[..., None], 0, 255).astype(np.uint8)
    return a


def render(job):
    i, split, family, path, forms = job
    r = random.Random(SEED + i)
    name = f"{split}-{i:05d}.png"
    try:
        a = page(path, r, forms)
    except OSError:  # a font PIL cannot use at this size
        return None
    cv2.imwrite(str(TMP / name), a, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"file": name, "split": split, "font": family}


def main():
    assert features.check_feature("raqm"), "Arabic shaping (raqm) required"
    cv2.setNumThreads(1)
    TMP.mkdir(parents=True, exist_ok=True)
    fonts = {p.stem.lower(): p for p in INPUT.rglob("*") if p.suffix.lower() == ".ttf"}
    meta = {k.lower(): v for k, v in json.loads(next(INPUT.rglob("fonts-meta.json")).read_text(encoding="utf8")).items()}
    by = {s: sorted(f for f, m in meta.items() if m["split"] == s and f in fonts) for s in ("train", "validation")}
    print("FAMILIES", {s: len(v) for s, v in by.items()}, flush=True)
    assert not set(by["train"]) & set(by["validation"])
    r = random.Random(SEED)
    jobs = [(i, "train", f, fonts[f], meta[f].get("persian_forms", True)) for i, f in enumerate(r.choices(by["train"], k=TRAIN_PAGES))]
    jobs += [(TRAIN_PAGES + i, "validation", f, fonts[f], meta[f].get("persian_forms", True)) for i, f in enumerate(r.choices(by["validation"], k=VALID_PAGES))]
    rows = []
    with mp.Pool(mp.cpu_count()) as pool:
        for n, row in enumerate(pool.imap(render, jobs, chunksize=16), 1):
            if row:
                rows.append(row)
            if n % 500 == 0:
                print("RENDERED", n, "/", len(jobs), flush=True)
    with tarfile.open(OUT / "synthetic-pages.tar", "w") as tar:  # one output file: thousands of them are awkward to attach to a kernel
        for row in rows:
            tar.add(TMP / row["file"], arcname=row["file"])
    (OUT / "synthetic-pages.json").write_text(json.dumps(rows), encoding="utf8")
    print("DONE", len(rows), "of", len(jobs), round((OUT / "synthetic-pages.tar").stat().st_size / 1e6), "MB", flush=True)


if __name__ == "__main__":
    main()
