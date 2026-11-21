"""Screen a font folder for receipt rendering. A font is usable only if a person could read Persian receipt
digits and letters in it and its glyphs match the labels the renderer would give them.

    .control-venv/bin/python receipts/screen_fonts.py <font-dir> <out.json>
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont

LETTERS = "ابپتثجچحخدذرزژسشصضطظعغفقلمنوه"  # plus ی/ک or ي/ك, checked separately
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"
PUNCT = ",/:*-."
DECOR = re.compile(r"outline|shadow|3d|shade|besmellah|animal|ornament|symbol|ding|tazhib|border|icon|logo|nastaliq|nasta|shekasteh|callig|hollow|contour|stencil|inline|sport|graff", re.I)
WEIGHT = re.compile(r"(?:[ _-]?(?:ultra|extra)?[ _-]?(?:bold|black|light|medium|mediom|heavy|thin|regular|italic|oblique)|bd|bold|blk|lt|it|_?mrt\d*|\(mrt\))+$", re.I)
FAMILY_CLEAN = re.compile(r"[ _-]+")


def bitmap(font, text):
    m = font.getmask(text, mode="L")
    a = np.asarray(m, dtype=np.uint8).reshape(m.size[1], m.size[0])
    return a


def square(a, side=48):
    """A glyph scaled into a side x side box, for comparing shapes regardless of size."""
    if a.size == 0:
        return np.zeros((side, side), np.uint8)
    h, w = a.shape
    s = (side - 4) / max(h, w)
    small = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side), np.uint8)
    y, x = (side - small.shape[0]) // 2, (side - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def iou(a, b):
    x, y = a > 110, b > 110
    return (x & y).sum() / max(1, (x | y).sum())


def topology(a):
    """(ink components, holes) of a bitmap."""
    b = (a > 110).astype(np.uint8)
    comps = cv2.connectedComponents(b, connectivity=8)[0] - 1
    holes = cv2.connectedComponents(cv2.copyMakeBorder(1 - b, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=1), connectivity=4)[0] - 2
    return comps, holes


def family_of(stem):
    s = WEIGHT.sub("", stem.strip())
    return FAMILY_CLEAN.sub("", s).lower() or stem.lower()


def screen(path):
    r = {"file": path.name, "family": family_of(path.stem), "reasons": []}
    try:
        tt = TTFont(str(path), fontNumber=0, lazy=True)
        cmap = tt.getBestCmap() or {}
        f = ImageFont.truetype(str(path), 120)
    except Exception as e:
        r["reasons"].append(f"unreadable: {type(e).__name__}")
        return r
    has = lambda s: all(ord(c) in cmap for c in s)
    if DECOR.search(path.stem):
        r["reasons"].append("decorative name")
    if not has(LETTERS):
        r["reasons"].append("letters missing (old non-Unicode encoding or not Arabic script)")
        return r
    r["persian_forms"] = has("یک")
    if not r["persian_forms"] and not has("يك"):
        r["reasons"].append("no ی/ک or ي/ك")
    source = PERSIAN_DIGITS if has(PERSIAN_DIGITS) else ARABIC_DIGITS if has(ARABIC_DIGITS) else None
    if source is None:
        r["reasons"].append("no Persian or Arabic-Indic digits")
        return r
    r["digit_codepoints"] = "persian" if source == PERSIAN_DIGITS else "arabic"
    glyphs = {p: square(bitmap(f, s)) for p, s in zip(PERSIAN_DIGITS, source)}
    if any((g > 110).sum() < 10 for g in glyphs.values()):
        r["reasons"].append("empty digit glyph")
        return r
    tofu = square(bitmap(f, "͸"))
    if sum(iou(g, tofu) > .85 for g in glyphs.values()) >= 3:
        r["reasons"].append("digits render as empty boxes")
    # Digit pairs a Persian reader tells apart must look different. ۴ drawn like ۳ teaches the model wrong labels.
    pairs = [("۲", "۳"), ("۳", "۴"), ("۲", "۴"), ("۶", "۴"), ("۷", "۸"), ("۱", "۹"), ("۰", "۵"), ("۱", "۲"), ("۹", "۳"), ("۶", "۹")]
    close = [a + b for a, b in pairs if iou(glyphs[a], glyphs[b]) > .72]
    if close:
        r["reasons"].append("digits look alike: " + " ".join(close))
    if len({g.tobytes() for g in glyphs.values()}) < 10:
        r["reasons"].append("two digits are the same glyph")
    # Hollow, 3D and ornamental faces: a plain alef or lam is one solid stroke.
    for c in "ال":
        comps, holes = topology(bitmap(f, c))
        if holes > 0 or comps > 1:
            r["reasons"].append("hollow or ornamental strokes")
            break
    # Shaping: letters must join.
    if f.getlength("ببببب", direction="rtl", language="fa") >= .985 * 5 * f.getlength("ب", direction="rtl", language="fa"):
        joined, iso = bitmap(f, "بب"), bitmap(f, "ب")
        if joined.shape[1] >= 1.9 * iso.shape[1]:
            r["reasons"].append("letters do not join")
    # Many Persian fonts draw ASCII digits with Persian shapes: then "12" must not be labelled as Latin.
    latin = has(LATIN_DIGITS) and has("AXIR")
    r["latin_digits_are_persian"] = bool(has(LATIN_DIGITS) and sum(iou(square(bitmap(f, a)), glyphs[b]) > .6 for a, b in zip(LATIN_DIGITS, PERSIAN_DIGITS)) >= 6)
    if latin and not r["latin_digits_are_persian"]:
        a = square(bitmap(f, "A"))
        latin = (a > 110).sum() > 10 and iou(a, tofu) < .85
    r["latin_ok"] = bool(latin and not r["latin_digits_are_persian"])
    r["missing_punct"] = "".join(c for c in PUNCT if ord(c) not in cmap)
    ref = "سلام ۱۲۳۴۵۶۷۸۹۰ گچژپ"
    r["hash"] = hashlib.sha256(square(bitmap(f, ref), 64).tobytes()).hexdigest()[:16]
    return r


def main():
    fonts = sorted(p for p in Path(sys.argv[1]).iterdir() if p.suffix.lower() in (".ttf", ".otf"))
    results = [screen(p) for p in fonts]
    seen = {}
    for r in results:
        if not r["reasons"]:
            if r["hash"] in seen:
                r["reasons"].append("same glyphs as " + seen[r["hash"]])
            else:
                seen[r["hash"]] = r["file"]
    ok = [r for r in results if not r["reasons"]]
    print("fonts", len(results), "usable", len(ok), "families", len({r["family"] for r in ok}))
    why = collections.Counter(re.sub(r":.*|as .*", "", x) for r in results for x in r["reasons"])
    print(why.most_common())
    print("latin ok", sum(r["latin_ok"] for r in ok), "| latin digits drawn Persian", sum(r["latin_digits_are_persian"] for r in ok), "| arabic digit codepoints", sum(r["digit_codepoints"] == "arabic" for r in ok), "| only ي/ك", sum(not r["persian_forms"] for r in ok))
    Path(sys.argv[2]).write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf8")


if __name__ == "__main__":
    main()
