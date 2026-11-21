"""Measure a classical card finder on MIDV-UP validation views with known corners. Validation cards only.

    PYTHONPATH=receipts/ids .control-venv/bin/python receipts/ids/find_card.py
"""
import collections
import json
import sys

import cv2
import numpy as np

import midv

CARD_RATIO = 1.585


def card_candidates(bgr):
    """Card-shaped regions of a page, best first: (score, four corners in page pixels)."""
    h, w = bgr.shape[:2]
    s = min(1.0, 800 / max(h, w))
    small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    area_all = small.shape[0] * small.shape[1]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    masks = []
    for ch in (lab[..., 0], lab[..., 1], lab[..., 2]):
        blur = cv2.GaussianBlur(ch, (7, 7), 0)
        _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks += [m, 255 - m]
    gray = cv2.GaussianBlur(lab[..., 0], (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 30, 90), np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    masks.append(closed)
    out = []
    for m in masks:
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < .02 * area_all or area > .97 * area_all:
                continue
            rect = cv2.minAreaRect(c)
            a, b = rect[1]
            if min(a, b) < 10:
                continue
            ratio, fill = max(a, b) / min(a, b), area / (a * b)
            if fill < .8 or not 1.3 <= ratio <= 1.9:
                continue
            out.append((area / area_all * fill ** 2 * (1 - abs(ratio - CARD_RATIO)), cv2.boxPoints(rect) / s))
    return sorted(out, key=lambda x: -x[0])


def quad_iou(a, b, size):
    ma, mb = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
    cv2.fillConvexPoly(ma, np.int32(a), 1)
    cv2.fillConvexPoly(mb, np.int32(b), 1)
    return (ma & mb).sum() / max(1, (ma | mb).sum())


def main():
    from remote_zip import open_remote

    z = open_remote(midv.URL, midv.SIZE)
    root = f"{midv.ROOT}"
    meta = {k: json.load(z.open(f"{root}/{k}")) for k in ("photo/annotations/irn_id_iphone.json", "photo/annotations/irn_id_samsung.json", "left_scans/annotations/irn_id.json", "center_scans/annotations/irn_id.json")}
    results = collections.defaultdict(list)
    for card in sorted(midv.VALIDATION):
        cam = midv.camera_of(card)
        for view, ann, name in (("photo", f"photo/annotations/irn_id_{cam}.json", f"{card}.jpg"), ("left_scan", "left_scans/annotations/irn_id.json", f"{card}.png"), ("center_scan", "center_scans/annotations/irn_id.json", f"{card}.png")):
            path = midv.LOCAL / "images" / f"{card}-{view}-{name}"
            if not path.exists():
                continue
            regions = [r for r in midv.regions(meta[ann], name) if r["region_attributes"].get("field_name") == "doc_quad"]
            if not regions:
                continue
            q = np.float32(list(zip(regions[0]["shape_attributes"]["all_points_x"], regions[0]["shape_attributes"]["all_points_y"])))
            img = cv2.imread(str(path))
            s = 1000 / max(img.shape[:2])
            cands = card_candidates(img)
            iou = quad_iou(q * s, cands[0][1] * s, (round(img.shape[0] * s), round(img.shape[1] * s))) if cands else 0.0
            results[view].append(iou)
    for view, xs in results.items():
        xs = np.array(xs)
        print(f"{view:12s} images {len(xs):3d}  median IoU {np.median(xs):.3f}  IoU>=.9 {100 * (xs >= .9).mean():5.1f}%  IoU>=.8 {100 * (xs >= .8).mean():5.1f}%  IoU>=.5 {100 * (xs >= .5).mean():5.1f}%  none {100 * (xs == 0).mean():5.1f}%")


if __name__ == "__main__":
    main()
