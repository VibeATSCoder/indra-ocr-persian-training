"""Measure approved source styles; export aggregate features, never source text."""

import argparse, collections, hashlib, json, re
from pathlib import Path
import numpy as np
from PIL import Image
from style_specs import FAMILIES

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def fingerprint_value(value):
    value = str(value).translate(DIGITS).translate(str.maketrans({"ي": "ی", "ك": "ک"}))
    value = re.sub(r"[^0-9a-zA-Z\u0600-\u06ff*]", "", value).lower()
    return hashlib.sha256(value.encode()).hexdigest() if len(value) >= 4 else None


def audit(data, out, private_guard=None):
    if "receipts-eval" in str(data):
        raise ValueError("Use approved dev76 inputs; the evaluation set stays sealed")
    files = {
        p.stem: p
        for p in (data / "images").iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    }
    expected = {f"t-cs{i:02d}" for v in FAMILIES.values() for i in v["sources"]}
    assert set(files) == expected, {
        "missing": sorted(expected - set(files)),
        "unclassified": sorted(set(files) - expected),
    }
    drafts_path = data.parents[0] / "drafts.json"
    drafts = json.loads(drafts_path.read_text()) if drafts_path.exists() else {}
    reports = {}
    private = set()
    banks = collections.Counter()
    for name, definition in FAMILIES.items():
        sizes = []
        colors = collections.Counter()
        heights = []
        dark = []
        image_hashes = []
        for number in definition["sources"]:
            key = f"t-cs{number:02d}"
            path = files[key]
            im = Image.open(path).convert("RGB")
            w, h = im.size
            sizes.append((w, h))
            small = np.asarray(im.resize((96, 160)))
            dark.append(float((small.mean(-1) < 80).mean()))
            quant = (small // 24 * 24 + 12).clip(0, 255).reshape(-1, 3)
            colors.update(map(tuple, quant.tolist()))
            image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            for row in drafts.get(f"{number}.jpg", drafts.get(f"{number}.png", [])):
                x0, y0, x1, y1 = row["box"]
                if 5 < y1 - y0 < 0.08 * w:
                    heights.append((y1 - y0) / w)
            label = json.loads((data / "labels" / (key + ".json")).read_text())["record"]
            banks[label.get("bank") or "unspecified"] += 1
            values = []
            for party in ["source", "destination"]:
                item = label.get(party) or {}
                values.extend(item.get(k) for k in ["name", "card_pan", "account_number", "iban"])
            values.extend(
                label.get(k)
                for k in [
                    "tracking_number",
                    "reference_number",
                    "transaction_id",
                    "document_number",
                    "payment_id",
                ]
            )
            private.update(v for value in values if value and (v := fingerprint_value(value)))
        reports[name] = {
            **definition,
            "source_count": len(sizes),
            "source_ids": [f"t-cs{i:02d}" for i in definition["sources"]],
            "median_width": int(np.median([w for w, h in sizes])),
            "median_height": int(np.median([h for w, h in sizes])),
            "median_aspect_ratio": float(np.median([h / w for w, h in sizes])),
            "median_dark_fraction": float(np.median(dark)),
            "body_ink_height_fraction": float(np.median(heights)) if heights else 0.024,
            "dominant_rgb": [list(rgb) for rgb, n in colors.most_common(6)],
            "source_image_hashes": image_hashes,
        }
    report = {
        "approved_images": len(files),
        "style_families": len(reports),
        "bank_counts": dict(banks),
        "families": reports,
        "derivation": "Manual visual grouping plus measured aspect ratios, ink heights and color distributions. Vector recipes recreate the styles; source rasters and customer text are not embedded.",
        "sealed_evaluation_used": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf8")
    if private_guard:
        private_guard.write_text(json.dumps({"normalized_value_sha256": sorted(private)}))
    print(
        json.dumps(
            {
                "approved_images": len(files),
                "families": len(reports),
                "private_value_fingerprints": len(private),
            }
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--private-guard", type=Path)
    a = p.parse_args()
    audit(a.data, a.out, a.private_guard)
