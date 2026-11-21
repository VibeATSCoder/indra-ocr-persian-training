"""Verify generated receipt assets, field truth, transformations and replay."""

import argparse, collections, hashlib, json, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from render import ROOT, render_background
from audit_dataset import DIGITS, fingerprint_value
from make_dataset import PRIVATE_KEYS


def checksums(record):
    for key in ["from_card", "to_card"]:
        text = record[key].translate(DIGITS)
        if any(c in text for c in "*•xX"):
            continue
        digits = [int(c) for c in re.sub(r"\D", "", text)]
        assert len(digits) == 16
        total = sum(
            (2 * v - 9 if 2 * v > 9 else 2 * v) if i % 2 else v
            for i, v in enumerate(reversed(digits))
        )
        assert total % 10 == 0
    for key in ["from_iban", "to_iban"]:
        text = re.sub(r"\s", "", record[key].translate(DIGITS))
        assert text.startswith("IR") and len(text) == 26
        assert int(text[4:] + "1827" + text[2:4]) % 97 == 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--custom", type=Path, required=True)
    p.add_argument("--private-guard", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((ROOT / "dataset_audit.json").read_text())
    assert (
        audit["approved_images"] == 76
        and audit["style_families"] == 25
        and not audit["sealed_evaluation_used"]
    )
    forbidden = set(json.loads(a.private_guard.read_text())["normalized_value_sha256"])
    specs = {
        q.stem: json.loads(q.read_text(encoding="utf8"))
        for q in (ROOT / "templates").glob("*.json")
        if not q.name.endswith(".gt.json")
    }
    assert len(specs) == 25
    for name, spec in specs.items():
        np.testing.assert_array_equal(
            np.asarray(render_background(spec)),
            np.asarray(Image.open(ROOT / "templates" / spec["base_image"])),
        )
        checksums(spec["example_record"])
        assert not any(
            fingerprint_value(spec["example_record"][key]) in forbidden for key in PRIVATE_KEYS
        )
    rows = [
        json.loads(s) for s in (a.dataset / "labels.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert len(rows) == 250 and len({r["family"] for r in rows}) == 25
    count = 0
    overlaps = []
    for row in rows:
        assert (
            row["synthetic_pixels"]
            and row["fully_synthetic"]
            and row["content_origin"] == "seeded_generator"
        )
        image = Image.open(a.dataset / row["file"])
        assert image.size == (row["width"], row["height"])
        checksums(row["record"])
        assert not any(fingerprint_value(row["record"][key]) in forbidden for key in PRIVATE_KEYS)
        slot_lines = collections.defaultdict(list)
        H = np.asarray(row["homography"])
        for line in row["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            assert 0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height
            b = line["source_bbox"]
            q = (
                np.asarray([[b[0], b[1], 1], [b[2], b[1], 1], [b[2], b[3], 1], [b[0], b[3], 1]])
                @ H.T
            )
            np.testing.assert_allclose(q[:, :2] / q[:, 2:], line["polygon"], atol=0.00051, rtol=0)
            slot_lines[line["slot"]].append(line["text"])
            count += 1
        for slot in specs[row["family"]]["slots"]:
            expected = (
                slot.get("prefix", "")
                + str(row["record"].get(slot["field"], ""))
                + slot.get("suffix", "")
                if slot["role"] == "field"
                else slot["text"]
            )
            assert " ".join(" ".join(slot_lines[slot["id"]]).split()) == " ".join(expected.split())
        if row["level"] == "clean":
            # Test untransformed boxes: transformed axis-aligned bounds can intersect after rotation.
            for i, line in enumerate(row["lines"]):
                b = line["source_bbox"]
                for other in row["lines"][i + 1 :]:
                    c = other["source_bbox"]
                    area = max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(
                        0, min(b[3], c[3]) - max(b[1], c[1])
                    )
                    if area > 4:
                        overlaps.append([row["family"], line["slot"], other["slot"], area])
    assert not overlaps, overlaps[:10]
    groups = collections.Counter(r["document_group"] for r in rows)
    assert len(groups) == 50 and set(groups.values()) == {5}
    custom = [
        json.loads(s) for s in (a.custom / "labels.jsonl").read_text(encoding="utf8").splitlines()
    ]
    for row in custom:
        assert not row["fully_synthetic"] and row["content_origin"] == "user_records"
        for line in row["lines"]:
            if line["field"] in ["amount", "sender", "receiver", "tracking"]:
                assert line["text"] == row["record"][line["field"]]
    repeat = Path(tempfile.mkdtemp(prefix="receipt-style-replay-"))
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "make_dataset.py"),
            "--count",
            "25",
            "--out",
            str(repeat),
            "--rotate",
            "6",
            "--perspective",
            ".025",
            "--fonts",
            str(ROOT / "fonts/Vazirmatn.ttf"),
            str(ROOT / "fonts/NotoSansArabic.ttf"),
            str(ROOT / "fonts/NotoNaskhArabic.ttf"),
            "--exclude-values",
            str(a.private_guard),
        ]
    )
    repeated = [
        json.loads(s) for s in (repeat / "labels.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert repeated == rows[:125]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(digest(repeat / r["file"]) == digest(a.dataset / r["file"]) for r in repeated)
    result = {
        "checks_passed": True,
        "source_receipts_reviewed": 76,
        "families": 25,
        "template_slots": sum(len(s["slots"]) for s in specs.values()),
        "generated_images_tested": len(rows),
        "text_regions_checked": count,
        "fonts_tested": 3,
        "source_customer_value_matches": 0,
        "source_raster_pixels_used": False,
        "all_base_images_reproduced_from_vectors": True,
        "overlapping_source_text_boxes": 0,
        "field_truth_checked": True,
        "card_and_iban_checksum_checks": True,
        "transformed_polygons_checked": True,
        "replayed_images_identical": len(repeated),
        "custom_record_images_tested": len(custom),
        "evaluation_set_used": False,
    }
    (ROOT / "VERIFICATION.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
