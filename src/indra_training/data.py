"""Prepare annotated line/page crops while keeping source documents disjoint."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._vendor.rec_preproc import crop_line
from ._vendor.rtl import normalize, visual_label

SPLITS = ("train", "validation", "test")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", normalize(str(text)))
    text = "".join(c for c in text if unicodedata.category(c) != "Cf" or c == "\u200c")
    return " ".join(text.split())


def canonical(text: str) -> str:
    return clean(text).translate(str.maketrans({"ي": "ی", "ك": "ک", "\u200c": ""})).replace(" ", "")


def group_split(
    document_id: str, seed: int, validation_fraction: float, test_fraction: float
) -> str:
    value = int(hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()[:16], 16) / 2**64
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "validation"
    return "train"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path.name}:{line_no}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path.name}:{line_no}")
                yield line_no, value


def image_ink(path: Path, bbox=None, max_width: int = 2048) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"))
    if bbox is None:
        if np.median(gray) < 110:
            gray = 255 - gray
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        keep = np.zeros(count, dtype=np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2
        ys, xs = np.where(keep[labels])
        if len(xs) < 3:
            raise ValueError("No visible text pixels")
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if len(bbox) != 4 or not all(math.isfinite(float(x)) for x in bbox):
        raise ValueError("bbox must be finite [x0, y0, x1, y1]")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("bbox must have positive width and height")
    return np.rint(crop_line(gray, bbox, max_width=max_width) * 255).astype(np.uint8)


def check_disjoint(rows: dict[str, list[dict]], allow_text_overlap: bool = False):
    seen = {key: {} for key in ("document_id", "image_sha256", "text_group")}
    for split in SPLITS:
        for row in rows.get(split, []):
            for key, lookup in seen.items():
                if key == "text_group" and allow_text_overlap:
                    continue
                value = row[key]
                previous = lookup.setdefault(value, split)
                if previous != split:
                    raise ValueError(f"Cross-split {key} overlap between {previous} and {split}")


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    seed: int = 20261104,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    max_width: int = 2048,
    allow_text_overlap: bool = False,
    drop_overlapping_images: bool = False,
) -> dict:
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh dataset output directory: {output_dir}")
    if min(validation_fraction, test_fraction) < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be nonnegative and sum to less than 1")
    if max_width < 8:
        raise ValueError("max_width must be at least 8")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rows = {split: [] for split in SPLITS}
    rejected, duplicates = Counter(), 0
    overlapping_images_dropped = Counter()
    seen_images = {}
    seen_documents = {}
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        (stage / "images").mkdir()
        for line_no, record in read_jsonl(input_path):
            for required in ("image", "text", "document_id"):
                if not isinstance(record.get(required), str) or not record[required].strip():
                    raise ValueError(f"Line {line_no}: missing nonempty string {required}")
            record["document_id"] = record["document_id"].strip()
            if (
                not isinstance(record.get("domain", "custom"), str)
                or not record.get("domain", "custom").strip()
            ):
                raise ValueError(f"Line {line_no}: domain must be a nonempty string")
            split = record.get("split") or group_split(
                record["document_id"], seed, validation_fraction, test_fraction
            )
            if split not in SPLITS:
                raise ValueError(f"Line {line_no}: split must be one of {SPLITS}")
            previous = seen_documents.setdefault(record["document_id"], split)
            if previous != split:
                raise ValueError(f"Line {line_no}: document_id appears in multiple splits")
            text = clean(record["text"])
            if not text:
                rejected["empty_text"] += 1
                continue
            image_path = Path(record["image"])
            if not image_path.is_absolute():
                image_path = input_path.parent / image_path
            try:
                ink = image_ink(image_path, record.get("bbox"), max_width)
            except FileNotFoundError as error:
                raise ValueError(f"Line {line_no}: image does not exist: {image_path}") from error
            except (ValueError, OSError, cv2.error):
                rejected["invalid_image_or_bbox"] += 1
                continue
            visual = visual_label(text)
            needed = len(visual) + sum(a == b for a, b in zip(visual, visual[1:]))
            if needed > math.ceil(ink.shape[1] / 4):
                rejected["ctc_too_short"] += 1
                continue
            digest = hashlib.sha256(ink.tobytes()).hexdigest()
            if digest in seen_images:
                old_split, old_text = seen_images[digest]
                if old_text != text:
                    raise ValueError(f"Line {line_no}: conflicting labels for an identical image")
                if old_split == split:
                    duplicates += 1
                    continue
                if not drop_overlapping_images:
                    raise ValueError(
                        f"Line {line_no}: identical cropped image appears across splits"
                    )
                # Keep held-out data: test > validation > train. Never move a
                # document into another split; remove only its duplicate line.
                if SPLITS.index(split) <= SPLITS.index(old_split):
                    overlapping_images_dropped[split] += 1
                    continue
                rows[old_split] = [row for row in rows[old_split] if row["image_sha256"] != digest]
                overlapping_images_dropped[old_split] += 1
            seen_images[digest] = split, text
            image_file = f"images/{digest}.png"
            Image.fromarray(ink).save(stage / image_file)
            rows[split].append(
                {
                    "image": image_file,
                    "image_sha256": digest,
                    "text": text,
                    "visual": visual,
                    "width": int(ink.shape[1]),
                    "document_id": record["document_id"],
                    "domain": record.get("domain", "custom"),
                    "text_group": hashlib.sha256(canonical(text).encode()).hexdigest(),
                }
            )
        check_disjoint(rows, allow_text_overlap)
        if not rows["train"] or not rows["validation"]:
            raise ValueError(
                "Dataset needs nonempty train and validation splits; set splits explicitly for tiny datasets"
            )
        digest = hashlib.sha256()
        for split, records in rows.items():
            content = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
            )
            (stage / f"{split}.jsonl").write_text(content, encoding="utf-8")
            digest.update(split.encode() + content.encode())
        manifest = {
            "format_version": 1,
            "data_sha256": digest.hexdigest(),
            "source_jsonl_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "max_width": max_width,
            "allow_text_overlap": allow_text_overlap,
            "drop_overlapping_images": drop_overlapping_images,
            "cross_split_duplicate_images_removed": dict(overlapping_images_dropped),
            "counts": {
                split: dict(Counter(r["domain"] for r in records))
                for split, records in rows.items()
            },
            "rejected": dict(rejected),
            "same_split_duplicate_images_removed": duplicates,
            "preprocessing": "Pinned Indra crop_line; 32px high uint8 ink, logical-to-visual RTL labels",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(output_dir)
    return manifest


def load_prepared(directory: str | Path) -> tuple[dict, dict[str, list[dict]]]:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported prepared dataset version")
    rows = {split: [row for _, row in read_jsonl(directory / f"{split}.jsonl")] for split in SPLITS}
    digest = hashlib.sha256()
    for split in SPLITS:
        content = (directory / f"{split}.jsonl").read_bytes()
        digest.update(split.encode() + content)
        for row in rows[split]:
            image_path = (directory / row["image"]).resolve()
            if not image_path.is_relative_to(directory):
                raise ValueError("Prepared image escapes dataset directory")
            ink = np.asarray(Image.open(image_path))
            if (
                ink.shape != (32, row["width"])
                or hashlib.sha256(ink.tobytes()).hexdigest() != row["image_sha256"]
            ):
                raise ValueError("Prepared image checksum or dimensions changed")
    if digest.hexdigest() != manifest["data_sha256"]:
        raise ValueError("Prepared label checksum changed; prepare a new dataset version")
    check_disjoint(rows, manifest.get("allow_text_overlap", False))
    if not rows["train"] or not rows["validation"]:
        raise ValueError("Prepared dataset needs train and validation records")
    return manifest, rows
