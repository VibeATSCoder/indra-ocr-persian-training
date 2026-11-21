"""Generate fully synthetic receipt data with labels, variants and stable groups."""

import argparse, collections, hashlib, json, random, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from render import ROOT, render, render_background
from records import sample
from audit_dataset import fingerprint_value
from degrade import transform, LEVELS

PRIVATE_KEYS = (
    "sender",
    "receiver",
    "from_card",
    "to_card",
    "from_account",
    "to_account",
    "from_iban",
    "to_iban",
    "tracking",
    "reference",
    "phone",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--templates", type=Path, default=ROOT / "templates")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=1717)
    p.add_argument("--families", help="Comma-separated family IDs; default all 25")
    p.add_argument("--sampling", choices=["balanced", "empirical"], default="balanced")
    p.add_argument("--levels", default="clean,light,medium,heavy,photo")
    p.add_argument("--rotate", type=float, default=0)
    p.add_argument("--perspective", type=float, default=0)
    p.add_argument("--fonts", nargs="+", type=Path, default=[ROOT / "fonts/Vazirmatn.ttf"])
    p.add_argument(
        "--records",
        type=Path,
        help="Optional JSONL field dictionaries; omitted for fully invented text",
    )
    p.add_argument(
        "--exclude-values",
        type=Path,
        help="Optional private source-value fingerprint guard; do not put it in shared exports",
    )
    p.add_argument("--jsonl-only", action="store_true", help="Skip per-image annotation sidecars")
    a = p.parse_args()
    if a.count < 1 or not 0 <= a.rotate <= 30 or not 0 <= a.perspective <= 0.1:
        p.error("Invalid count or geometric distortion")
    levels = a.levels.split(",")
    unknown = set(levels) - ({"clean", "photo"} | set(LEVELS))
    if unknown:
        p.error("Unknown quality levels: " + ",".join(unknown))
    paths = sorted(p for p in a.templates.glob("*.json") if not p.name.endswith(".gt.json"))
    specs = [json.loads(p.read_text(encoding="utf8")) for p in paths]
    if a.families:
        names = set(a.families.split(","))
        specs = [s for s in specs if s["id"] in names]
        if names - {s["id"] for s in specs}:
            p.error("Unknown family in --families")
    if not specs:
        p.error("No templates found; run build_templates.py")
    supplied = (
        [
            json.loads(line)
            for line in a.records.read_text(encoding="utf8").splitlines()
            if line.strip()
        ]
        if a.records
        else None
    )
    if supplied == []:
        p.error("Records file is empty")
    forbidden = (
        set(json.loads(a.exclude_values.read_text())["normalized_value_sha256"])
        if a.exclude_values
        else set()
    )
    if (a.out / "labels.jsonl").exists():
        p.error("Choose a fresh output folder; labels.jsonl already exists")
    a.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    rng = random.Random(a.seed)
    cached = {}
    counts = collections.Counter()
    retries = 0
    images = 0
    groups = set()
    regions = 0
    order = list(range(len(specs)))
    rng.shuffle(order)
    with (a.out / "labels.jsonl").open("w", encoding="utf8") as manifest:
        for i in range(a.count):
            j = (
                order[i % len(order)]
                if a.sampling == "balanced"
                else rng.choices(range(len(specs)), weights=[s["source_count"] for s in specs])[0]
            )
            spec = specs[j]
            r = random.Random(a.seed + i)
            font = a.fonts[i % len(a.fonts)]
            if spec["id"] not in cached:
                cached[spec["id"]] = render_background(spec)
            for attempt in range(200):
                provided = supplied[i % len(supplied)] if supplied else None
                record = (
                    {**spec["example_record"], **provided}
                    if provided is not None
                    else sample(r, spec["status"])
                )
                if any(
                    fingerprint_value(record[k]) in forbidden for k in PRIVATE_KEYS if record.get(k)
                ):
                    if supplied:
                        raise ValueError("Supplied record matches a protected source value")
                    retries += 1
                    continue
                try:
                    image, labels = render(
                        spec, record, font_override=font, background=cached[spec["id"]]
                    )
                    break
                except ValueError:
                    if supplied:
                        raise
                    retries += 1
            else:
                raise RuntimeError("Cannot generate a fitting record for " + spec["id"])
            identity = provided if provided is not None else record
            group = str(
                identity.get("record_id")
                or hashlib.sha256(
                    json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()[:20]
            )
            groups.add(group)
            counts[spec["id"]] += 1
            original = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for k, level in enumerate(levels):
                rs = np.random.default_rng([a.seed, i, k])
                array, lines, H = transform(
                    original.copy(),
                    json.loads(json.dumps(labels)),
                    level,
                    rs,
                    a.rotate,
                    a.perspective,
                )
                name = f"{i:07d}-{spec['id']}-{level}"
                Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB)).save(
                    a.out / (name + ".png")
                )
                row = {
                    "file": name + ".png",
                    "family": spec["id"],
                    "document_group": group,
                    "seed": a.seed + i,
                    "level": level,
                    "synthetic_pixels": True,
                    "fully_synthetic": supplied is None,
                    "content_origin": "user_records" if supplied else "seeded_generator",
                    "width": array.shape[1],
                    "height": array.shape[0],
                    "homography": H,
                    "record": record,
                    "lines": lines,
                }
                if not a.jsonl_only:
                    (a.out / (name + ".gt.json")).write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf8"
                    )
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                images += 1
                regions += len(lines)
            if (i + 1) % 25 == 0:
                print("Generated", i + 1, "documents /", images, "images", flush=True)
    summary = {
        "documents": a.count,
        "unique_document_groups": len(groups),
        "images": images,
        "text_regions": regions,
        "family_document_counts": dict(counts),
        "quality_levels": levels,
        "seed": a.seed,
        "sampling": a.sampling,
        "all_pixels_vector_generated": True,
        "fully_synthetic": supplied is None,
        "source_value_guard_used": bool(forbidden),
        "resampled_records": retries,
    }
    (a.out / "dataset-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
