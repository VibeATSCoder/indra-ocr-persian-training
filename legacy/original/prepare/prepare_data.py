"""Prepare licensed Persian OCR samples on Kaggle CPU, without any credentials."""
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pyarrow", "pillow"])
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import unicodedata
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

os.environ["HF_HUB_DISABLE_XET"] = "1"
ROOT = Path("/kaggle/working")
CACHE = Path("/tmp/persian-ocr-downloads")
ROOT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "garshasp": {"repo": "AliShafiee2003/persian-ocr-garshasp-70c", "revision": "36bfdcdeac20c02231f4ee08472f80db2fc467bb", "license": "CC-BY-4.0", "author": "Ali Shafiee"},
    "persian230k": {"repo": "ordaktaktak/Persian-OCR-230k", "revision": "05d248b30988b41bb905f7919c0bac39cab38d70", "license": "MIT", "author": "ordaktaktak"},
}

def download(source, filename):
    s = SOURCES[source]
    print("Downloading", source, filename, flush=True)
    return hf_hub_download(s["repo"], filename, repo_type="dataset", revision=s["revision"], cache_dir=CACHE)

def group_key(text):
    normalized = unicodedata.normalize("NFKC", text).translate(str.maketrans({"ي":"ی", "ك":"ک", "\u200c":"", "ـ":""}))
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode()).hexdigest()

rows = []
seen_images = set()
rejected = collections.Counter()

def add(image_bytes, text, source, source_id, official_test=False):
    text = unicodedata.normalize("NFC", str(text)).strip()
    if not 3 <= len(text) <= 200 or not any("\u0600" <= c <= "\u06ff" for c in text):
        rejected["label"] += 1
        return
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im.verify()
    except Exception:
        rejected["image"] += 1
        return
    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest in seen_images:
        rejected["duplicate_image"] += 1
        return
    seen_images.add(digest)
    rows.append({"image": image_bytes, "text": text, "source": source, "source_id": source_id, "text_group": group_key(text), "image_sha256": digest, "official_test": official_test})

randomizer = random.Random(20261101)
for split, cap in [("plain_text", 25000), ("plain_text_blur", 25000), ("plain_text_shadow", 10000)]:
    path = download("garshasp", f"data/{split}-00000-of-00001.parquet")
    # Deterministic reservoir sampling spans the entire source shard.
    reservoir = []
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for item in batch.to_pylist():
            if n < cap:
                reservoir.append((n, item))
            else:
                j = randomizer.randrange(n + 1)
                if j < cap:
                    reservoir[j] = (n, item)
            n += 1
    for index, item in sorted(reservoir):
        image = item["image"]
        content = image["bytes"] if isinstance(image, dict) else image
        add(content, item["text"], f"garshasp/{split}", f"{split}:{index}")
    print("Prepared", split, "rows", len(rows), flush=True)
    del reservoir

archive = zipfile.ZipFile(download("persian230k", "data/Images.zip"))
members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
for split, cap in [("train", 40000), ("test", 2000)]:
    with open(download("persian230k", f"data/{split}.csv"), encoding="utf-8-sig", newline="") as handle:
        labels = list(csv.DictReader(handle))
    print("CSV fields", split, list(labels[0]), flush=True)
    randomizer.shuffle(labels)
    accepted = 0
    for index, item in enumerate(labels):
        name = Path(item["fname"]).name
        if name not in members:
            rejected["missing_image"] += 1
            continue
        before = len(rows)
        add(archive.read(members[name]), item["text"], f"persian230k/{split}", item["fname"], split == "test")
        accepted += len(rows) - before
        if accepted >= cap:
            break
    print("Prepared", split, accepted, flush=True)

test_groups = {r["text_group"] for r in rows if r["official_test"]}
counts = collections.Counter()
writers = {}
schema = pa.schema([(k, pa.binary() if k == "image" else pa.string()) for k in ("image", "text", "source", "source_id", "text_group", "image_sha256", "split")])
buffers = collections.defaultdict(list)
for row in rows:
    official_test = row.pop("official_test")
    key = row["text_group"]
    bucket = int(key[:8], 16) % 100
    split = "test" if key in test_groups or bucket >= 98 else "validation" if bucket >= 96 else "train"
    row["split"] = split
    buffers[split].append(row)
    counts[f"{split}:{row['source']}"] += 1
for split, data in buffers.items():
    pq.write_table(pa.Table.from_pylist(data, schema=schema), ROOT / f"{split}.parquet", compression="zstd", row_group_size=256)
    print(split, len(data), flush=True)

report = {"sources": SOURCES, "seed": 20261101, "counts": dict(counts), "rejected": dict(rejected), "rows": len(rows), "split_rule": "96/2/2 by normalized text SHA256, official 230k test labels always held out across every source", "modifications": "deterministic subset, exact image deduplication, Unicode NFC, text-group split assignment; image bytes unmodified", "scope": "printed synthetic line recognition; no claim of real-book validation"}
(ROOT / "data-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
(ROOT / "ATTRIBUTION.md").write_text("# Persian OCR training sources\n\n" + "\n\n".join(f"{v['author']}: https://huggingface.co/datasets/{v['repo']} at {v['revision']}. License: {v['license']}." for v in SOURCES.values()) + "\n\nChanges: deterministic sampling, image deduplication, NFC labels and grouped train/validation/test splits. Original image bytes retained.\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/\nMIT: https://opensource.org/license/mit\n")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
