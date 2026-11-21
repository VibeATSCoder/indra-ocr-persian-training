"""Convert the recovered synthetic receipt annotations into the trainer's JSONL format."""

import argparse
import json
import os
from pathlib import Path


def convert(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        source.open(encoding="utf-8") as input_file,
        destination.open("w", encoding="utf-8") as output,
    ):
        for value in input_file:
            record = json.loads(value)
            if not record.get("document_group"):
                raise ValueError("document_group is required to keep augmented variants together")
            image = (source.parent / record["file"]).resolve()
            for line in record["lines"]:
                if line.get("ignore_training") or line.get("role") == "provenance":
                    continue
                row = {
                    "image": os.path.relpath(image, destination.parent),
                    "text": line["text"],
                    "bbox": line["bbox"],
                    "document_id": str(record["document_group"]),
                    "domain": "receipt",
                }
                if record.get("split"):
                    row["split"] = record["split"]
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Converted {convert(args.input, args.output)} line annotations")
