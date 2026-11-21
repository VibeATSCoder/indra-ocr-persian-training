"""Create small, entirely invented Persian line images for an end-to-end smoke run."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features


def make_demo(output: Path, font_path: Path):
    if output.exists():
        raise FileExistsError(f"Use a fresh demo output directory: {output}")
    if not features.check_feature("raqm"):
        raise RuntimeError("Persian rendering requires a Pillow build with libraqm")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    font = ImageFont.truetype(str(font_path), 30)
    texts = {
        "train": [
            "آزمایش خواندن متن فارسی",
            "این یک نمونه آموزشی است",
            "کتاب تازه روی میز قرار دارد",
            "امروز هوا روشن است",
            "شماره نمونه ۱۲۳۴",
            "سامانه پردازش تصویر",
        ],
        "validation": ["پاراگرافی برای ارزیابی مدل", "روز دیگری فرا رسیده است"],
        "test": ["دقت خواندن نوشته بررسی شد", "نمونه مستقل برای آزمون"],
    }
    records = []
    for split, lines in texts.items():
        for index, text in enumerate(lines):
            image = Image.new("RGB", (760, 85), "white")
            ImageDraw.Draw(image).text(
                (740, 12), text, font=font, fill="black", direction="rtl", anchor="ra"
            )
            name = f"images/{split}-{index}.png"
            image.save(output / name)
            records.append(
                {
                    "image": name,
                    "text": text,
                    "document_id": f"demo-{split}-{index}",
                    "split": split,
                    "domain": "custom",
                }
            )
    (output / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
    )
    return len(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument(
        "--font", type=Path, default=Path("tools/synthetic_receipts/fonts/Vazirmatn.ttf")
    )
    args = parser.parse_args()
    print(f"Created {make_demo(args.output, args.font)} synthetic line images")
