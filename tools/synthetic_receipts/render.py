"""Standalone vector receipt renderer. It never opens source receipt images."""

from __future__ import annotations
import argparse, functools, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "lib")]
from layout import runs
from indra_ocr.rtl import flip, visual_label


@functools.lru_cache(maxsize=512)
def font_at(path, size, weight=400):
    font = ImageFont.truetype(str(path), int(size))
    try:
        axes = font.get_variation_axes()
        font.set_variation_by_axes(
            [weight if a["name"] in (b"Weight", "Weight") else a["default"] for a in axes]
        )
    except (OSError, AttributeError):
        pass
    return font


def pieces(text):
    for rtl, part in runs(visual_label(text)):
        yield (
            (flip(part, base_rtl=True), {"direction": "rtl", "language": "fa"})
            if rtl
            else (part, {"direction": "ltr", "features": ["-liga"]})
        )


def measure(text, font):
    return sum(font.getlength(part, **kw) for part, kw in pieces(text))


def draw_line(draw, text, font, color, region, align):
    x0, y0, x1, y1 = region
    width = measure(text, font)
    scratch = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    extents = []
    x = 0
    for part, kw in pieces(text):
        extents.append(scratch.textbbox((x, 0), part, font=font, **kw))
        x += font.getlength(part, **kw)
    if not extents:
        return None
    top = min(b[1] for b in extents)
    bottom = max(b[3] for b in extents)
    y = y0 + (y1 - y0 - bottom + top) / 2 - top
    x = x0 if align == "left" else x1 - width if align == "right" else (x0 + x1 - width) / 2
    boxes = []
    for part, kw in pieces(text):
        boxes.append(draw.textbbox((x, y), part, font=font, **kw))
        draw.text((x, y), part, font=font, fill=tuple(color), **kw)
        x += font.getlength(part, **kw)
    return [
        math.floor(min(b[0] for b in boxes)),
        math.floor(min(b[1] for b in boxes)),
        math.ceil(max(b[2] for b in boxes)),
        math.ceil(max(b[3] for b in boxes)),
    ]


def wrap(text, font, width, limit):
    if limit == 1:
        return [text]
    lines = []
    current = ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if current and measure(trial, font) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def render_background(spec, scale=2):
    image = Image.new(
        "RGB", (spec["width"] * scale, spec["height"] * scale), tuple(spec["background_color"])
    )
    draw = ImageDraw.Draw(image)
    for p in spec["primitives"]:
        kind = p["kind"]
        color = tuple(p.get("fill") or (0, 0, 0))
        outline = tuple(p["outline"]) if p.get("outline") else None
        box = [round(v * scale) for v in p.get("box", [])]
        width = max(1, round(p.get("stroke", 1) * scale))
        if kind == "rect":
            draw.rounded_rectangle(
                box,
                radius=p.get("radius", 0) * scale,
                fill=color if p.get("fill") is not None else None,
                outline=outline,
                width=width,
            )
        elif kind == "ellipse":
            draw.ellipse(
                box, fill=color if p.get("fill") is not None else None, outline=outline, width=width
            )
        elif kind == "line":
            draw.line(
                [tuple(v * scale for v in pt) for pt in p["points"]],
                fill=color,
                width=width,
                joint="curve",
            )
        elif kind == "polygon":
            draw.polygon([tuple(v * scale for v in pt) for pt in p["points"]], fill=color)
        elif kind == "gradient":
            x0, y0, x1, y1 = box
            a = np.asarray(p["start"])
            b = np.asarray(p["end"])
            data = np.linspace(a, b, max(1, y1 - y0)).clip(0, 255).astype(np.uint8)[:, None, :]
            data = np.repeat(data, max(1, x1 - x0), axis=1)
            image.paste(Image.fromarray(data), (x0, y0))
            draw = ImageDraw.Draw(image)
    return image.resize((spec["width"], spec["height"]), Image.Resampling.LANCZOS)


def render(spec, record=None, *, mode="filled", font_override=None, background=None):
    if not features.check_feature("raqm"):
        raise RuntimeError(
            "Persian rendering needs Pillow with libraqm; use the verified WSL environment."
        )
    record = {**spec["example_record"], **(record or {})}
    image = background.copy() if background is not None else render_background(spec)
    if mode == "blank":
        return image, []
    draw = ImageDraw.Draw(image)
    annotations = []
    for slot in spec["slots"]:
        if mode == "form" and slot["role"] == "field":
            continue
        text = str(record.get("slots", {}).get(slot["id"], slot.get("text", "")))
        if slot["role"] == "field" and slot["id"] not in record.get("slots", {}):
            text = (
                slot.get("prefix", "") + str(record.get(slot["field"], "")) + slot.get("suffix", "")
            )
        if not text:
            continue
        path = Path(font_override or ROOT / slot.get("font", "fonts/Vazirmatn.ttf"))
        if not path.is_absolute():
            path = ROOT / path
        size = slot["font_size"]
        minimum = max(9, int(size * 0.6))
        region = slot["region"]
        limit = slot.get("max_lines", 1)
        while size >= minimum:
            font = font_at(str(path), size, slot.get("weight", 400))
            lines = wrap(text, font, region[2] - region[0] - 4, limit)
            heights = []
            for line in lines:
                bb = [font.getbbox(part, **kw) for part, kw in pieces(line)]
                heights.append(max(b[3] for b in bb) - min(b[1] for b in bb) if bb else 0)
            if (
                len(lines) <= limit
                and all(measure(line, font) <= region[2] - region[0] - 3 for line in lines)
                and max(heights, default=0) * len(lines) <= region[3] - region[1] - 2
            ):
                break
            size -= 1
        else:
            raise ValueError(
                f"Cannot fit text in {spec['id']}/{slot['id']}: use shorter text or enlarge its region"
            )
        step = (region[3] - region[1]) / max(1, len(lines))
        for i, line in enumerate(lines):
            bounds = [region[0], region[1] + i * step, region[2], region[1] + (i + 1) * step]
            box = draw_line(draw, line, font, slot["color"], bounds, slot.get("align", "right"))
            if box and not (
                0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
            ):
                raise ValueError(f"Clipped text in {spec['id']}/{slot['id']}")
            if box:
                annotations.append(
                    {
                        "slot": slot["id"],
                        "field": slot.get("field"),
                        "role": slot["role"],
                        "text": line,
                        "bbox": box,
                        "polygon": [
                            [box[0], box[1]],
                            [box[2], box[1]],
                            [box[2], box[3]],
                            [box[0], box[3]],
                        ],
                        "font": path.name,
                        "font_size": size,
                        "weight": slot.get("weight", 400),
                        "ignore_training": slot["role"] == "provenance",
                    }
                )
    return image, annotations


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("template", type=Path)
    p.add_argument("--record", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["blank", "form", "filled"], default="filled")
    p.add_argument("--font", type=Path)
    a = p.parse_args()
    spec = json.loads(a.template.read_text(encoding="utf8"))
    record = json.loads(a.record.read_text(encoding="utf8")) if a.record else None
    image, labels = render(spec, record, mode=a.mode, font_override=a.font)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    image.save(a.out)
    a.out.with_suffix(".gt.json").write_text(
        json.dumps(
            {
                "synthetic_pixels": True,
                "fully_synthetic": a.record is None,
                "content_origin": "user_record" if a.record else "synthetic_example",
                "template": spec["id"],
                "lines": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    print("Saved", a.out)
