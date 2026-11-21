"""Create an offline gallery of synthetic assets only."""

import html, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "catalog.json").read_text())
canvas = Image.new("RGB", (1350, 2400), "#dfe3e7")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(16)
cards = []
for i, row in enumerate(rows):
    name = row["id"]
    image = Image.open(ROOT / "templates" / (name + ".example.png")).convert("RGB")
    image.thumbnail((252, 430))
    x = i % 5 * 270
    y = i // 5 * 480
    canvas.paste(image, (x + (270 - image.width) // 2, y + 32))
    draw.text((x + 8, y + 6), name, font=font, fill="black")
    cells = "".join(
        f'<figure><a href="templates/{name}.{suffix}.png"><img loading="lazy" src="templates/{name}.{suffix}.png"></a><figcaption>{caption}</figcaption></figure>'
        for suffix, caption in [
            ("base", "Text-free background"),
            ("form", "Fixed labels"),
            ("example", "Synthetic example"),
        ]
    )
    cards.append(
        f'<section><h2>{html.escape(name)}</h2><p>Style group: {row["sources"]} approved source receipts · {row["width"]} × {row["height"]} · {row["slots"]} text regions</p><div class="images">{cells}</div><p><a href="templates/{name}.json">Editable layout JSON</a> · <a href="templates/{name}.example.gt.json">Example labels</a></p></section>'
    )
canvas.save(ROOT / "preview-styles.jpg", quality=92)
text = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic receipt style kit</title>
<style>body{font:16px system-ui;background:#e9edf2;color:#223047;margin:24px auto;max-width:1320px;padding:0 18px}h1{font-size:30px}section{background:white;padding:20px;border-radius:14px;margin:24px 0}h2{font-size:21px;margin-top:0}.images{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}figure{margin:0;text-align:center}img{max-width:100%;max-height:600px;border:1px solid #dfe4eb}figcaption{margin:8px}a{color:#0a61ad}p{line-height:1.6}@media(max-width:700px){.images{grid-template-columns:1fr}}</style>
<h1>25 synthetic receipt styles from your dataset</h1><p>76 approved receipts were reviewed. Every image below is newly drawn from vector shapes and invented values. Click an image for the full-size PNG. Each family includes a text-free background, a fixed-label form, an editable layout and exact text/box labels.</p><p><a href="README.md">Usage guide</a> · <a href="dataset_audit.json">Style audit</a> · <a href="VERIFICATION.json">Verification results</a></p>"""
(ROOT / "gallery.html").write_text(text + "\n".join(cards) + "</html>", encoding="utf8")
print("Created synthetic-only gallery and preview")
