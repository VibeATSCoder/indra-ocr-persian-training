"""Independent full-page Persian benchmark; evaluation only, never training."""
import argparse
import collections
import concurrent.futures
import io
import json
from pathlib import Path
import re
import time
import unicodedata
import requests
import numpy as np
from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein

REV = "b3d11e260c5d2ed199bf6a42034e5f12195cbc6a"
REPO = "PersianML/persian-ocr-benchmark"
ROOT = Path(__file__).resolve().parent
CACHE = ROOT/"benchmark"

def download():
    CACHE.mkdir(exist_ok=True)
    manifest = CACHE/"ocr_eval.jsonl"
    if not manifest.exists():
        response = requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/bench_data/ocr_eval.jsonl",timeout=60)
        response.raise_for_status()
        manifest.write_bytes(response.content)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    def get_image(row):
        filename = Path(row["image"]).name
        path = CACHE/filename
        if not path.exists():
            response = requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/bench_data/{row['image']}",timeout=90)
            response.raise_for_status()
            path.write_bytes(response.content)
        return row
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(get_image,rows))
    for sheet_index in range((len(rows)+24)//25):
        sheet = Image.new("RGB",(1000,1500),"#dddddd")
        draw = ImageDraw.Draw(sheet)
        for offset,row in enumerate(rows[sheet_index*25:(sheet_index+1)*25]):
            im = Image.open(CACHE/Path(row["image"]).name).convert("RGB")
            im.thumbnail((190,270))
            x,y = (offset%5)*200,(offset//5)*300
            sheet.paste(im,(x+(190-im.width)//2,y+20))
            draw.text((x+5,y+3),row["id"],fill="black")
        sheet.save(CACHE/f"contact-{sheet_index+1}.jpg")
    return rows

def normalize(text):
    text = unicodedata.normalize("NFKC",text).translate(str.maketrans({"ي":"ی","ك":"ک","ـ":"","\u200c":" "}))
    return " ".join(text.split())

def evaluate(model_dir,rows):
    import indra_ocr
    converter = indra_ocr.load(path=model_dir,threads=2,mode="ocr",furniture=True)
    predictions = []
    totals = collections.Counter()
    for row in rows:
        path = CACHE/Path(row["image"]).name
        before = time.monotonic()
        pages = list(converter.convert(path))
        predicted = "\n".join(page.text for _,page in pages)
        reference = normalize(row["text"])
        text = normalize(predicted)
        ref_words = reference.split()
        pred_words = text.split()
        common = collections.Counter(ref_words) & collections.Counter(pred_words)
        edits = Levenshtein.distance(reference,text)
        word_edits = Levenshtein.distance(ref_words,pred_words)
        no_space_reference, no_space_text = reference.replace(" ",""), text.replace(" ","")
        record = {"id":row["id"],"char_edits":edits,"chars":len(reference),"no_space_edits":Levenshtein.distance(no_space_reference,no_space_text),"no_space_chars":len(no_space_reference),"word_edits":word_edits,"words":len(ref_words),"matched_words":sum(common.values()),"predicted_words":len(pred_words),"cer":edits/max(1,len(reference)),"seconds":round(time.monotonic()-before,3)}
        predictions.append(record)
        for key in ("char_edits","chars","no_space_edits","no_space_chars","word_edits","words","matched_words","predicted_words"):
            totals[key] += record[key]
        print(row["id"],"CER",round(record["cer"],4),flush=True)
    return {"dataset":REPO,"revision":REV,"pages":len(rows),"scope":"Full pages, mixed printed and handwritten documents; held out of fine-tuning. NFKC, Persian Kaf/Yeh, whitespace and half-space normalization applied equally. Space-insensitive CER also removes spaces on both sides.","cer":totals["char_edits"]/max(1,totals["chars"]),"no_space_cer":totals["no_space_edits"]/max(1,totals["no_space_chars"]),"wer":totals["word_edits"]/max(1,totals["words"]),"word_recall":totals["matched_words"]/max(1,totals["words"]),"word_precision":totals["matched_words"]/max(1,totals["predicted_words"]),"totals":dict(totals),"per_page":predictions}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only",action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--out",default="page-metrics.json")
    parser.add_argument("--ids",help="Comma-separated page IDs, selected without model predictions")
    args = parser.parse_args()
    rows = download()
    print("Benchmark images downloaded:",len(rows),flush=True)
    if not args.download_only:
        if args.ids:
            include = set(args.ids.split(","))
            rows = [row for row in rows if row["id"] in include]
        result = evaluate(args.model_dir,rows)
        Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k != "per_page"}),flush=True)
