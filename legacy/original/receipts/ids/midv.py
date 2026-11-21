"""MIDV-UP (Chernyshova, Ilyukhin, Arlazarov, ICDAR 2025; CC BY-SA 2.5): mock Iranian ID documents in real photos, scans and video.

Splits are by card, so all views of a card stay together. Test cards are for the final measure only and never reach Kaggle.
iphone captures hold cards 001-125, samsung captures 126-250.

    PYTHONPATH=receipts/ids .control-venv/bin/python receipts/ids/midv.py fetch      # validation and test cards of irn_id, local
"""
import concurrent.futures
import json
import sys
from pathlib import Path

URL = "ftp://smartengines.com/midv-up/dataset.zip"
SIZE = 12586662513
ROOT = "dataset/dataset"
LOCAL = Path(__file__).resolve().parent / "midv-up/local"
FIELD_NAMES = {"id_number": "national_id", "name": "first_name", "last_name": "last_name", "birth_date": "birth_date", "father_name": "father_name", "expiry_date": "expiry_date"}
TEST = {f"{i:03d}" for i in (*range(106, 126), *range(231, 251))}
VALIDATION = {f"{i:03d}" for i in (*range(91, 106), *range(216, 231))}


def split_of(card):
    return "test" if card in TEST else "validation" if card in VALIDATION else "train"


def camera_of(card):
    return "iphone" if int(card) <= 125 else "samsung"


def views(card, doc="irn_id", names=None):
    """(view, member path, annotation member, annotation file name) of every image of a card."""
    cam = camera_of(card)
    out = [("template", f"{ROOT}/templates/images/{doc}/{card}.png", f"{ROOT}/templates/annotations/{doc}.json", f"{card}.png"),
           ("left_scan", f"{ROOT}/left_scans/images/{doc}/{card}.png", f"{ROOT}/left_scans/annotations/{doc}.json", f"{card}.png"),
           ("center_scan", f"{ROOT}/center_scans/images/{doc}/{card}.png", f"{ROOT}/center_scans/annotations/{doc}.json", f"{card}.png"),
           ("photo", f"{ROOT}/photo/images/{doc}_{cam}/{card}.jpg", f"{ROOT}/photo/annotations/{doc}_{cam}.json", f"{card}.jpg")]
    prefix = f"{ROOT}/video_frames/images/{doc}_{cam}/{card}/"
    for n in sorted(m for m in names if m.startswith(prefix) and m.endswith(".jpg")):
        out.append(("frame", n, f"{ROOT}/video_frames/annotations/{doc}_{cam}/{card}/{card}.json", Path(n).name))
    return out


def regions(meta, filename):
    """The VIA regions recorded for a file name. iPhone annotations drop the zeros ("1.jpg" for 001.jpg); video frames have none."""
    want = (Path(filename).stem.lstrip("0"), Path(filename).suffix)
    for row in meta["_via_img_metadata"].values():
        name = Path(row.get("filename", ""))
        if (name.stem.lstrip("0"), name.suffix) == want:
            return row["regions"]
    return []


def fetch():
    from remote_zip import open_remote

    z = open_remote(URL, SIZE)
    names = [n for n in z.namelist() if not n.endswith("/")]
    (LOCAL / "images").mkdir(parents=True, exist_ok=True)
    cards = sorted(TEST | VALIDATION)
    jobs = [(c, v) for c in cards for v in views(c, names=names)]
    annotations = sorted({v[2] for _, v in jobs})
    meta = {}
    for a in annotations:
        meta[a] = json.load(z.open(a))
    template = meta[f"{ROOT}/templates/annotations/irn_id.json"]

    def one(job):
        card, (view, member, ann, fname) = job
        dest = LOCAL / "images" / f"{card}-{view}-{Path(member).name}"
        if not dest.exists():
            with open_remote(URL, SIZE, 1 << 22) as zz:
                dest.write_bytes(zz.read(member))
        return {"card": card, "view": view, "file": dest.name, "split": split_of(card), "regions": regions(meta[ann], fname)}

    rows = []
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for n, row in enumerate(pool.map(one, jobs), 1):
            rows.append(row)
            if n % 50 == 0:
                print("fetched", n, "/", len(jobs), flush=True)
    truth = {c: regions(template, f"{c}.png") for c in cards}
    (LOCAL / "index.json").write_text(json.dumps({"views": rows, "template_regions": truth}, ensure_ascii=False), encoding="utf8")
    print("done", len(rows), "images;", sum(r["split"] == "test" for r in rows), "test", round(sum(p.stat().st_size for p in (LOCAL / "images").iterdir()) / 1e6), "MB", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch()
