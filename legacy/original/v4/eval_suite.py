"""Full-page evaluation of Indra model bundles (CPU). Evaluation only, never training.

Only rec_rtl.onnx/meta.json differ between bundles, so line detection and reading order are
identical across models: paired per-page differences isolate the recognizer.

Sets
  printed  27 machine-printed prose/book-like pages of PersianML/persian-ocr-benchmark
  forms    23 printed forms / tables / receipts from the same benchmark
  synth    known-text Persian scans rendered in 27 fonts (chatbot_x parser_bench corpus)

Page subsets were chosen from thumbnails only, before any v4 model predictions.

    PYTHONPATH=vendor/indra .control-venv/bin/python v4/eval_suite.py \
        --model base=base-model --model books=outputs-books/model --out v4/eval-books.json
"""
import argparse
import collections
import json
import random
import time
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmark"
SYNTH = Path("/var/services/chatbot_x/parser_bench/corpus/synthetic")
PRINTED = ["003", "005", "008", "011", "013", "014", "026", "044", "046", "047", "051", "053", "056", "062",
           "069", "070", "071", "074", "075", "078", "087", "090", "092", "095", "098", "099", "025"]
FORMS = ["002", "021", "027", "028", "033", "036", "037", "049", "050", "054", "055", "057", "058", "065",
         "066", "072", "083", "088", "093", "094", "096", "097", "100"]
SYNTH_VARIANTS = ["pdf_scan", "pdf_scan_bad", "pdf_scan_short", "pdf_scan_bad_short"]

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def norm(text):
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({"ي": "ی", "ك": "ک", "ـ": "", "‌": " "}))
    text = "".join(c for c in text if not ("ً" <= c <= "ٟ" or c == "ٰ"))  # harakat
    return " ".join(text.translate(DIGITS).split())


def score(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    rw, hw = ref.split(), hyp.split()
    common = sum((collections.Counter(rw) & collections.Counter(hw)).values())
    return {"char_edits": Levenshtein.distance(ref, hyp), "chars": len(ref),
            "ns_edits": Levenshtein.distance(ref.replace(" ", ""), hyp.replace(" ", "")), "ns_chars": len(ref.replace(" ", "")),
            "word_edits": Levenshtein.distance(rw, hw), "words": len(rw), "matched": common, "pred_words": len(hw)}


def summarize(recs):
    t = collections.Counter()
    for r in recs:
        t.update({k: v for k, v in r.items() if isinstance(v, int)})
    p, r = t["matched"] / max(1, t["pred_words"]), t["matched"] / max(1, t["words"])
    return {"pages": len(recs), "cer": t["char_edits"] / max(1, t["chars"]), "cer_nospace": t["ns_edits"] / max(1, t["ns_chars"]),
            "wer": t["word_edits"] / max(1, t["words"]), "bow_f1": 2 * p * r / max(1e-9, p + r)}


def items():
    rows = {json.loads(l)["id"]: json.loads(l) for l in (BENCH / "ocr_eval.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()}
    out = []
    for name, ids in (("printed", PRINTED), ("forms", FORMS)):
        for i in ids:
            row = rows[f"image_{i}"]
            out.append((name, row["id"], BENCH / Path(row["image"]).name, row["text"]))
    if SYNTH.exists():
        for d in json.loads((SYNTH / "manifest.json").read_text(encoding="utf-8")):
            if d["variant"] in SYNTH_VARIANTS:
                p = Path("/var/services/chatbot_x/parser_bench") / d["path"]
                gt = (Path("/var/services/chatbot_x/parser_bench") / d["gt"]).read_text(encoding="utf-8")
                out.append(("synth", f"{d['variant']}/{p.stem}", p, gt))
    return out


def run_model(model_dir, work):
    import indra_ocr

    conv = indra_ocr.load(path=str(model_dir), threads=4, mode="ocr", furniture=True)
    res = {}
    for subset, pid, path, ref in work:
        t0 = time.monotonic()
        text = "\n".join(page.text for _, page in conv.convert(path))
        res[pid] = {"subset": subset, **score(ref, text), "seconds": round(time.monotonic() - t0, 2), "text": text}
    return res


def paired(a, b, ids, key="char_edits", denom="chars", n=2000):
    """Bootstrap CI for the aggregate CER difference b - a over pages (negative = b better)."""
    rng = random.Random(0)
    diffs = []
    for _ in range(n):
        s = [rng.choice(ids) for _ in ids]
        ea, eb, c = sum(a[i][key] for i in s), sum(b[i][key] for i in s), max(1, sum(a[i][denom] for i in s))
        diffs.append((eb - ea) / c)
    diffs.sort()
    better = sum(b[i][key] < a[i][key] for i in ids)
    worse = sum(b[i][key] > a[i][key] for i in ids)
    return {"delta": (sum(b[i][key] for i in ids) - sum(a[i][key] for i in ids)) / max(1, sum(a[i][denom] for i in ids)),
            "ci95": [diffs[int(.025 * n)], diffs[int(.975 * n)]], "pages_better": better, "pages_worse": worse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=dir (first one is the reference)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = items()
    models = dict(m.split("=", 1) for m in args.model)
    results = {}
    for name, d in models.items():
        print("evaluating", name, d, flush=True)
        results[name] = run_model(ROOT / d if not Path(d).is_absolute() else Path(d), work)
    subsets = sorted({w[0] for w in work}) + ["synth:" + v for v in SYNTH_VARIANTS]
    summary = {}
    for name, res in results.items():
        summary[name] = {}
        for s in subsets:
            recs = [r for pid, r in res.items() if (r["subset"] == s if ":" not in s else pid.startswith(s.split(":")[1] + "/"))]
            if recs:
                summary[name][s] = summarize(recs)
    ref = next(iter(results))
    comparisons = {}
    for name in list(results)[1:]:
        comparisons[name] = {}
        for s in ("printed", "forms", "synth"):
            ids = [pid for pid, r in results[ref].items() if r["subset"] == s]
            if ids:
                comparisons[name][s] = {"cer": paired(results[ref], results[name], ids),
                                        "cer_nospace": paired(results[ref], results[name], ids, "ns_edits", "ns_chars")}
    Path(args.out).write_text(json.dumps({"models": models, "summary": summary, "paired_vs_" + ref: comparisons,
                                          "per_page": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in subsets:
        line = f"{s:24s}" + "".join(f"  {n}: CER {summary[n][s]['cer']:.4f} ns {summary[n][s]['cer_nospace']:.4f} BoW {summary[n][s]['bow_f1']:.3f}"
                                    for n in results if s in summary[n])
        print(line)
    print(json.dumps(comparisons, indent=1))


if __name__ == "__main__":
    main()
