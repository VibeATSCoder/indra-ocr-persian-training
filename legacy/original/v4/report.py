"""Write <run>/REPORT.md from the Kaggle results and the local page-suite evaluation.

    .control-venv/bin/python v4/report.py v4/results/results.json v4/eval-v4.json   # -> v4/REPORT.md
    .control-venv/bin/python v4/report.py v5/results/results.json v5/eval-v5.json   # -> v5/REPORT.md
"""
import json
import sys
from pathlib import Path

res = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
RUN = res.get("run", "v4")
DEST = Path(sys.argv[1]).resolve().parents[1] / "REPORT.md"
ev = json.loads(Path(sys.argv[2]).read_text(encoding="utf8"))
DOMAINS = ["real", "bookfont", "font", "arshasb", "persian230k", "garshasp", "all"]
LABEL = {"real": "Real document lines (machine labels, held-out docs)", "bookfont": "Scan-degraded unseen book font (Nazli)",
         "font": "Clean unseen fonts (Amiri, Scheherazade)", "arshasb": "Arshasb rendered book lines",
         "persian230k": "Persian-OCR-230k", "garshasp": "Garshasp", "all": "All test lines"}


def pct(x):
    return f"{100 * x:.2f}%"


L = [f"# Indra Persian OCR fine-tune {RUN}", ""]
L.append(f"Selected on validation: **{res['selected']}**. Selection scores (lower is better): "
         + ", ".join(f"{k} {v:.5f}" for k, v in sorted(res["selection_scores"].items(), key=lambda kv: kv[1])) + ".")
L.append("")
L.append("## Held-out test lines (opened once, after selection)")
L.append("")
models = ["baseline_test"] + [k for k in res if k.startswith("test_")]
L.append("| Domain | " + " | ".join("Original" if m == "baseline_test" else m[5:] for m in models) + " |")
L.append("|---|" + "---:|" * len(models))
for d in DOMAINS:
    if d in res["baseline_test"]:
        L.append(f"| {LABEL[d]} | " + " | ".join(pct(res[m][d]["cer"]) if d in res[m] else "–" for m in models) + " |")
L.append("")
L.append("Per test font:")
L.append("")
for src in sorted(k for k in res["baseline_test"] if "/" in k):
    L.append(f"- {src}: " + ", ".join(f"{'Original' if m == 'baseline_test' else m[5:]} {pct(res[m][src]['cer'])}" for m in models if src in res[m]))
L.append("")
L.append("## Full-page pipeline (local CPU, Indra detector and reading order unchanged)")
L.append("")
names = list(ev["summary"])
L.append("| Page set | " + " | ".join(names) + " |")
L.append("|---|" + "---:|" * len(names))
for s in ev["summary"][names[0]]:
    L.append(f"| {s} CER / no-space CER / word F1 | " + " | ".join(
        f"{pct(ev['summary'][n][s]['cer'])} / {pct(ev['summary'][n][s]['cer_nospace'])} / {ev['summary'][n][s]['bow_f1']:.3f}" for n in names) + " |")
L.append("")
ref = names[0]
L.append(f"Paired change vs {ref} (aggregate CER difference, 95% bootstrap CI over pages, pages better/worse):")
L.append("")
for n, comp in ev[f"paired_vs_{ref}"].items():
    for s, v in comp.items():
        c, ns = v["cer"], v["cer_nospace"]
        L.append(f"- {n} / {s}: CER {100 * c['delta']:+.2f} pt [{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}] "
                 f"({c['pages_better']}/{c['pages_worse']}); no-space {100 * ns['delta']:+.2f} pt "
                 f"[{100 * ns['ci95'][0]:+.2f}, {100 * ns['ci95'][1]:+.2f}] ({ns['pages_better']}/{ns['pages_worse']})")
L.append("")
L.append("## Setup")
L.append("")
L.append(f"- Data: persian-ocr-large-mixed-v3 plus {sum(res['counts']['train'].get('bookfont', 0) for _ in [0]):,} scan-degraded "
         "book-font training lines (13 free fonts; Nahid validation, Nazli test). Train counts: "
         + ", ".join(f"{k} {v:,}" for k, v in res["counts"]["train"].items()) + ".")
L.append(f"- {len(res['configs'])} candidates, two at a time (one per T4), {int(res['train_seconds_per_candidate'])} s each, "
         + ("time-based cosine learning rate; " if "run" in res else "one-cycle learning rate (decayed early); ")
         + "; ".join(f"{c['name']} lr {c['lr']}, anchor {c['anchor']}, distill {c['distill']}" for c in res["configs"]) + ".")
L.append("- Objective: CTC + KL to the frozen original recognizer (T=2) + L2 anchor to original weights; scan-style augmentation.")
L.append("- Only rec_rtl.onnx and meta.json change; detector, layout, order and other recognizers keep their original files.")
L.append("- Real-document labels are machine-generated and filtered, not human ground truth. Font families are disjoint across splits.")
DEST.write_text("\n".join(L) + "\n", encoding="utf8")
print("\n".join(L))
