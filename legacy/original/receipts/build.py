"""Bundle pinned Indra source plus a receipts stage script into a Kaggle script kernel.
Never bundles credentials.

    .control-venv/bin/python receipts/build.py render   # free CPU, renders receipt lines
    .control-venv/bin/python receipts/build.py train --run 3   # T4 x2; the run's data and start model are in train.py RUNS
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["render", "train"])
ap.add_argument("--run", default="", help="run suffix for kernel names, such as 2")
args = ap.parse_args()
work = ROOT / "receipts" / f"kernel-{args.stage}{args.run}"
work.mkdir(exist_ok=True)

src = """import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','rapidfuzz':'rapidfuzz','pyarrow':'pyarrow','cv2':'opencv-python-headless','PIL':'pillow','requests':'requests'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-receipts-code');(code/'indra_ocr').mkdir(parents=True,exist_ok=True)
"""
for path in sorted((ROOT / "vendor/indra/indra_ocr").glob("*.py")):
    src += f"(code/'indra_ocr'/{path.name!r}).write_text({path.read_text(encoding='utf8')!r},encoding='utf8')\n"
files = [ROOT / "v3/common.py", ROOT / "receipts" / f"{args.stage}.py"]
files += [ROOT / "receipts/fields.py", ROOT / "receipts/layout.py"] if args.stage == "render" else [ROOT / "indra_rtl_trainable.py"]
for p in files:
    text = p.read_text(encoding="utf8")
    assert "KGAT_" not in text and "hf_" + "eV" not in text
    compile(text, p.name, "exec")
    src += f"(code/{p.name!r}).write_text({text!r},encoding='utf8')\n"
src += f"env=dict(os.environ,PYTHONPATH=str(code),RUN={args.run or '2'!r})\n"
src += f"subprocess.check_call([sys.executable,'-u',str(code/{args.stage + '.py'!r})],env=env,cwd=str(code))\n"
(work / "run.py").write_text(src, encoding="utf8")

meta = {"id": f"sharifi76/indra-persian-receipts{args.run}-{args.stage}", "title": f"Indra Persian Receipts{args.run} {args.stage.title()}",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_internet": True, "enable_gpu": args.stage == "train",
        "dataset_sources": ["sharifi76/persian-ocr-large-mixed-v3"] + (["sharifi76/persian-free-fonts-ocr", "sharifi76/persian-receipt-fonts-ocr", "sharifi76/persian-receipt-fonts-384"] if args.stage == "render" else ["sharifi76/persian-receipt-real-lines"] if args.run == "6" else []),
        "kernel_sources": ["sharifi76/indra-persian-v4-render", {"3": "sharifi76/indra-persian-receipts2-train", "4": "sharifi76/indra-persian-receipts3-train", "5": "sharifi76/indra-persian-receipts3-train", "6": "sharifi76/indra-persian-receipts3-train"}.get(args.run, "sharifi76/indra-persian-v5-train"),
                            f"sharifi76/indra-persian-receipts{'2' if args.run in ('3', '4', '5', '6') else args.run}-render"] + (["sharifi76/indra-persian-ids-prepare"] if args.run in ("4", "5", "6") else []) if args.stage == "train" else [],
        "competition_sources": [], "model_sources": []}
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "receipts/holdout"))
from guard import check_sources as _check_sources  # noqa: E402
_check_sources(meta)
if args.stage == "train":
    meta["machine_shape"] = "NvidiaTeslaT4"
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("bundled", args.stage, len(src), "bytes", hashlib.sha256(src.encode()).hexdigest()[:16])
