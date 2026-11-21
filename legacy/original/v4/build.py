"""Bundle pinned Indra source plus a v4 stage script into a Kaggle script kernel.
Never bundles credentials.

    .control-venv/bin/python v4/build.py render   # CPU, renders book-font lines
    .control-venv/bin/python v4/build.py train    # T4 x2, fine-tunes rec_rtl
    .control-venv/bin/python v4/build.py train --run v5   # candidate grid RUNS["v5"] in train.py
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["render", "train"])
ap.add_argument("--run", default="v4", help="training run name (train.py RUNS key)")
args = ap.parse_args()
name = "v4" if args.stage == "render" else args.run
work = ROOT / "v4" / (f"kernel-{args.stage}" if name == "v4" else f"kernel-{args.stage}-{name}")
work.mkdir(exist_ok=True)

src = """import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','rapidfuzz':'rapidfuzz','pyarrow':'pyarrow','cv2':'opencv-python-headless','PIL':'pillow','requests':'requests'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-v4-code');(code/'indra_ocr').mkdir(parents=True,exist_ok=True)
"""
for path in sorted((ROOT / "vendor/indra/indra_ocr").glob("*.py")):
    src += f"(code/'indra_ocr'/{path.name!r}).write_text({path.read_text(encoding='utf8')!r},encoding='utf8')\n"
files = [ROOT / "v3/common.py", ROOT / "v4" / f"{args.stage}.py"]
if args.stage == "train":
    files.append(ROOT / "indra_rtl_trainable.py")
for p in files:
    text = p.read_text(encoding="utf8")
    assert "KGAT_" not in text
    compile(text, p.name, "exec")
    src += f"(code/{p.name!r}).write_text({text!r},encoding='utf8')\n"
src += f"env=dict(os.environ,PYTHONPATH=str(code),RUN={name!r})\n"
src += f"subprocess.check_call([sys.executable,'-u',str(code/{args.stage + '.py'!r})],env=env,cwd=str(code))\n"
(work / "run.py").write_text(src, encoding="utf8")

meta = {"id": f"sharifi76/indra-persian-{name}-{args.stage}", "title": f"Indra Persian {name} {args.stage.title()}",
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_internet": True, "enable_gpu": args.stage == "train",
        "dataset_sources": ["sharifi76/persian-ocr-large-mixed-v3"] + (["sharifi76/persian-free-fonts-ocr"] if args.stage == "render" else []),
        "kernel_sources": ["sharifi76/indra-persian-v4-render"] if args.stage == "train" else [],
        "competition_sources": [], "model_sources": []}
if args.stage == "train":
    meta["machine_shape"] = "NvidiaTeslaT4"
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "receipts/holdout"))
from guard import check_sources as _check_sources  # noqa: E402
_check_sources(meta)
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("bundled", args.stage, len(src), "bytes", hashlib.sha256(src.encode()).hexdigest()[:16])
