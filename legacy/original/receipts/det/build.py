"""Bundle a detector stage into a Kaggle script kernel. Never bundles credentials.

    .control-venv/bin/python receipts/det/build.py render   # free CPU: synthetic pages
    .control-venv/bin/python receipts/det/build.py train    # T4 x2
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["render", "train"])
args = ap.parse_args()
work = ROOT / "receipts/det" / f"kernel-{args.stage}"
work.mkdir(exist_ok=True)

src = """import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','cv2':'opencv-python-headless','PIL':'pillow','requests':'requests'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-det-code');(code/'indra_ocr').mkdir(parents=True,exist_ok=True)
"""
script = {"render": "pages.py", "train": "train.py"}[args.stage]
if args.stage == "render":
    files = [ROOT / "receipts/det/pages.py", ROOT / "receipts/fields.py", ROOT / "receipts/layout.py"]
    for path in sorted((ROOT / "vendor/indra/indra_ocr").glob("*.py")):
        src += f"(code/'indra_ocr'/{path.name!r}).write_text({path.read_text(encoding='utf8')!r},encoding='utf8')\n"
else:
    files = [ROOT / "receipts/det/train.py", ROOT / "receipts/det/det_trainable.py", ROOT / "vendor/indra/indra_ocr/det_post.py", ROOT / "receipts/real/lowq.py"]
for p in files:
    text = p.read_text(encoding="utf8")
    assert "KGAT_" not in text and "hf_" + "eV" not in text
    compile(text, p.name, "exec")
    src += f"(code/{p.name!r}).write_text({text!r},encoding='utf8')\n"
src += "env=dict(os.environ,PYTHONPATH=str(code))\n"
src += f"subprocess.check_call([sys.executable,'-u',str(code/{script!r})],env=env,cwd=str(code))\n"
(work / "run.py").write_text(src, encoding="utf8")

meta = {"id": f"sharifi76/indra-persian-det-{args.stage}", "title": f"Indra Persian Det {args.stage.title()}", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_internet": True, "enable_gpu": args.stage == "train",
        "dataset_sources": ["sharifi76/persian-free-fonts-ocr", "sharifi76/persian-receipt-fonts-ocr", "sharifi76/persian-receipt-fonts-384"] if args.stage == "render" else ["sharifi76/persian-receipt-real-pages"],
        "kernel_sources": ["sharifi76/indra-persian-det-render"] if args.stage == "train" else [], "competition_sources": [], "model_sources": []}
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "receipts/holdout"))
from guard import check_sources as _check_sources  # noqa: E402
_check_sources(meta)
if args.stage == "train":
    meta["machine_shape"] = "NvidiaTeslaT4"
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("bundled", args.stage, len(src), "bytes", hashlib.sha256(src.encode()).hexdigest()[:16])
