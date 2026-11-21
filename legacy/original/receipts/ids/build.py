"""Bundle an ID card stage into a Kaggle script kernel. Never bundles credentials.

    .control-venv/bin/python receipts/ids/build.py          # free CPU: line data; then kaggle_cli.py kernels push -p receipts/ids/kernel-prepare
    .control-venv/bin/python receipts/ids/build.py card     # T4 x2: the card finder; push with --accelerator NvidiaTeslaT4
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) > 1 and sys.argv[1] == "card":
    work = ROOT / "receipts/ids/kernel-card"
    work.mkdir(exist_ok=True)
    src = """import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','cv2':'opencv-python-headless'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-card-code');code.mkdir(parents=True,exist_ok=True)
"""
    for p in (ROOT / "receipts/ids/train_card.py", ROOT / "receipts/ids/midv.py"):
        text = p.read_text(encoding="utf8")
        assert "KGAT_" not in text and "hf_" + "eV" not in text
        compile(text, p.name, "exec")
        src += f"(code/{p.name!r}).write_text({text!r},encoding='utf8')\n"
    src += "env=dict(os.environ,PYTHONPATH=str(code))\nsubprocess.check_call([sys.executable,'-u',str(code/'train_card.py')],env=env,cwd=str(code))\n"
    (work / "run.py").write_text(src, encoding="utf8")
    meta = {"id": "sharifi76/indra-persian-ids-card", "title": "Indra Persian IDs Card", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_internet": True, "enable_gpu": True,
            "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"}
    (work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("bundled card", len(src), "bytes", hashlib.sha256(src.encode()).hexdigest()[:16])
    raise SystemExit
work = ROOT / "receipts/ids/kernel-prepare"
work.mkdir(exist_ok=True)
src = """import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','rapidfuzz':'rapidfuzz','pyarrow':'pyarrow','cv2':'opencv-python-headless','PIL':'pillow','requests':'requests'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-ids-code');(code/'indra_ocr').mkdir(parents=True,exist_ok=True)
"""
for path in sorted((ROOT / "vendor/indra/indra_ocr").glob("*.py")):
    src += f"(code/'indra_ocr'/{path.name!r}).write_text({path.read_text(encoding='utf8')!r},encoding='utf8')\n"
files = [ROOT / "v3/common.py", ROOT / "receipts/ids/prepare.py", ROOT / "receipts/ids/midv.py", ROOT / "receipts/render.py", ROOT / "receipts/fields.py", ROOT / "receipts/layout.py",
         ROOT / "receipts/real/lowq.py", ROOT / "receipts/real/enhance_test.py"]
for p in files:
    text = p.read_text(encoding="utf8")
    assert "KGAT_" not in text and "hf_" + "eV" not in text
    compile(text, p.name, "exec")
    src += f"(code/{p.name!r}).write_text({text!r},encoding='utf8')\n"
src += "env=dict(os.environ,PYTHONPATH=str(code))\n"
src += "subprocess.check_call([sys.executable,'-u',str(code/'prepare.py')],env=env,cwd=str(code))\n"
(work / "run.py").write_text(src, encoding="utf8")
meta = {"id": "sharifi76/indra-persian-ids-prepare", "title": "Indra Persian IDs Prepare", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_internet": True, "enable_gpu": False,
        "dataset_sources": ["sharifi76/persian-ocr-large-mixed-v3", "sharifi76/persian-free-fonts-ocr", "sharifi76/persian-receipt-fonts-ocr", "sharifi76/persian-receipt-fonts-384"],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "receipts/holdout"))
from guard import check_sources as _check_sources  # noqa: E402
_check_sources(meta)
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print("bundled", len(src), "bytes", hashlib.sha256(src.encode()).hexdigest()[:16])
