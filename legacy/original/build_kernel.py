"""Bundle the reviewed source into a self-contained private Kaggle script."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE_REV = "e20355319b5aa5705a876a8d2b40aca606d7df5f"

bootstrap = '''import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import concurrent.futures
import urllib.request

required = {"onnx":"onnx>=1.16", "onnxruntime":"onnxruntime>=1.18", "rapidfuzz":"rapidfuzz", "huggingface_hub":"huggingface_hub", "pyarrow":"pyarrow", "pypdfium2":"pypdfium2", "cv2":"opencv-python-headless"}
missing = [package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*missing])
os.environ["HF_HUB_DISABLE_XET"] = "1"
work = Path("/kaggle/working")
code = Path("/tmp/indra-code")
(code/"indra_ocr").mkdir(parents=True,exist_ok=True)
modules = ["__init__", "__main__", "cli", "det_post", "document", "layout_post", "line_mask", "lm", "math", "models", "order_post", "parallel", "pipeline", "rec_preproc", "rtl", "textlayer", "viewer", "vocab"]
def fetch(name):
    url = "https://raw.githubusercontent.com/datalab-to/indra/REVISION/indra_ocr/" + name + ".py"
    with urllib.request.urlopen(url,timeout=60) as response:
        (code/"indra_ocr"/(name+".py")).write_bytes(response.read())
with concurrent.futures.ThreadPoolExecutor(6) as pool:
    list(pool.map(fetch,modules))
os.environ["PYTHONPATH"] = str(code)
'''.replace("REVISION", CODE_REV)
for filename in ("indra_rtl_trainable.py", "train.py"):
    text = (ROOT/filename).read_text(encoding="utf-8")
    compile(text, filename, "exec")
    bootstrap += f"(work/{filename!r}).write_text({text!r},encoding='utf-8')\n"
bootstrap += "subprocess.check_call([sys.executable,'-u',str(work/'train.py')])\n"
book_run = os.environ.get("OCR_BUILD_BOOK_RUN") == "1"
if book_run:
    settings = {"OCR_EPOCHS":"5","OCR_MAX_LR":"0.00005","OCR_FREEZE_BACKBONE":"1","OCR_CHECKPOINT_METRIC":"balanced_book","OCR_BOOK_LONG_LINES":"1"}
    bootstrap = bootstrap.replace("subprocess.check_call([sys.executable,'-u',str(work/'train.py')])", "os.environ.update("+repr(settings)+")\nsubprocess.check_call([sys.executable,'-u',str(work/'train.py')])")
directory = ROOT/("training_books" if book_run else "training")
directory.mkdir(exist_ok=True)
(directory/"run_training.py").write_text(bootstrap,encoding="utf-8")
metadata = {"id":"sharifi76/indra-persian-ocr-finetune", "title":"Indra Persian OCR Finetune", "code_file":"run_training.py", "language":"python", "kernel_type":"script", "is_private":True, "enable_gpu":True,"enable_internet":True,"machine_shape":"NvidiaTeslaT4", "dataset_sources":[], "competition_sources":[], "kernel_sources":["sharifi76/persian-ocr-corpus-preparation"]}
if book_run:
    metadata["id"] = "sharifi76/indra-persian-books-finetune"
    metadata["title"] = "Indra Persian Books Finetune"
    metadata["kernel_sources"] = ["sharifi76/persian-ocr-book-line-preparation"]
(directory/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2))
print("Bundled",len(bootstrap),"bytes; sha256",hashlib.sha256(bootstrap.encode()).hexdigest())
