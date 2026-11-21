"""Build an upload-ready private Kaggle kernel from the reusable trainer."""

import base64
import io
import json
import zipfile
from pathlib import Path


def build_kernel(root, output, owner, slug, datasets, config_path):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh kernel directory: {output}")
    if not datasets or any("receipts-eval" in name for name in datasets):
        raise ValueError(
            "Provide private model/data sources; never attach receipts-eval to training"
        )
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    from .config import load_config

    load_config(config_path)  # Refuse unknown keys, including accidental credential fields.
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted((root / "src").rglob("*")):
            if path.is_file() and (path.suffix == ".py" or path.name == "LICENSE.txt"):
                handle.writestr(path.relative_to(root).as_posix(), path.read_bytes())
        handle.writestr("pyproject.toml", (root / "pyproject.toml").read_bytes())
        handle.writestr("README.md", "# Indra training kernel\n")
    encoded = base64.b64encode(archive.getvalue()).decode("ascii")
    source = """import base64, io, json, os, subprocess, sys, zipfile
from pathlib import Path
root = Path('/tmp/indra-training-code')
root.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(__ARCHIVE__))) as package:
    package.extractall(root)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', str(root) + '[train]'])
inputs = Path('/kaggle/input')
if any('receipts-eval' in str(p) or p.name == 'labels-r2' for p in inputs.rglob('*')):
    raise RuntimeError('A sealed evaluation source is mounted in this training kernel')
manifests = []
for p in inputs.rglob('manifest.json'):
    try:
        if json.loads(p.read_text()).get('format_version') == 1:
            manifests.append(p)
    except (ValueError, OSError):
        continue
models = list(inputs.rglob('rec_rtl.onnx'))
if len(manifests) != 1 or len(models) != 1:
    raise RuntimeError('Attach exactly one prepared dataset and one base-model bundle')
config = __SETTINGS__
config.update(dataset_dir=str(manifests[0].parent), base_model_dir=str(models[0].parent), output_dir='/kaggle/working/run')
path = root / 'run-config.json'
path.write_text(json.dumps(config))
subprocess.check_call([sys.executable, '-m', 'indra_training.cli', 'train', '--config', str(path)])
"""
    source = source.replace("__ARCHIVE__", repr(encoded)).replace("__SETTINGS__", repr(settings))
    compile(source, "run.py", "exec")
    output.mkdir(parents=True)
    (output / "run.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": "Persian Indra OCR fine-tuning",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": datasets,
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata
