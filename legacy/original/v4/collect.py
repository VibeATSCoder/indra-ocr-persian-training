"""Download a finished Kaggle training run and unpack its model bundles.

    .control-venv/bin/python v4/collect.py sharifi76/indra-persian-v4-train v4/results
    .control-venv/bin/python v4/collect.py sharifi76/indra-persian-large-train v3/results
"""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
kernel, out = sys.argv[1], ROOT / sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
subprocess.check_call([sys.executable, str(ROOT / "kaggle_cli.py"), "kernels", "output", kernel, "-p", str(out),
                       "--file-pattern", r".*(\.zip|\.json|\.jsonl|\.md|\.log)$"])
for z in out.glob("*.zip"):
    dest = out / z.stem
    with zipfile.ZipFile(z) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert (dest / name).resolve().is_relative_to(dest.resolve()), "unsafe path in archive"
        archive.extractall(dest)
    # bundles only replace rec_rtl.onnx/meta.json; make sure every other Indra file is present
    for f in (ROOT / "base-model").iterdir():
        if f.suffix in (".onnx", ".npz") and f.name not in ("smoke-updated.onnx", "roundtrip.onnx") and not (dest / f.name).exists():
            (dest / f.name).write_bytes(f.read_bytes())
    print("bundle", dest)
res = out / "results.json"
if res.exists():
    r = json.loads(res.read_text(encoding="utf8"))
    keys = [k for k in r if k.startswith(("baseline_test", "test_", "selected_test"))]
    print("selected:", r.get("selected") or r.get("selected_candidate", {}).get("config"))
    for k in keys:
        v = r[k]
        print(k, {d: round(v[d]["cer"], 4) for d in v if isinstance(v[d], dict) and "cer" in v[d] and "/" not in d})
