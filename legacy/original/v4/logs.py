"""Print filtered live log lines of a running Kaggle script kernel (the CLI's `logs -f` shows nothing).

    .control-venv/bin/python v4/logs.py indra-persian-v4-train [seconds]
"""
import itertools
import json
import os
import sys
import threading

os.environ.setdefault("KAGGLE_API_TOKEN", open(os.path.expanduser("~/.config/tune-ocr/kaggle/access_token")).read().strip())
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionLogsStreamRequest  # noqa: E402

slug = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 20
KEYS = ("START", "CACHE", "STEP", "VALID", "BASELINE", "SHARDS", "SPLITS", "CACHED", "DEVICES", "CANDIDATES", "PARITY", "SELECTION", "FINAL", "Error", "Traceback")
api = KaggleApi()
api.authenticate()
threading.Timer(seconds, lambda: os._exit(0)).start()
with api.build_kaggle_client() as client:
    req = ApiGetKernelSessionLogsStreamRequest()
    req.user_name, req.kernel_slug, req.wait_for_logs_url_seconds = "sharifi76", slug, 30
    for raw in client.kernels.kernels_api_client.get_kernel_session_logs_stream(req).iter_lines():
        line = raw.decode("utf8", "replace")
        if not line.startswith("data:"):
            continue
        try:
            e = json.loads(line[5:])
        except json.JSONDecodeError:
            continue
        d = e.get("data", "").strip()
        if any(k in d for k in KEYS) and (not d.startswith("STEP") or "VALID" in d or int(json.loads(d[5:])["steps"]) % 2000 == 0):
            print(round(e.get("time", 0)), d[:330], flush=True)
os._exit(0)
