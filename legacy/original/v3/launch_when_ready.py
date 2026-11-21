"""Bounded CPU-preparation wait and source validation before reserving a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import kaggle_cli
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate();last=None;start=time.monotonic()
while time.monotonic()-start<10800:
    state=api.kernels_status('sharifi76/indra-persian-large-prepare/1');value=str(state.status)
    if value!=last:print('Preparation:',value,flush=True);last=value
    if value.endswith('COMPLETE'):break
    if value.endswith(('ERROR','CANCELED','CANCELLED')):raise SystemExit('Preparation failed: '+state.failure_message)
    time.sleep(20)
else:raise SystemExit('Preparation monitoring window exhausted')
data=ROOT/'v3/data';data.mkdir(exist_ok=True)
api.kernels_output('sharifi76/indra-persian-large-prepare/1',path=str(data),file_pattern=r'^(manifest.json|ATTRIBUTION.md|font-licenses.json)$',quiet=True)
manifest=json.loads((data/'manifest.json').read_text(encoding='utf8'))
print('Prepared rows:',manifest['rows'],json.dumps(manifest['counts']),flush=True)
subprocess.check_call([sys.executable,str(ROOT/'v3/build.py'),'train'])
for attempt in range(8):
    check=subprocess.run([sys.executable,str(ROOT/'kaggle_cli.py'),'kernels','push','-p',str(ROOT/'v3/kernel-train'),'--no-run'],capture_output=True,text=True,encoding='utf8')
    if check.returncode==0 and 'not valid kernel sources' not in check.stdout+check.stderr:
        print(check.stdout.strip(),flush=True);break
    if attempt==7:raise SystemExit('Prepared source did not become attachable')
    print('Waiting for Kaggle source publication',flush=True);time.sleep(40)
subprocess.check_call([sys.executable,str(ROOT/'kaggle_cli.py'),'kernels','push','-p',str(ROOT/'v3/kernel-train'),'--accelerator','NvidiaTeslaT4','-t','5000'])
print('GPU run submitted with prepared data attached',flush=True)
