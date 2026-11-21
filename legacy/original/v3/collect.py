"""Retrieve the completed new experiment and run the unchanged page benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kaggle_cli
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate();kernel='sharifi76/indra-persian-large-train';start=time.monotonic();last=None
while time.monotonic()-start<18000:
    try:
        r=api.kernels_status(kernel);state=str(r.status)
        if state!=last:print('Large GPU run:',state,flush=True);last=state
        if state.endswith(('ERROR','CANCELLED','CANCELED')):raise SystemExit('GPU experiment failed: '+r.failure_message)
        if state.endswith('COMPLETE'):
            q=ApiListKernelSessionOutputRequest();q.user_name='sharifi76';q.kernel_slug='indra-persian-large-train';q.page_size=200
            with api.build_kaggle_client() as client:files=client.kernels.kernels_api_client.list_kernel_session_output(q)
            if any(f['fileName']=='results.json' for f in files.to_dict().get('files',[])):break
    except SystemExit:raise
    except Exception:pass
    time.sleep(20)
else:raise SystemExit('GPU experiment monitoring window exhausted')
out=ROOT/'v3/results';out.mkdir(exist_ok=True)
api.kernels_output(kernel,path=str(out),file_pattern=r'.*(\.zip|\.pt|\.json|\.jsonl|\.md)$',quiet=True,page_size=200)
with zipfile.ZipFile(out/'indra-persian-large-model.zip') as archive:
    assert archive.testzip() is None
    assert all((out/'model'/n).resolve().is_relative_to((out/'model').resolve()) for n in archive.namelist())
    archive.extractall(out/'model')
result=json.loads((out/'results.json').read_text(encoding='utf8'))
print('SELECTED',json.dumps({'candidate':result['selected_candidate']['config'],'epoch':result['selected_candidate']['epoch'],'test_baseline':result['baseline_test'],'test_selected':result['selected_test']}),flush=True)
env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'vendor/indra');env['PYTHONUTF8']='1'
with (out/'external-pages.log').open('w',encoding='utf8') as log:
    subprocess.check_call([sys.executable,str(ROOT/'evaluate_pages.py'),'--model-dir',str(out/'model'),'--out',str(out/'external-pages.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
print('New external page comparison complete',flush=True)
