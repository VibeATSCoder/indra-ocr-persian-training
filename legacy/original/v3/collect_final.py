"""Freeze a validation-only trial choice before inspecting any test accuracy."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kaggle_cli
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
kernels={'legacy':'sharifi76/indra-persian-large-train','fast':'sharifi76/indra-persian-large-train-fast'}
last={};start=time.monotonic();complete=set()
while time.monotonic()-start<18000:
    for name,kernel in kernels.items():
        if name in complete:continue
        r=api.kernels_status(kernel);state=str(r.status)
        if state!=last.get(name):print(name,state,flush=True);last[name]=state
        if state.endswith('COMPLETE'):complete.add(name)
        if state.endswith(('ERROR','CANCELLED','CANCELED')):raise SystemExit(name+' experiment failed: '+r.failure_message)
    if len(complete)==len(kernels):break
    time.sleep(20)
else:raise SystemExit('Trial monitoring deadline reached')

choices={}
for name,kernel in kernels.items():
    path=ROOT/'v3'/('run-'+name);path.mkdir(exist_ok=True)
    api.kernels_output(kernel,path=str(path),file_pattern=r'^(results.json|training-history.json)$',quiet=True,page_size=200)
    # Test fields are never referenced by this selection function.
    data=json.loads((path/'results.json').read_text(encoding='utf8'))
    chosen=data['selected_candidate']
    choices[name]={'kernel':kernel,'validation_score':chosen['score'],
        'config':chosen['config'],'epoch':chosen['epoch'],
        'result_sha256':hashlib.sha256((path/'results.json').read_bytes()).hexdigest()}
    print('VALIDATION ONLY',name,json.dumps(choices[name]),flush=True)
winner=min(choices,key=lambda name:(choices[name]['validation_score'],name!='fast'))
decision={'criterion':'Minimum prespecified weighted validation score; original control eligible within each trial',
    'test_metrics_used':False,'frozen_before_test_inspection':True,'winner':winner,'trials':choices}
(ROOT/'v3/selection-decision.json').write_text(json.dumps(decision,indent=2))

out=ROOT/'v3/results';out.mkdir(exist_ok=True)
api.kernels_output(kernels[winner],path=str(out),file_pattern=r'.*(\.zip|\.pt|\.json|\.jsonl|\.md)$',quiet=True,page_size=200)
with zipfile.ZipFile(out/'indra-persian-large-model.zip') as archive:
    assert archive.testzip() is None
    assert all((out/'model'/name).resolve().is_relative_to((out/'model').resolve()) for name in archive.namelist())
    archive.extractall(out/'model')
shutil.copyfile(ROOT/'v3/selection-decision.json',out/'selection-decision.json')
result=json.loads((out/'results.json').read_text(encoding='utf8'))
print('FROZEN FINAL CHOICE',winner,json.dumps({'candidate':result['selected_candidate']['config'],
    'epoch':result['selected_candidate']['epoch'],'test_baseline':result['baseline_test'],
    'test_selected':result['selected_test']}),flush=True)
env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'vendor/indra');env['PYTHONUTF8']='1'
with (out/'external-pages.log').open('w',encoding='utf8') as log:
    subprocess.check_call([sys.executable,str(ROOT/'evaluate_pages.py'),'--model-dir',str(out/'model'),
        '--out',str(out/'external-pages.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
print('Final 100-page external comparison complete',flush=True)
