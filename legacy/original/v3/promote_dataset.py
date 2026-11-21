"""Create the private larger Kaggle dataset using scoped cloud-to-cloud uploads."""
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kaggle_cli
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest,ApiStartBlobUploadRequest,ApiBlobType
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest,ApiDatasetNewFile
api=KaggleApi();api.authenticate()
manifest_path=ROOT/'v3/data/manifest.json';start=time.monotonic()
while not manifest_path.exists():
    if time.monotonic()-start>10800:raise SystemExit('Preparation did not finish within monitoring window')
    state=str(api.kernels_status('sharifi76/indra-persian-large-prepare/1').status)
    if state.endswith(('ERROR','CANCELLED','CANCELED')):raise SystemExit('Preparation failed; import not attempted')
    time.sleep(20)
manifest=json.loads(manifest_path.read_text(encoding='utf8'))
q=ApiListKernelSessionOutputRequest();q.user_name='sharifi76';q.kernel_slug='indra-persian-large-prepare';q.page_size=200
with api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.list_kernel_session_output(q)
file_urls={r['fileName']:r['url'] for r in response.to_dict()['files']}
sizes={r['file']:r['bytes'] for r in manifest['shards']}
for name in ['manifest.json','font-licenses.json','ATTRIBUTION.md']:
    r=requests.get(file_urls[name],headers={'Range':'bytes=0-0'},timeout=30)
    assert r.status_code==206
    sizes[name]=int(r.headers['Content-Range'].split('/')[-1])
private=Path.home()/'.codex/tmp/tune-ocr-credentials/large-import';private.mkdir(exist_ok=True)
state_file=private/'state.json'
if state_file.exists():state=json.loads(state_file.read_text())
else:
    state=[]
    for name,size in sizes.items():
        request=ApiStartBlobUploadRequest();request.type=ApiBlobType.DATASET;request.name=name;request.content_length=size;request.last_modified_epoch_seconds=int(time.time())
        with api.build_kaggle_client() as client:r=client.blobs.blob_api_client.start_blob_upload(request)
        state.append({'name':name,'size':size,'url':r.create_url,'token':r.token})
    state_file.write_text(json.dumps(state))
plan=[{k:v for k,v in r.items() if k!='token'} for r in state]
code='''import hashlib,json
from pathlib import Path
import requests
PLAN=PLAN_VALUE
source=list(Path('/kaggle/input').rglob('manifest.json'))
assert len(source)==1
root=source[0].parent;results=[]
for item in PLAN:
    path=root/item['name'];assert path.stat().st_size==item['size']
    try:
        response=requests.put(item['url'],headers={'Content-Length':'0','Content-Range':f"bytes */{item['size']}"},timeout=30)
        if response.status_code not in (200,201):
            assert response.status_code==308
            current=response.headers.get('Range','');offset=int(current.split('-')[-1])+1 if current else 0
            with path.open('rb') as stream:
                stream.seek(offset)
                response=requests.put(item['url'],data=stream,headers={'Content-Length':str(item['size']-offset),'Content-Range':f"bytes {offset}-{item['size']-1}/{item['size']}"},timeout=180)
            assert response.status_code in (200,201)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        results.append({'name':item['name'],'bytes':item['size'],'sha256':digest})
        print('Imported',item['name'],flush=True)
    except Exception as error:
        raise RuntimeError('Scoped data import failed for '+item['name']+' ('+type(error).__name__+')') from None
Path('/kaggle/working/import-result.json').write_text(json.dumps(results,indent=2))
'''.replace('PLAN_VALUE',repr(plan))
(private/'import.py').write_text(code)
metadata={'id':'sharifi76/indra-persian-large-import','title':'Indra Persian Large Import','code_file':'import.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':False,'enable_internet':True,'dataset_sources':[],'kernel_sources':['sharifi76/indra-persian-large-prepare'],'competition_sources':[]}
(private/'kernel-metadata.json').write_text(json.dumps(metadata))
for attempt in range(8):
    result=subprocess.run([sys.executable,str(ROOT/'kaggle_cli.py'),'kernels','push','-p',str(private),'--no-run'],capture_output=True,text=True,encoding='utf8')
    if result.returncode==0 and 'not valid kernel sources' not in result.stdout+result.stderr:break
    time.sleep(30)
else:raise SystemExit('Prepared corpus not attachable for import')
api.kernels_push(str(private),timeout=1800)
print('Private cloud import submitted:',len(plan),'files',flush=True)
while time.monotonic()-start<14400:
    r=api.kernels_status('sharifi76/indra-persian-large-import');status=str(r.status)
    if status.endswith('COMPLETE'):break
    if status.endswith(('ERROR','CANCELLED','CANCELED')):raise SystemExit('Cloud import failed: '+r.failure_message)
    time.sleep(20)
else:raise SystemExit('Import monitoring deadline reached')
request=ApiCreateDatasetRequest();request.owner_slug='sharifi76';request.slug='persian-ocr-large-mixed-v3';request.title='Persian OCR Large Mixed Corpus v3';request.license_name='other';request.is_private=True
request.description=f"Private mixed Persian OCR experiment: {manifest['rows']:,} inference-preprocessed line samples. Garshasp CC BY 4.0; Persian-OCR-230k and Arshasb MIT; added Google Fonts renderings use OFL fonts. Component terms and revisions remain in force. Community Argilla images and model transcripts have no declared upstream license; no redistribution rights or human-verification claim are granted. Real-document references are pseudo-labels, filtered by publisher confidence and agreement with the fixed original model; they are not human ground truth. Complete source documents are held out, and the added known-font families are disjoint. Other corpora do not provide per-image font IDs, so global font novelty cannot be guaranteed. Original evaluation labels and the external benchmark are protected. Ink maps are cached for efficient GPU training. Source revisions, rejected samples, splits and transformations are in manifest.json; full font licenses are included."
request.files=[]
for item in state:
    file=ApiDatasetNewFile();file.token=item['token'];request.files.append(file)
with api.build_kaggle_client() as client:result=client.datasets.dataset_api_client.create_dataset(request)
if result.error:raise SystemExit('Dataset creation failed: '+result.error)
public_state={'url':result.url,'rows':manifest['rows'],'private':True,'files':len(state)}
(ROOT/'v3/dataset-created.json').write_text(json.dumps(public_state,indent=2))
print(json.dumps(public_state),flush=True)
