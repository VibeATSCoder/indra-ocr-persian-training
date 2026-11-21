"""Measure CTC backend, numerical equivalence and throughput on actual crops."""
import json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from torch.nn import functional as F
from indra_rtl_trainable import IndraRTL
from ctc_objective import padded_ctc
assert torch.cuda.is_available();torch.set_num_threads(2)
manifest_path=next(Path('/kaggle/input').rglob('manifest.json'))
m=json.loads(manifest_path.read_text());chars=m['characters'];ids={c:i+1 for i,c in enumerate(chars)}
shard=next(s for s in m['shards'] if s['split']=='train');p=pq.ParquetFile(manifest_path.parent/shard['file'])
rows=[]
for index in [0,min(40,p.num_row_groups-1)]:rows.extend(p.read_row_group(index).to_pylist()[:64])
widths=np.asarray([r['width'] for r in rows],np.int64);max_width=(max(widths)+3)//4*4
x=np.zeros((len(rows),1,32,max_width),np.float32)
for i,r in enumerate(rows):x[i,0,:,:r['width']]=np.frombuffer(r['ink'],np.uint8).reshape(32,r['width'])/255
labels=torch.tensor([ids[c] for r in rows for c in r['visual']],dtype=torch.long)
sizes=torch.tensor([len(r['visual']) for r in rows],dtype=torch.long)
path=Path('/tmp/rec_rtl.onnx');response=requests.get('https://huggingface.co/datalab-to/indra-ocr/resolve/bd02167fdf2005a19962cc15188f8496a2c87d3b/rec_rtl.onnx',timeout=120);response.raise_for_status();path.write_bytes(response.content)
model=IndraRTL.from_onnx(path);old=1355
if len(chars)>old:model.add_characters(chars[old:],chars.index(' ')+1)
model.cuda().eval();x=torch.from_numpy(x).cuda();widths=torch.from_numpy(widths).cuda()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):raw,lengths=model(x,widths)
native=lambda logits:F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),labels.cuda(),lengths.cpu(),sizes,zero_infinity=False)
fast=lambda logits:padded_ctc(logits,lengths,labels,sizes)
results={};gradients={}
for name,objective in [('native',native),('padded_cudnn',fast)]:
    y=raw.float().detach().clone().requires_grad_(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        loss=objective(y);loss.backward();torch.cuda.synchronize()
    gradients[name]=y.grad.detach().cpu();times=[]
    for _ in range(3):
        y=raw.float().detach().clone().requires_grad_(True);torch.cuda.synchronize();t=time.monotonic()
        value=objective(y);value.backward();torch.cuda.synchronize();times.append(time.monotonic()-t)
    results[name]={'loss':loss.item(),'seconds_per_forward_backward':float(np.median(times)),
        'operators':[e.key for e in profile.key_averages() if 'ctc' in e.key]}
    print(name,json.dumps(results[name]),flush=True)
assert abs(results['native']['loss']-results['padded_cudnn']['loss'])<2e-4
np.testing.assert_allclose(gradients['native'].numpy(),gradients['padded_cudnn'].numpy(),atol=2e-5,rtol=1e-3)
for parallel in [False,True]:
    network=torch.nn.DataParallel(model) if parallel else model
    for name,objective in [('native',native),('padded_cudnn',fast)]:
        times=[]
        for repeat in range(3):
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.float16):pred,_=network(x,widths)
            objective(pred).backward();torch.cuda.synchronize()
            if repeat:times.append(time.monotonic()-start)
        results[name]['two_gpu' if parallel else 'single_gpu']=float(np.median(times))
        print('TRAIN TIMING',parallel,name,times,flush=True)
results.update({'samples':len(rows),'max_width':int(max_width),'max_gradient_delta':float((gradients['native']-gradients['padded_cudnn']).abs().max()),'equivalence_passed':True,
    'speedup':results['native']['seconds_per_forward_backward']/results['padded_cudnn']['seconds_per_forward_backward']})
Path('/kaggle/working/ctc-profile.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
