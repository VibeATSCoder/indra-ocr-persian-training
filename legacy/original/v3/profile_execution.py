"""Compare whole-batch gathering and replica-local loss on actual line crops."""
import copy,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import requests
import torch
from torch.nn import functional as F
from indra_rtl_trainable import IndraRTL
assert torch.cuda.device_count()==2;torch.set_num_threads(2)
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
padded=torch.zeros((len(rows),int(sizes.max())),dtype=torch.long)
for i,r in enumerate(rows):padded[i,:len(r['visual'])]=torch.tensor([ids[c] for c in r['visual']])
path=Path('/tmp/rec_rtl.onnx');response=requests.get('https://huggingface.co/datalab-to/indra-ocr/resolve/bd02167fdf2005a19962cc15188f8496a2c87d3b/rec_rtl.onnx',timeout=120);response.raise_for_status();path.write_bytes(response.content)
model=IndraRTL.from_onnx(path)
if len(chars)>1355:model.add_characters(chars[1355:],chars.index(' ')+1)
model.cuda().eval();teacher=copy.deepcopy(model).requires_grad_(False).eval()
x=torch.from_numpy(x).cuda();widths=torch.from_numpy(widths).cuda()

class LocalLoss(torch.nn.Module):
    def __init__(self,student,teacher):super().__init__();self.student=student;self.teacher=teacher
    def forward(self,x,widths,targets,sizes):
        logits,lengths=self.student(x,widths)
        with torch.no_grad():old,_=self.teacher(x,widths)
        loss=F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),targets,lengths.cpu(),sizes.cpu(),reduction='none',zero_infinity=False)
        mask=torch.arange(logits.shape[1],device=x.device)[None,:]<lengths[:,None]
        kl=(F.kl_div(F.log_softmax(logits[:,:,:1356].float()/2,-1),F.softmax(old[:,:,:1356].float()/2,-1),reduction='none').sum(-1)*mask).sum()*4
        return torch.stack([(loss/sizes).sum(),loss.new_tensor(len(x)),kl,mask.sum()])[None,:]
student_parallel=torch.nn.DataParallel(model);teacher_parallel=torch.nn.DataParallel(teacher)
local=torch.nn.DataParallel(LocalLoss(model,teacher));padded=padded.cuda();gpu_sizes=sizes.cuda()
def global_loss():
    with torch.autocast('cuda',dtype=torch.float16):
        logits,lengths=student_parallel(x,widths)
        with torch.no_grad():old,_=teacher_parallel(x,widths)
    ctc=F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),labels.cuda(),lengths.cpu(),sizes,zero_infinity=False)
    mask=torch.arange(logits.shape[1],device=x.device)[None,:]<lengths[:,None]
    kd=(F.kl_div(F.log_softmax(logits[:,:,:1356].float()/2,-1),F.softmax(old[:,:,:1356].float()/2,-1),reduction='none').sum(-1)*mask).sum()/mask.sum()*4
    return ctc+.15*kd
def local_loss():
    with torch.autocast('cuda',dtype=torch.float16):s=local(x,widths,padded,gpu_sizes).sum(0)
    return s[0]/s[1]+.15*s[2]/s[3]
results={};parameter_gradients={}
for name,objective in [('global_gather',global_loss),('local_loss',local_loss)]:
    times=[]
    for repeat in range(3):
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
        loss=objective();loss.backward();torch.cuda.synchronize()
        if repeat:times.append(time.monotonic()-t)
    parameter_gradients[name]={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
    results[name]={'seconds_per_batch':float(np.median(times)),'loss':loss.item()}
    print(name,json.dumps(results[name]),flush=True)
delta=0
for name,a in parameter_gradients['global_gather'].items():
    b=parameter_gradients['local_loss'][name]
    torch.testing.assert_close(a,b,atol=5e-4,rtol=5e-3)
    delta=max(delta,float((a-b).abs().max()))
assert abs(results['global_gather']['loss']-results['local_loss']['loss'])<2e-4
results.update({'samples':len(rows),'max_width':int(max_width),'full_objective_equivalence_passed':True,'max_parameter_gradient_delta':delta,'local_loss_speedup':results['global_gather']['seconds_per_batch']/results['local_loss']['seconds_per_batch']})
x=x.repeat(2,1,1,1);widths=widths.repeat(2);padded=padded.repeat(2,1);gpu_sizes=gpu_sizes.repeat(2)
torch.cuda.reset_peak_memory_stats();times=[]
for repeat in range(3):
    model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
    value=local_loss();value.backward();torch.cuda.synchronize()
    assert torch.isfinite(value)
    if repeat:times.append(time.monotonic()-start)
results['local_batch256']={'seconds_per_batch':float(np.median(times)),'samples_per_second':256/float(np.median(times)),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
Path('/kaggle/working/execution-profile.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
