"""Exercise the new regularized loss on actual Persian image/label pairs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/indra'),str(ROOT),str(ROOT/'v3')]
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from common import clean,crop_image,fits
from indra_ocr.rec_preproc import pad_batch
from indra_ocr.rtl import visual_label
from indra_rtl_trainable import IndraRTL,verify_parity
torch.set_num_threads(2)
chars=json.loads(json.loads((ROOT/'base-model/meta.json').read_text())['vocab_rtl'])['chars']
teacher=IndraRTL.from_onnx(ROOT/'base-model/rec_rtl.onnx');teacher.requires_grad_(False)
student=IndraRTL.from_onnx(ROOT/'base-model/rec_rtl.onnx');student.add_characters(['\u200c']);chars+=['\u200c'];ids={c:i+1 for i,c in enumerate(chars)}
rows=pq.ParquetFile(ROOT/'book-corpus/test.parquet').read_row_group(0).to_pylist()[:12]
items=[]
for row in rows:
    ink=crop_image(row['image']);text=clean(row['text']);label=visual_label(text)
    if fits(ink,text) and not set(label)-set(chars):items.append((ink,[ids[c] for c in label]))
x,widths=pad_batch([im.astype(np.float32)/255 for im,label in items]);x=torch.from_numpy(x);widths=torch.from_numpy(widths)
anchors={n:p.detach().clone() for n,p in student.named_parameters()}
student.train();logits,lengths=student(x,widths)
with torch.no_grad():old,_=teacher(x,widths)
labels=torch.tensor([c for im,label in items for c in label]);sizes=torch.tensor([len(label) for im,label in items])
ctc=F.ctc_loss(logits.log_softmax(-1).transpose(0,1),labels,lengths,sizes,zero_infinity=False)
mask=torch.arange(logits.shape[1])[None,:]<lengths[:,None]
kd=(F.kl_div((logits[:,:,:1356]/2).log_softmax(-1),(old/2).softmax(-1),reduction='none').sum(-1)*mask).sum()/mask.sum()*4
anchor=sum((p-anchors[n]).square().sum() for n,p in student.named_parameters())
loss=ctc+.15*kd+.001*anchor;assert torch.isfinite(loss);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
assert all(p.grad is None for p in teacher.parameters())
torch.optim.AdamW(student.parameters(),lr=2e-5).step()
path=ROOT/'artifacts/v3-smoke.onnx';student.eval();student.write_onnx(ROOT/'base-model/rec_rtl.onnx',path)
result={'samples':len(items),'ctc':float(ctc.detach()),'kl':float(kd.detach()),'all_student_gradients_finite':True,'teacher_frozen':True,'export_parity':verify_parity(path,student)}
(ROOT/'v3/objective-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
