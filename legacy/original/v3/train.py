"""Regularized multi-domain Indra fine-tuning, with validation-only selection."""
import collections
import concurrent.futures
import copy
import json
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import time
import cv2
import numpy as np
import pyarrow.parquet as pq
import requests
from rapidfuzz.distance import Levenshtein
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
from common import clean,canonical
from indra_ocr.rec_preproc import pad_batch
from indra_ocr.rtl import flip
from indra_rtl_trainable import IndraRTL,verify_parity
from local_objective import LocalObjective

assert torch.cuda.is_available(),'GPU required'
SEED=20261102;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.set_num_threads(2);cv2.setNumThreads(1)
OUT=Path('/kaggle/working');TMP=Path('/tmp/large-train');TMP.mkdir(exist_ok=True)
manifest_paths=list(Path('/kaggle/input').rglob('manifest.json'));assert len(manifest_paths)==1
manifest_path=manifest_paths[0];source=manifest_path.parent;manifest=json.loads(manifest_path.read_text())
CHARS=manifest['characters'];ORIGINAL_CLASSES=1356;BATCH=256;EPOCHS=3
TOKEN_IDS={c:i+1 for i,c in enumerate(CHARS)}
DEVICE={'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'torch':torch.__version__,'cuda':torch.version.cuda}
print('DEVICE',json.dumps(DEVICE),flush=True)
MODEL_REV='bd02167fdf2005a19962cc15188f8496a2c87d3b'
base=TMP/'base';base.mkdir(exist_ok=True)
files=['det.onnx','rec.onnx','rec_cjk.onnx','rec_world.onnx','rec_rtl.onnx','order.onnx','orient.onnx','script.onnx','meta.json','lm.npz','lm_cjk.npz','lm_world.npz','LICENSE']
def download(name):
    path=base/name
    response=requests.get(f'https://huggingface.co/datalab-to/indra-ocr/resolve/{MODEL_REV}/{name}',timeout=120);response.raise_for_status();path.write_bytes(response.content)
with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(download,files))
original=base/'rec_rtl.onnx';parity=verify_parity(original);print('PARITY',json.dumps(parity),flush=True)

# Decode and crop on CPU once. GPU workers read precomputed uint8 maps from a
# local memory-mapped file rather than decoding a giant image table in RAM.
records=collections.defaultdict(list);offset=0;ink_file=TMP/'ink.bin'
groups=collections.defaultdict(set);image_sets=collections.defaultdict(set);real_docs=collections.defaultdict(set);font_sets=collections.defaultdict(set)
for number,shard in enumerate(manifest['shards']):
    shutil.copyfile(source/shard['file'],TMP/shard['file'])
    if (number+1)%10==0:print('STAGED cache files',number+1,flush=True)
# New page-held-out sources can repeat short labels from earlier corpora.
# Resolve all such conflicts before training: test > validation > train.
reserved_eval=collections.defaultdict(set)
for shard in manifest['shards']:
    if shard['split'] in ('validation','test'):
        for batch in pq.ParquetFile(TMP/shard['file']).iter_batches(columns=['group'],batch_size=4096):
            reserved_eval[shard['split']].update(batch.column(0).to_pylist())
split_conflicts=collections.Counter()
all_eval_groups=reserved_eval['validation']|reserved_eval['test']
with ink_file.open('wb',buffering=16*1024*1024) as handle:
    for number,shard in enumerate(manifest['shards']):
        local=TMP/shard['file']
        for batch in pq.ParquetFile(local).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if shard['split']=='train' and row['group'] in all_eval_groups:
                    split_conflicts['train_repeated_eval_label']+=1;continue
                if shard['split']=='validation' and row['group'] in reserved_eval['test']:
                    split_conflicts['validation_repeated_test_label']+=1;continue
                data=row.pop('ink');assert len(data)==32*row['width']
                assert hashlib.sha256(data).hexdigest()==row['image_sha256'],'Cached image integrity failure'
                row['offset']=offset;offset+=len(data);handle.write(data)
                row['target']=[TOKEN_IDS[c] for c in row['visual']]
                records[shard['split']].append(row)
                groups[shard['split']].add(row['group']);image_sets[shard['split']].add(row['image_sha256'])
                if row['domain']=='real':real_docs[shard['split']].add(row['doc_id'])
                if row['domain']=='font':font_sets[shard['split']].add(row['doc_id'])
        local.unlink()
        print('CACHE',number+1,'/',len(manifest['shards']),'rows',sum(map(len,records.values())),'GiB',round(offset/2**30,2),flush=True)
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not groups[a]&groups[b],f'Text overlap {a}/{b}'
    assert not image_sets[a]&image_sets[b],f'Image overlap {a}/{b}'
    assert not real_docs[a]&real_docs[b],f'Real source-document overlap {a}/{b}'
    assert not font_sets[a]&font_sets[b],f'Font-family overlap {a}/{b}'
print('SPLITS',json.dumps({k:len(v) for k,v in records.items()}),'real documents',json.dumps({k:len(v) for k,v in real_docs.items()}),flush=True)

class Lines(Dataset):
    def __init__(self,rows,augment=False):self.rows=rows;self.augment=augment;self.data=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if self.data is None:self.data=np.memmap(ink_file,dtype=np.uint8,mode='r')
        row=self.rows[i];ink=np.asarray(self.data[row['offset']:row['offset']+32*row['width']]).reshape(32,row['width'])
        if self.augment and random.random()<.65:
            ink=ink.copy()
            if random.random()<.45:
                scale=random.uniform(.55,1);small=cv2.resize(ink,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA);ink=cv2.resize(small,(ink.shape[1],32),interpolation=cv2.INTER_LINEAR)
            if random.random()<.4:ink=cv2.GaussianBlur(ink,(3,3),random.uniform(.2,.85))
            if random.random()<.35:
                ink=np.clip(ink.astype(np.float32)*random.uniform(.65,1.15)+np.random.normal(0,random.uniform(0,3),ink.shape),0,255).astype(np.uint8)
            if random.random()<.25:
                mat=np.float32([[1,random.uniform(-.025,.025),0],[0,1,random.uniform(-1,1)]])
                ink=cv2.warpAffine(ink,mat,(ink.shape[1],32),borderValue=0)
        return ink,row

def collate(items):
    x,widths=pad_batch([im.astype(np.float32)/255 for im,row in items]);rows=[r for im,r in items]
    sizes=torch.tensor([len(r['target']) for r in rows],dtype=torch.long)
    labels=torch.zeros((len(rows),int(sizes.max())),dtype=torch.long)
    for i,r in enumerate(rows):labels[i,:len(r['target'])]=torch.tensor(r['target'],dtype=torch.long)
    return torch.from_numpy(x),torch.from_numpy(widths),labels,sizes,rows

class BatchPlan(Sampler):
    def __init__(self,rows,train=False):
        self.rows=rows;self.train=train;self.epoch=0
        self.n=min(600000,len(rows)) if train else len(rows)
        count=collections.Counter(r['domain'] for r in rows)
        mass={'real':.15,'font':.20,'arshasb':.15,'persian230k':.15,'garshasp':.35}
        self.p=np.asarray([mass[r['domain']]/count[r['domain']] for r in rows]);self.p/=self.p.sum()
    def __len__(self):return math.ceil(self.n/BATCH)
    def __iter__(self):
        rng=np.random.default_rng(SEED+self.epoch);self.epoch+=1
        indices=rng.choice(len(self.rows),self.n,replace=True,p=self.p) if self.train else np.arange(self.n)
        batches=[]
        for start in range(0,self.n,8192):
            pool=sorted(indices[start:start+8192].tolist(),key=lambda i:self.rows[i]['width'])
            batches.extend(pool[i:i+BATCH] for i in range(0,len(pool),BATCH))
        if self.train:rng.shuffle(batches)
        yield from batches

def loader(split,train=False):
    return DataLoader(Lines(records[split],train),batch_sampler=BatchPlan(records[split],train),collate_fn=collate,num_workers=4,pin_memory=True,persistent_workers=True)
val=loader('validation');train_loader=loader('train',True)
def parallel(model):return torch.nn.DataParallel(model) if torch.cuda.device_count()>1 else model
def unwrap(model):return model.module if hasattr(model,'module') else model
def decode(logits,lengths,chars):
    seqs=logits.argmax(-1).cpu().numpy();texts=[]
    for seq,length in zip(seqs,lengths.cpu().tolist()):
        last=-1;out=[]
        for token in seq[:length]:
            if token and token!=last:out.append(chars[int(token)-1])
            last=token
        texts.append(clean(flip(''.join(out))))
    return texts

@torch.inference_mode()
def evaluate(model,data,chars,save=None):
    model.eval();sums=collections.defaultdict(collections.Counter);predictions=[]
    for x,widths,labels,sizes,rows in data:
        with torch.autocast('cuda',dtype=torch.float16):logits,lengths=model(x.cuda(non_blocking=True),widths.cuda(non_blocking=True))
        for row,prediction in zip(rows,decode(logits,lengths,chars)):
            text=row['text']
            metrics={'char_edits':Levenshtein.distance(text,prediction),'chars':len(text),'word_edits':Levenshtein.distance(text.split(),prediction.split()),'words':len(text.split()),'canonical_edits':Levenshtein.distance(canonical(text),canonical(prediction)),'canonical_chars':len(canonical(text)),'samples':1,'exact':int(text==prediction)}
            for domain in ['all',row['domain']]:sums[domain].update(metrics)
            if save:predictions.append({'source':row['source'],'document':row['doc_id'],'image_sha256':row['image_sha256'],'reference':text,'prediction':prediction})
    result={k:{**dict(v),'cer':v['char_edits']/max(1,v['chars']),'wer':v['word_edits']/max(1,v['words']),'canonical_cer':v['canonical_edits']/max(1,v['canonical_chars'])} for k,v in sums.items()}
    if save:save.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in predictions),encoding='utf8')
    return result

def score(metrics):
    # Real labels are pseudo-ground-truth. Keep the known-text font probe as an
    # independent anchor, rather than selecting on synthetic accuracy alone.
    weights={'font':.50,'real':.40,'garshasp':.04,'arshasb':.03,'persian230k':.03}
    valid={k:w for k,w in weights.items() if k in metrics and metrics[k]['samples']>=100}
    return sum(w*(.7*metrics[k]['cer']+.3*metrics[k]['canonical_cer']) for k,w in valid.items())/sum(valid.values())

teacher=IndraRTL.from_onnx(original).cuda().eval()
original_chars=json.loads(json.loads((base/'meta.json').read_text())['vocab_rtl'])['chars']
teacher.requires_grad_(False);teacher_parallel=parallel(teacher)
baseline_val=evaluate(teacher_parallel,val,original_chars)
print('BASELINE VALIDATION',json.dumps(baseline_val),flush=True)
history=[];best_score=score(baseline_val);best_info={'config':{'name':'original_control'},'epoch':0,'score':best_score,'validation':baseline_val}
control=IndraRTL.from_onnx(original);control.add_characters(CHARS[len(original_chars):],original_chars.index(' ')+1)
torch.save({'state_dict':control.state_dict(),'characters':CHARS,'selection':best_info},OUT/'best-checkpoint.pt');del control
started=time.monotonic();max_seconds=3600
configs=[{'name':'adapt','lr':2e-5,'anchor':1e-3,'distill':.15},{'name':'conservative','lr':5e-6,'anchor':5e-3,'distill':.30}]
for config_index,config in enumerate(configs):
    if time.monotonic()-started>max_seconds:break
    candidate_deadline=time.monotonic()+(max_seconds-(time.monotonic()-started))/(len(configs)-config_index)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    train_loader=loader('train',True)
    student=IndraRTL.from_onnx(original).cuda();student.add_characters(CHARS[len(original_chars):],original_chars.index(' ')+1)
    anchors={name:p.detach().clone() for name,p in student.named_parameters()}
    wrapped=parallel(student);objective=parallel(LocalObjective(student,teacher,ORIGINAL_CLASSES))
    optimizer=torch.optim.AdamW(student.parameters(),lr=config['lr'],weight_decay=0)
    total_steps=EPOCHS*len(train_loader);scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config['lr'],total_steps=total_steps,pct_start=.05,div_factor=10,final_div_factor=20)
    scaler=torch.amp.GradScaler('cuda')
    for epoch in range(EPOCHS):
        wrapped.train();objective.train();running=collections.Counter();batches=0;examples=0
        for x,widths,labels,sizes,rows in train_loader:
            if time.monotonic()>candidate_deadline:break
            x=x.cuda(non_blocking=True);widths=widths.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                sums=objective(x,widths,labels.cuda(non_blocking=True),sizes.cuda(non_blocking=True)).sum(0)
            ctc=sums[0]/sums[1];kd=sums[2]/sums[3].clamp_min(1)
            anchor=sum((p-anchors[name]).square().sum() for name,p in student.named_parameters())
            loss=ctc+config['distill']*kd+config['anchor']*anchor
            assert torch.isfinite(loss),'Nonfinite objective'
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(student.parameters(),5,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();scheduler.step();batches+=1;examples+=len(rows)
            running.update({'loss':loss.item(),'ctc':ctc.item(),'kl':kd.item(),'anchor':anchor.item()})
            if batches==1 or batches%100==0:print('STEP',json.dumps({'candidate':config['name'],'epoch':epoch+1,'batch':batches,'examples':examples,'seconds':round(time.monotonic()-started),**{k:v/batches for k,v in running.items()}}),flush=True)
        metrics=evaluate(wrapped,val,CHARS);value=score(metrics)
        row={'config':config,'epoch':epoch+1,'score':value,'validation':metrics,'seconds':round(time.monotonic()-started),'batches':batches,'examples':examples,'complete_epoch':batches==len(train_loader)};history.append(row)
        print('VALIDATION',json.dumps(row),flush=True)
        if value<best_score:
            best_score=value;best_info=row
            torch.save({'state_dict':student.state_dict(),'characters':CHARS,'selection':row},OUT/'best-checkpoint.pt')
        (OUT/'training-history.json').write_text(json.dumps(history,indent=2))
        if time.monotonic()>candidate_deadline:break
    del wrapped,objective,student,optimizer,anchors;torch.cuda.empty_cache()

# Freeze checkpoint selection before opening the new test split.
assert best_info is not None
checkpoint=torch.load(OUT/'best-checkpoint.pt',map_location='cpu',weights_only=False)
student=IndraRTL.from_onnx(original);student.add_characters(CHARS[len(original_chars):],original_chars.index(' ')+1);student.load_state_dict(checkpoint['state_dict']);student.cuda().eval();wrapped=parallel(student)
test=loader('test')
baseline_test=evaluate(teacher_parallel,test,original_chars,OUT/'baseline-test-predictions.jsonl')
selected_test=evaluate(wrapped,test,CHARS,OUT/'test-predictions.jsonl')
selected_val=evaluate(wrapped,val,CHARS)
bundle=OUT/'model';bundle.mkdir(exist_ok=True)
for name in files:shutil.copyfile(base/name,bundle/name)
student.cpu().eval();student.write_onnx(original,bundle/'rec_rtl.onnx')
meta=json.loads((base/'meta.json').read_text());meta['vocab_rtl']=json.dumps({'blank':0,'chars':CHARS},ensure_ascii=False);(bundle/'meta.json').write_text(json.dumps(meta,ensure_ascii=False),encoding='utf8')
export_parity=verify_parity(bundle/'rec_rtl.onnx',student,atol=5e-4)
import onnxruntime as ort
so=ort.SessionOptions();so.intra_op_num_threads=2
session=ort.InferenceSession(str(bundle/'rec_rtl.onnx'),so,providers=['CPUExecutionProvider'])
probe=[]
for domain in sorted({r['domain'] for r in records['test']}):
    probe.extend([r for r in records['test'] if r['domain']==domain][:3])
    # Keep this deployment check small while covering actual cached images.
    probe=probe[:16]
    if len(probe)>=16:break
data=Lines(probe);x,w,_,_,_=collate([data[i] for i in range(len(data))])
with torch.inference_mode():expected,lengths=student(x,w)
actual,actual_lengths=session.run(['ids','lengths'],{'x':x.numpy(),'widths':w.numpy()})
np.testing.assert_array_equal(actual,expected.argmax(-1).numpy());np.testing.assert_array_equal(actual_lengths,lengths.numpy())
export_parity['actual_cached_samples_checked']=len(probe);export_parity['actual_sample_ids_identical']=True
report={'device':DEVICE,'manifest_rows':manifest['rows'],'counts':{k:len(v) for k,v in records.items()},'split_conflicts_removed':dict(split_conflicts),'baseline_validation':baseline_val,'baseline_test':baseline_test,'selected_validation':selected_val,'selected_test':selected_test,'selected_candidate':best_info,'training_seconds':round(time.monotonic()-started),'pretrained_parity':parity,'exported_parity':export_parity,'data_integrity':{'text_overlap':0,'image_overlap':0,'real_document_overlap':0,'known_font_family_overlap':0},'test_selection':'Test opened only after validation-selected candidate frozen. Previous external 100-page benchmark remains a development comparison.','real_label_caveat':manifest['real_labels'],'font_caveat':'Added rendering families are disjoint. Other source corpora lack per-image font IDs, so global font novelty relative to all fine-tuning or pretraining cannot be guaranteed.','model_revision':MODEL_REV}
report.update({'unique_text_groups':{k:len(v) for k,v in groups.items()},
    'real_document_groups':{k:len(v) for k,v in real_docs.items()},
    'added_font_families':{k:sorted(v) for k,v in font_sets.items()},
    'source_counts':{k:dict(collections.Counter(r['source'] for r in rows)) for k,rows in records.items()},
    'zwnj_label_samples':{k:sum('\u200c' in r['text'] for r in rows) for k,rows in records.items()},
    'candidate_loop_budget_seconds':max_seconds,
    'optimizer_examples':{config['name']:sum(r['examples'] for r in history if r['config']==config) for config in configs}})
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(manifest_path,OUT/'data-manifest.json');shutil.copyfile(source/'ATTRIBUTION.md',OUT/'DATA-ATTRIBUTION.md')
shutil.make_archive(str(OUT/'indra-persian-large-model'),'zip',bundle)
print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ['selected_candidate','baseline_validation','selected_validation']}),flush=True)
