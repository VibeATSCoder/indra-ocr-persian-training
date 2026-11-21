"""Prepare a large mixed-domain corpus and cache inference-compatible ink maps."""
import ast
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tarfile
import time
import urllib.parse
import zipfile
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image,ImageDraw,ImageFont,features
from rapidfuzz.distance import Levenshtein
from common import clean,key,crop_image,fits
from indra_ocr.pipeline import OrtOCR
from indra_ocr.rec_preproc import pad_batch,width_batches
from indra_ocr.rtl import flip,visual_label

cv2.setNumThreads(1)
OUT=Path('/kaggle/working');TMP=Path('/tmp/large-ocr');TMP.mkdir(exist_ok=True)
SEED=20261102;rng=random.Random(SEED)
MODEL_REV='bd02167fdf2005a19962cc15188f8496a2c87d3b'
GREV='36bfdcdeac20c02231f4ee08472f80db2fc467bb'
PREV='05d248b30988b41bb905f7919c0bac39cab38d70'
AREV='a297702166812d6fc7a6825321d16226e2521101'
RREV='4dc1024494d431a9f2ba098b2f21b94e7177347d'
START=time.monotonic()
assert features.check_feature('raqm'),'Arabic shaping support required before preparation'

def download(url,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size:return path
    for attempt in range(4):
        try:
            with requests.get(url,stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with path.open('wb') as handle:
                    for data in response.iter_content(4*1024*1024):handle.write(data)
            return path
        except Exception:
            if attempt==3:raise RuntimeError('Public source download failed: '+path.name) from None
            time.sleep(attempt+1)

def hf(repo,rev,name):
    url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}'
    return download(url,TMP/(repo.replace('/','_')+'_'+Path(name).name))

base=TMP/'base';base.mkdir(exist_ok=True)
for name in ['det.onnx','rec.onnx','rec_rtl.onnx','meta.json']:
    download(f'https://huggingface.co/datalab-to/indra-ocr/resolve/{MODEL_REV}/{name}',base/name)
chars=json.loads(json.loads((base/'meta.json').read_text())['vocab_rtl'])['chars']
if '\u200c' not in chars:chars.append('\u200c')
allowed=set(chars)
reserved={}
old=list(Path('/kaggle/input').rglob('train.parquet'))
assert len(old)==1
for split in ['validation','test']:
    for batch in pq.ParquetFile(old[0].parent/f'{split}.parquet').iter_batches(columns=['text'],batch_size=2048):
        for text in batch.column(0).to_pylist():reserved[key(text)]=split
csv_paths={s:hf('ordaktaktak/Persian-OCR-230k',PREV,f'data/{s}.csv') for s in ['train','test']}
labels={}
for split,path in csv_paths.items():
    with path.open(encoding='utf-8-sig',newline='') as handle:labels[split]=list(csv.DictReader(handle))
for r in labels['test']:reserved[key(r['text'])]='test'

def split_text(text):
    k=key(text)
    if k in reserved:return reserved[k]
    bucket=int(k[:8],16)%100
    return 'test' if bucket>=98 else 'validation' if bucket>=96 else 'train'

schema=pa.schema([('ink',pa.binary()),('width',pa.int32()),('text',pa.string()),('visual',pa.string()),('source',pa.string()),('domain',pa.string()),('doc_id',pa.string()),('group',pa.string()),('image_sha256',pa.string())])
buffers=collections.defaultdict(list);counts=collections.Counter();rejected=collections.Counter();seen={};shards=[];shard_index=collections.Counter()

def flush(split):
    if not buffers[split]:return
    name=f'{split}-{shard_index[split]:04d}.parquet';shard_index[split]+=1
    table=pa.Table.from_pylist(buffers[split],schema=schema)
    pq.write_table(table,OUT/name,compression='zstd',compression_level=3,row_group_size=256)
    shards.append({'file':name,'split':split,'rows':len(buffers[split]),'bytes':(OUT/name).stat().st_size})
    buffers[split]=[]

def add(content,text,source,domain,doc_id,split=None,ink=None):
    text=clean(text)
    if not 3<=len(text)<=240 or not any('\u0600'<=c<='\u06ff' for c in text):rejected['label']+=1;return
    if set(visual_label(text))-allowed:rejected['unsupported_label']+=1;return
    split=split or split_text(text)
    if split=='train' and split_text(text)!='train':rejected['reserved_text']+=1;return
    if split=='validation' and reserved.get(key(text))=='test':rejected['reserved_test']+=1;return
    try:
        ink=ink if ink is not None else crop_image(content)
        if not fits(ink,text):rejected['empty_or_ctc_length']+=1;return
    except Exception:rejected['image']+=1;return
    digest=hashlib.sha256(ink.tobytes()).hexdigest()
    if digest in seen:rejected['duplicate_image']+=1;return
    seen[digest]=split
    buffers[split].append({'ink':ink.tobytes(),'width':ink.shape[1],'text':text,'visual':visual_label(text),'source':source,'domain':domain,'doc_id':doc_id,'group':key(text),'image_sha256':digest})
    counts[f'{split}:{domain}']+=1
    if len(buffers[split])>=20000:flush(split)

# Real document crops use an independent publisher's model transcripts. They
# remain explicitly pseudo-labelled; no human-ground-truth claim is made.
print('PREP real documents',flush=True)
real_path=hf('Reza2kn/persian-ocr-community-dataset-argilla',RREV,'data/train.parquet')
real_pages=pq.read_table(real_path).to_pylist()
protection=json.loads((Path(__file__).parent/'exclusion.json').read_text())
protected_hashes=set(protection['sha256']);protected_visual=[int(x,16) for x in protection['dhash512']]
def dhash(gray):
    a=cv2.resize(gray,(33,16));bits=(a[:,1:]>a[:,:-1]).ravel();v=0
    for bit in bits:v=(v<<1)|int(bit)
    return v
detector=OrtOCR(base,threads=2)
so=ort.SessionOptions();so.intra_op_num_threads=2
reader=ort.InferenceSession(str(base/'rec_rtl.onnx'),so,providers=['CPUExecutionProvider'])
base_chars=chars[:-1] if chars[-1]=='\u200c' else chars
def read_boxes(gray,boxes):
    inks=[crop_image(gray,b) for b in boxes];texts=['']*len(inks)
    for ids in width_batches([a.shape[1] for a in inks],max_batch=32):
        x,w=pad_batch([inks[i].astype(np.float32)/255 for i in ids])
        seq,conf,lengths=reader.run(['ids','conf','lengths'],{'x':x,'widths':w})
        for row,j in enumerate(ids):
            previous=-1;out=[]
            for token in seq[row,:int(lengths[row])]:
                if token and token!=previous:out.append(base_chars[int(token)-1])
                previous=token
            texts[j]=clean(flip(''.join(out)))
    return texts,inks

page_candidates=[];per_doc=collections.Counter()
for row in sorted(real_pages,key=lambda r:hashlib.sha256(r['image'].encode()).hexdigest()):
    blocks=[b for b in json.loads(row['label']) if b.get('label') in ('Text','Section-Header','Caption','Footnote') and b.get('confidence',0)>=.65 and not b.get('truncated') and len(clean(b.get('text','')))>=8]
    doc=row['image'].split('assets/')[-1].split('-pages-')[0]
    if blocks and per_doc[doc]<4:
        page_candidates.append((row,blocks,doc));per_doc[doc]+=1
    if len(page_candidates)>=2000:break
real_rows=[];real_docs=collections.Counter()
def fetch_page(item):
    row,blocks,doc=item
    # Pin asset requests to the same immutable dataset revision as annotations.
    asset=row['image'].split('/resolve/',1)[-1].split('/',1)[-1]
    url=f'https://huggingface.co/datasets/Reza2kn/persian-ocr-community-dataset-argilla/resolve/{RREV}/{asset}'
    response=requests.get(url,timeout=60);response.raise_for_status()
    return response.content,blocks,doc
with concurrent.futures.ThreadPoolExecutor(6) as pool:
    futures=[pool.submit(fetch_page,item) for item in page_candidates]
    for page_index,future in enumerate(futures):
        try:
            content,blocks,doc=future.result();gray=np.asarray(Image.open(io.BytesIO(content)).convert('L'))
            if hashlib.sha256(content).hexdigest() in protected_hashes or any((dhash(gray)^h).bit_count()<=4 for h in protected_visual):rejected['protected_external_page']+=1;continue
            boxes=detector.detect(gray)
            if not len(boxes):continue
            predictions,inks=read_boxes(gray,boxes)
            bucket=int(hashlib.sha256(doc.encode()).hexdigest()[:8],16)%100
            split='test' if bucket>=90 else 'validation' if bucket>=80 else 'train'
            for block in blocks:
                text=clean(block['text']);x0,y0,x1,y1=[block[k] for k in ['x0','y0','x1','y1']]
                selected=[i for i,b in enumerate(boxes) if max(0,min(x1,b[2])-max(x0,b[0]))*max(0,min(y1,b[3])-max(y0,b[1]))/max(1,(b[2]-b[0])*(b[3]-b[1]))>.8]
                selected.sort(key=lambda i:(boxes[i,1],-boxes[i,0]))
                if not selected:continue
                joined=' '.join(predictions[i] for i in selected)
                if not joined or Levenshtein.normalized_similarity(joined,text)<.65:continue
                mapping={}
                for op in Levenshtein.opcodes(joined,text):
                    if op.tag in ('equal','replace'):
                        for p in range(op.src_start,op.src_end):mapping[p]=op.dest_start+min(max(0,op.dest_end-op.dest_start-1),int((p-op.src_start)*max(1,op.dest_end-op.dest_start)/max(1,op.src_end-op.src_start)))
                pos=0
                for i in selected:
                    prediction=predictions[i];end=pos+len(prediction)
                    mapped=[mapping[p] for p in range(pos,end) if p in mapping]
                    pos=end+1
                    if not mapped:continue
                    left,right=min(mapped),max(mapped)+1
                    while left>0 and text[left-1]!=' ' and min(mapped)-left<8:left-=1
                    while right<len(text) and text[right]!=' ' and right-max(mapped)-1<8:right+=1
                    reference=text[left:right].strip()
                    if len(reference)<8 or Levenshtein.normalized_similarity(prediction,reference)<.80:continue
                    if not fits(inks[i],reference):continue
                    real_rows.append((inks[i],reference,doc,split));real_docs[doc]+=1
        except Exception:rejected['real_page_download_or_parse']+=1
        if (page_index+1)%100==0:print('REAL pages',page_index+1,'accepted lines',len(real_rows),'seconds',round(time.monotonic()-START),flush=True)
for ink,text,doc,split in real_rows:
    if split=='test':reserved[key(text)]='test'
    elif split=='validation' and reserved.get(key(text))!='test':reserved[key(text)]='validation'
for ink,text,doc,split in real_rows:add(None,text,'community_argilla','real',doc,split,ink)
del real_rows

print('PREP Garshasp',flush=True)
garshasp_jobs=[('train-00000-of-00003',200000),('plain_text-00000-of-00001',50000),('plain_text_blur-00000-of-00001',50000),('plain_text_shadow-00000-of-00001',25000),('plain_text_darkness-00000-of-00001',25000),('white_noise-00000-of-00001',25000),('noisy_Bg-00000-of-00001',25000)]
for stem,cap in garshasp_jobs:
    path=hf('AliShafiee2003/persian-ocr-garshasp-70c',GREV,'data/'+stem+'.parquet')
    total=pq.ParquetFile(path).metadata.num_rows;chosen=set(rng.sample(range(total),min(cap,total)));index=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        raw=[r for j,r in enumerate(batch.to_pylist(),start=index) if j in chosen];index+=batch.num_rows
        def convert(row):
            content=row['image']['bytes'] if isinstance(row['image'],dict) else row['image']
            try:return row['text'],crop_image(content)
            except Exception:return row['text'],None
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            for text,ink in pool.map(convert,raw):
                if ink is not None:add(None,text,'garshasp/'+stem,'garshasp',stem,ink=ink)
    print('GARS',stem,'total accepted',sum(counts.values()),flush=True)
    path.unlink()

print('PREP complete 230k',flush=True)
archive=zipfile.ZipFile(hf('ordaktaktak/Persian-OCR-230k',PREV,'data/Images.zip'))
members={Path(n).name:n for n in archive.namelist() if not n.endswith('/')}
for split in ['train','test']:
    for index,row in enumerate(labels[split]):
        name=Path(row['fname']).name
        if name in members:add(archive.read(members[name]),row['text'],'persian230k/'+split,'persian230k',name,'test' if split=='test' else None)
        if (index+1)%30000==0:print('230K',split,index+1,flush=True)
archive.close()

print('PREP all Arshasb 7K pages',flush=True)
book_archive=download('https://drive.usercontent.google.com/download?id=1G9JEZY9MSzaND8ynnFodIXQvMMM1_6J3&export=download&confirm=t',TMP/'books.tar.gz')
with tarfile.open(book_archive) as tar:tar.extractall(TMP/'books',filter='data')
source=(Path(__file__).parent/'old_book_parser.py').read_text();module=ast.parse(source)
namespace={'collections':collections,'ast':ast,'unicodedata':__import__('unicodedata'),'hashlib':hashlib,'zipfile':zipfile,'ET':ET,'Image':Image,'io':io}
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'book-parser','exec'),namespace)
for index,path in enumerate(sorted((TMP/'books').rglob('label_*.xlsx'))):
    for row in namespace['page_lines'](path):add(row['image'],row['text'],'arshasb/Far_ketab','arshasb',row['source_id'].split(':')[0],row['split'])
    if (index+1)%500==0:print('BOOK pages',index+1,flush=True)

print('PREP disjoint font families',flush=True)
assert features.check_feature('raqm'),'Arabic shaping support required'
families={'vazirmatn':'train','notosansarabic':'train','markazitext':'train','lateef':'train','mirza':'train','notonaskharabic':'validation','amiri':'test','scheherazadenew':'test'}
font_files={};licenses={}
for family in families:
    result=requests.get(f'https://api.github.com/repos/google/fonts/contents/ofl/{family}',timeout=30);result.raise_for_status();files=result.json()
    font=next(x for x in files if x['name'].endswith('.ttf') and ('Regular' in x['name'] or '[' in x['name']))
    font_files[family]=download(font['download_url'],TMP/'fonts'/font['name'])
    license_file=next(x for x in files if x['name']=='OFL.txt')
    licenses[family]=requests.get(license_file['download_url'],timeout=30).text
texts={s:[] for s in ['train','validation','test']}
for row in labels['train']+labels['test']:
    text=clean(row['text']);s=split_text(text)
    if 25<=len(text)<=130 and not set(text)-allowed:texts[s].append(text)
for s in texts:texts[s]=list(dict.fromkeys(texts[s]));rng.shuffle(texts[s])
for family,split in families.items():
    size=40;font=ImageFont.truetype(str(font_files[family]),size)
    cap=8000 if split=='train' else 1500
    for text in texts[split][:cap]:
        probe=Image.new('L',(4,4),255);draw=ImageDraw.Draw(probe);box=draw.textbbox((0,0),text,font=font,direction='rtl',language='fa')
        image=Image.new('L',(box[2]-box[0]+24,box[3]-box[1]+24),255)
        ImageDraw.Draw(image).text((12-box[0],12-box[1]),text,font=font,fill=0,direction='rtl',language='fa')
        add(np.asarray(image),text,'font/'+family,'font',family,split)
    print('FONT',family,split,flush=True)

for split in list(buffers):flush(split)
manifest={'seed':SEED,'counts':dict(counts),'rejected':dict(rejected),'rows':sum(counts.values()),'shards':shards,'characters':chars,'source_revisions':{'garshasp':GREV,'persian230k':PREV,'arshasb':AREV,'community_argilla':RREV},'real_documents':len(real_docs),'real_pages_considered':len(page_candidates),'real_labels':'Independent machine-generated block transcripts aligned to fixed original-Indra line boxes, confidence >=0.65 and line agreement >=0.80. Not human ground truth. Entire source document groups held out.','font_families':families,'external_benchmark_protected':True,'split_notes':'Original evaluation labels remain reserved; official 230k test stays test. Real documents split by document hash. Added font families are disjoint between train, validation and test. Duplicate cached images omitted globally.'}
assert manifest['rows']>=400000,f'Unexpectedly small corpus: {manifest["rows"]}'
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/'font-licenses.json').write_text(json.dumps(licenses,ensure_ascii=False))
(OUT/'ATTRIBUTION.md').write_text('Garshasp: Ali Shafiee, CC BY 4.0; Persian-OCR-230k: ordaktaktak, MIT; Arshasb: Copyright 2021 persiandataset, MIT. Google Fonts: SIL OFL, full texts in font-licenses.json. Community Argilla pages/transcripts: Reza2kn, upstream terms not declared; private research inputs, no new rights or human-verification claim. Source revisions and transformations in manifest.json.\n')
print('FINAL DATA',json.dumps({k:v for k,v in manifest.items() if k not in ('shards','characters')}),flush=True)
