"""Add annotated full-page Persian print to the short-line corpus."""
import ast
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

OUT = Path('/kaggle/working')
TMP = Path('/tmp/arshasb')
TMP.mkdir(exist_ok=True)
REV = 'a297702166812d6fc7a6825321d16226e2521101'
URL = 'https://drive.usercontent.google.com/download?id=1G9JEZY9MSzaND8ynnFodIXQvMMM1_6J3&export=download&confirm=t'
archive = TMP/'arshasb.tar.gz'
print('Downloading MIT-licensed Arshasb 7K full pages',flush=True)
urllib.request.urlretrieve(URL,archive)
print('Downloaded bytes',archive.stat().st_size,flush=True)
with tarfile.open(archive) as tar:
    tar.extractall(TMP,filter='data')

def xlsx_rows(path):
    ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        shared = [''.join(node.itertext()) for node in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        matrix = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
            values = []
            for cell in row.findall('m:c',ns):
                v = cell.find('m:v',ns)
                value = v.text if v is not None else ''
                if cell.attrib.get('t') == 's':
                    value = shared[int(value)]
                values.append(value)
            matrix.append(values)
    return [dict(zip(matrix[0],row)) for row in matrix[1:]]

def group(text):
    text = unicodedata.normalize('NFKC',text).translate(str.maketrans({'ي':'ی','ك':'ک','\u200c':'','ـ':''}))
    return hashlib.sha256(' '.join(text.split()).encode()).hexdigest()

def page_lines(label_path):
    page_id = label_path.stem.removeprefix('label_')
    image_path = label_path.with_name(f'page_{page_id}.png')
    image = Image.open(image_path).convert('RGB')
    words = collections.defaultdict(list)
    for row in xlsx_rows(label_path):
        if row.get('line'):
            words[str(int(float(row['line'])))].append(str(row.get('word','')))
    page_text = ' '.join(w for line in words.values() for w in line)
    bucket = int(group(page_text)[:8],16)%100
    split = 'test' if bucket >= 98 else 'validation' if bucket >= 96 else 'train'
    rows = []
    for line in xlsx_rows(label_path.with_name(f'line_{page_id}.xlsx')):
        line_id = str(int(float(line['line'])))
        text = unicodedata.normalize('NFC',' '.join(words[line_id])).strip()
        if not 3 <= len(text) <= 240:
            continue
        points = [ast.literal_eval(line[f'point{i}']) for i in range(1,5)]
        x0,y0 = min(p[0] for p in points),min(p[1] for p in points)
        x1,y1 = max(p[0] for p in points),max(p[1] for p in points)
        if x1<=x0 or y1<=y0:
            continue
        crop = image.crop((x0,y0,x1,y1))
        stream = io.BytesIO()
        crop.save(stream,format='PNG')
        content = stream.getvalue()
        rows.append({'image':content,'text':text,'source':'arshasb/Far_ketab','source_id':f'{page_id}:{line_id}','text_group':group(text),'image_sha256':hashlib.sha256(content).hexdigest(),'split':split})
    return rows

labels = sorted(TMP.rglob('label_*.xlsx'))
random.Random(20261101).shuffle(labels)
# Span 2,000 full source pages while limiting the single-font corpus weight.
labels = labels[:2000]
book_rows = []
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    for index,rows in enumerate(pool.map(page_lines,labels)):
        book_rows.extend(rows)
        if (index+1)%250 == 0:
            print('Processed pages',index+1,'lines',len(book_rows),flush=True)
corpus_paths = list(Path('/kaggle/input').rglob('train.parquet'))
assert len(corpus_paths)==1
corpus = corpus_paths[0].parent
rows = []
for split in ('train','validation','test'):
    rows.extend(pq.read_table(corpus/f'{split}.parquet').to_pylist())
rows.extend(book_rows)
test_groups = {r['text_group'] for r in rows if r['split']=='test'}
validation_groups = {r['text_group'] for r in rows if r['split']=='validation'}
seen = set()
filtered = []
removed = collections.Counter()
for row in rows:
    if row['split']!='test' and row['text_group'] in test_groups:
        removed['test_text_overlap']+=1
        continue
    if row['split']=='train' and row['text_group'] in validation_groups:
        removed['validation_text_overlap']+=1
        continue
    if row['image_sha256'] in seen:
        removed['duplicate_image']+=1
        continue
    seen.add(row['image_sha256'])
    filtered.append(row)
counts = collections.Counter(f"{r['split']}:{r['source']}" for r in filtered)
for split in ('train','validation','test'):
    data = [r for r in filtered if r['split']==split]
    pq.write_table(pa.Table.from_pylist(data),OUT/f'{split}.parquet',compression='zstd',row_group_size=256)
    print(split,len(data),flush=True)
report = json.loads((corpus/'data-report.json').read_text())
report.update({'book_source':{'url':'https://github.com/persiandataset/Arshasb','revision':REV,'license':'MIT','sampled_pages':len(labels),'lines':len(book_rows),'font':'Far_ketab','notes':'Rendered full-page Persian print, not scanned real books; precise word/line geometry. Page-level train/validation/test split, cross-source duplicate text excluded from training.'},'counts':dict(counts),'rows':len(filtered),'additional_removed':dict(removed)})
(OUT/'data-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
license_text = urllib.request.urlopen(f'https://raw.githubusercontent.com/persiandataset/Arshasb/{REV}/LICENSE',timeout=30).read().decode()
(OUT/'ARSHASB-LICENSE.txt').write_text(license_text)
(OUT/'ATTRIBUTION.md').write_text((corpus/'ATTRIBUTION.md').read_text()+f'\nArshasb: Copyright (c) 2021 persiandataset. MIT license. https://github.com/persiandataset/Arshasb at {REV}. Changes: line crops derived from supplied geometry, text labels joined by line, deterministic 2,000-page subset and page-level splits.\n')
print(json.dumps(report),flush=True)
