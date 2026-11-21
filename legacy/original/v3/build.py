"""Bundle pinned Indra source plus this experiment; never bundle credentials."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--variant',choices=['base','fast'],default='base');args=parser.parse_args()
suffix='-fast' if args.variant=='fast' else ''
work=ROOT/'v3'/('kernel-'+args.stage+suffix);work.mkdir(exist_ok=True)
source='''import importlib.util,os,sys,subprocess
from pathlib import Path
required={'onnx':'onnx','onnxruntime':'onnxruntime','rapidfuzz':'rapidfuzz','pyarrow':'pyarrow','cv2':'opencv-python-headless','PIL':'pillow','requests':'requests','pypdfium2':'pypdfium2'}
missing=[p for m,p in required.items() if importlib.util.find_spec(m) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
code=Path('/tmp/indra-large-code');(code/'indra_ocr').mkdir(parents=True,exist_ok=True)
'''
for path in (ROOT/'vendor/indra/indra_ocr').glob('*.py'):
    source+=f"(code/'indra_ocr'/{path.name!r}).write_text({path.read_text(encoding='utf8')!r},encoding='utf8')\n"
source+="os.environ['PYTHONPATH']=str(code)\n"
for name in ['common.py',args.stage+'.py']+(['local_objective.py'] if args.stage=='train' else []):
    p=ROOT/'v3'/name;compile(p.read_text(encoding='utf8'),name,'exec')
    source+=f"(code/{name!r}).write_text({p.read_text(encoding='utf8')!r},encoding='utf8')\n"
if args.stage=='train':
    p=ROOT/'indra_rtl_trainable.py'
    source+=f"(code/{p.name!r}).write_text({p.read_text(encoding='utf8')!r},encoding='utf8')\n"
if args.stage=='prepare':
    p=ROOT/'prepare_books/prepare_books.py'
    source+=f"(code/'old_book_parser.py').write_text({p.read_text(encoding='utf8')!r},encoding='utf8')\n"
    p=ROOT/'v3/exclusion.json'
    source+=f"(code/'exclusion.json').write_text({p.read_text(encoding='utf8')!r},encoding='utf8')\n"
source+=f"subprocess.check_call([sys.executable,'-u',str(code/{(args.stage+'.py')!r})])\n"
(work/'run.py').write_text(source,encoding='utf8')
meta={'id':'sharifi76/indra-persian-large-'+args.stage,'title':'Indra Persian Large '+args.stage.title(),'code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_internet':True,'enable_gpu':args.stage=='train','dataset_sources':['sharifi76/persian-ocr-book-and-printed-lines-139k'] if args.stage=='prepare' else [],'kernel_sources':[] if args.stage=='prepare' else ['sharifi76/indra-persian-large-prepare'],'competition_sources':[]}
if suffix:meta['id']+=suffix;meta['title']+=' Fast'
if args.stage=='train':meta['machine_shape']='NvidiaTeslaT4'
(work/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('Bundled',args.stage,len(source),'bytes',hashlib.sha256(source.encode()).hexdigest())
