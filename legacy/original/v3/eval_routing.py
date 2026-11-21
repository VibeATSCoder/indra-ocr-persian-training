"""Development-only ablation of ambiguous RTL versus Latin fallback routing."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'vendor/indra'),str(ROOT)]
import indra_ocr
import evaluate_pages
parser=argparse.ArgumentParser();parser.add_argument('--model-dir',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
original_load=indra_ocr.load
stats={'rtl_fallbacks_suppressed':0,'lines_classified':0}
def load(*a,**kw):
    converter=original_load(*a,**kw);original_groups=converter.ocr.line_groups
    def groups(crops,batch=64):
        names,sure=original_groups(crops,batch)
        stats['lines_classified']+=len(names)
        stats['rtl_fallbacks_suppressed']+=sum(g=='rtl' and not s for g,s in zip(names,sure))
        return names,[True if g=='rtl' else s for g,s in zip(names,sure)]
    converter.ocr.line_groups=groups
    return converter
indra_ocr.load=load
rows=evaluate_pages.download();result=evaluate_pages.evaluate(args.model_dir,rows)
result['routing_ablation']=stats;result['development_only']=True
Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(stats))
