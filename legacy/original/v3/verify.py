"""Check the delivered bundle, paired results and an actual scanned PDF."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import onnx

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'v3/results';MODEL=OUT/'model'
result=json.loads((OUT/'results.json').read_text(encoding='utf8'))
summary=json.loads((ROOT/'v3/summary.json').read_text(encoding='utf8'))
history=json.loads((OUT/'training-history.json').read_text(encoding='utf8'))
selection=json.loads((ROOT/'v3/selection-decision.json').read_text())
assert selection['test_metrics_used'] is False and selection['frozen_before_test_inspection'] is True
assert selection['winner']==min(selection['trials'],key=lambda k:(selection['trials'][k]['validation_score'],k!='fast'))
assert json.loads((ROOT/'v3/dataset-verified.json').read_text())['private'] is True
assert json.loads((ROOT/'v3/performance-batch256/execution-profile.json').read_text())['full_objective_equivalence_passed'] is True
assert result['manifest_rows']>=400000
assert result['counts']['train']>130774
assert result['counts']['test']>5186
assert all(v==0 for v in result['data_integrity'].values())
assert {r['config']['name'] for r in history if r['batches']>0}=={'adapt','conservative'}
assert result['exported_parity']['actual_sample_ids_identical']
assert result['exported_parity']['actual_cached_samples_checked']>=10
for domain,baseline in result['baseline_test'].items():
    selected=result['selected_test'][domain]
    for field in ['samples','chars','words','canonical_chars']:
        assert selected[field]==baseline[field],(domain,field)
expected={'det.onnx','rec.onnx','rec_cjk.onnx','rec_world.onnx','rec_rtl.onnx',
          'order.onnx','orient.onnx','script.onnx','meta.json','lm.npz',
          'lm_cjk.npz','lm_world.npz','LICENSE'}
assert {p.name for p in MODEL.iterdir()}==expected
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
unchanged=[]
for name in sorted(expected-{'rec_rtl.onnx','meta.json'}):
    assert digest(MODEL/name)==digest(ROOT/'base-model'/name),name
    unchanged.append(name)
before=json.loads((ROOT/'base-model/meta.json').read_text(encoding='utf8'))
after=json.loads((MODEL/'meta.json').read_text(encoding='utf8'))
old_vocab=json.loads(before.pop('vocab_rtl'));new_vocab=json.loads(after.pop('vocab_rtl'))
old_chars=old_vocab.pop('chars');new_chars=new_vocab.pop('chars')
assert before==after
assert old_vocab==new_vocab
assert new_chars==old_chars+['\u200c']
onnx.checker.check_model(str(MODEL/'rec_rtl.onnx'))
with zipfile.ZipFile(OUT/'indra-persian-large-model.zip') as archive:
    assert archive.testzip() is None
    assert set(archive.namelist())==expected
env=dict(os.environ);env['PYTHONUTF8']='1'
smoke=OUT/'pdf-smoke.txt'
subprocess.check_call([sys.executable,str(ROOT/'ocr_pdf.py'),
    str(ROOT/'tmp/pdfs/persian-scan-smoke.pdf'),'--model-dir',str(MODEL),
    '--out',str(smoke),'--mode','ocr','--cores','2'],env=env)
text=smoke.read_text(encoding='utf8')
persian=sum('\u0600'<=c<='\u06ff' for c in text)
assert persian>2000
report={'checks_passed':True,'raw_rows':result['manifest_rows'],
        'effective_counts':result['counts'],'trained_candidates':sorted({r['config']['name'] for r in history}),
        'optimizer_examples':{name:sum(r['examples'] for r in history if r['config']['name']==name) for name in ['adapt','conservative']},
        'unchanged_bundle_files':unchanged,'modified_files':['rec_rtl.onnx','meta.json'],
        'vocabulary_added':['ZWNJ'],'exported_parity':result['exported_parity'],
        'scanned_pdf_persian_characters':persian,
        'bundle_sha256':digest(OUT/'indra-persian-large-model.zip'),
        'modern_book_gate_passed':summary['book_recommendation_gate_passed']}
(ROOT/'v3/final-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
