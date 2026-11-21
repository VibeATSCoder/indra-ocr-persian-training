"""Summarize the larger experiment, preserving negative results and caveats."""
import collections
import json
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'v3/results'
result=json.loads((OUT/'results.json').read_text(encoding='utf8'))
selection=json.loads((ROOT/'v3/selection-decision.json').read_text())
assert selection['test_metrics_used'] is False and selection['frozen_before_test_inspection'] is True
before=json.loads((ROOT/'outputs/page-baseline.json').read_text(encoding='utf8'))
after=json.loads((OUT/'external-pages.json').read_text(encoding='utf8'))
ids=set(json.loads((ROOT/'benchmark-selection.json').read_text())['modern_printed_ids'])

def page_totals(data):
    rows=[r for r in data['per_page'] if r['id'] in ids]
    return {'pages':len(rows),'cer':sum(r['char_edits'] for r in rows)/sum(r['chars'] for r in rows),'wer':sum(r['word_edits'] for r in rows)/sum(r['words'] for r in rows),'no_space_cer':sum(r['no_space_edits'] for r in rows)/sum(r['no_space_chars'] for r in rows),'word_recall':sum(r['matched_words'] for r in rows)/sum(r['words'] for r in rows)}

def paired_ci(groups):
    rows=np.asarray(list(groups.values()),np.float64)
    if len(rows)<2:return None
    sample=np.random.default_rng(20261102).integers(0,len(rows),(4000,len(rows)))
    draws=rows[sample].sum(1)
    delta=(draws[:,1]-draws[:,0])/draws[:,2]
    return {'groups':len(rows),'cer_delta_percentage_points':100*(rows[:,1].sum()-rows[:,0].sum())/rows[:,2].sum(),'bootstrap_95_percent_interval_pp':(100*np.quantile(delta,[.025,.975])).tolist()}

old_pages={r['id']:r for r in before['per_page']}
new_pages={r['id']:r for r in after['per_page']}
page_ci=paired_ci({i:[old_pages[i]['char_edits'],new_pages[i]['char_edits'],old_pages[i]['chars']] for i in ids})
old_predictions=[json.loads(s) for s in (OUT/'baseline-test-predictions.jsonl').read_text(encoding='utf8').splitlines()]
new_predictions=[json.loads(s) for s in (OUT/'test-predictions.jsonl').read_text(encoding='utf8').splitlines()]
assert len(old_predictions)==len(new_predictions)
documents=collections.defaultdict(lambda:[0,0,0]);fonts=collections.defaultdict(lambda:[0,0,0])
for a,b in zip(old_predictions,new_predictions):
    assert (a['image_sha256'],a['reference'])==(b['image_sha256'],b['reference'])
    values=[Levenshtein.distance(a['reference'],a['prediction']),Levenshtein.distance(b['reference'],b['prediction']),len(a['reference'])]
    target=documents if a['source']=='community_argilla' else fonts if a['source'].startswith('font/') else None
    if target is not None:
        for j,n in enumerate(values):target[a['document']][j]+=n

old=page_totals(before);new=page_totals(after)
test=result['selected_test'];baseline=result['baseline_test']
positive=(result['selected_candidate']['config']['name']!='original_control' and new['cer']<old['cer'] and new['wer']<=old['wer'] and test.get('real',{}).get('cer',1)<=baseline.get('real',{}).get('cer',0) and test['font']['cer']<baseline['font']['cer'])
summary={'selected_candidate':result['selected_candidate']['config'],'selected_epoch':result['selected_candidate']['epoch'],'raw_corpus_rows':result['manifest_rows'],'effective_split_counts':result['counts'],'split_conflicts_removed':result['split_conflicts_removed'],'new_test':{'original':baseline,'selected':test},'modern_external_pages':{'original':old,'selected':new,'paired_page_bootstrap':page_ci},'pseudo_label_document_bootstrap':paired_ci(documents),'held_out_rendering_fonts':{k:{'original_cer':v[0]/v[2],'selected_cer':v[1]/v[2],'reference_chars':v[2]} for k,v in fonts.items()},'book_recommendation_gate_passed':positive,'font_caveat':result['font_caveat'],'real_label_caveat':result['real_label_caveat'],'external_caveat':'The 13-page modern subset and 100-page reference benchmark are reused development comparisons. They are small and are not pristine final tests. New document/font test splits were not used for checkpoint selection.'}
summary['all_external_pages']={name:{k:data[k] for k in ['pages','cer','wer','no_space_cer','word_recall']} for name,data in [('original',before),('selected',after)]}
summary['execution_selection']=selection
summary['unique_text_groups']=result['unique_text_groups']
summary['real_document_groups']=result['real_document_groups']
summary['optimizer_examples']=result['optimizer_examples']
(ROOT/'v3/summary.json').write_text(json.dumps(summary,indent=2))
def pct(n):return f'{100*n:.2f}%'
lines=['# Larger Persian Indra experiment','',f"Prepared {result['manifest_rows']:,} cached samples. Effective splits after cross-source text filtering: "+', '.join(f'{k} {v:,}' for k,v in result['counts'].items())+'.','',f"Selected {result['selected_candidate']['config']['name']} at epoch {result['selected_candidate']['epoch']} using validation only. The unchanged original was an eligible control.",'','## New held-out cropped-line tests','','| Domain | Lines | Original CER | Selected CER | Original WER | Selected WER |','|---|---:|---:|---:|---:|---:|']
for domain in ['real','font','arshasb','garshasp','persian230k','all']:
    if domain in test:
        a,b=baseline[domain],test[domain];lines.append(f"| {domain} | {b['samples']:,} | {pct(a['cer'])} | {pct(b['cer'])} | {pct(a['wer'])} | {pct(b['wer'])} |")
lines.extend(['','Real-domain references are independent machine transcripts filtered against a fixed baseline, not human ground truth. This selection can favor baseline-readable lines. Source documents are disjoint. Added rendering font families are disjoint, but fonts in the older opaque corpora and original pretraining are not fully documented.','', '## External modern printed pages','',f"On the fixed 13-page subset: CER {pct(old['cer'])} -> {pct(new['cer'])}; WER {pct(old['wer'])} -> {pct(new['wer'])}; word recall {pct(old['word_recall'])} -> {pct(new['word_recall'])}.",f"Paired page-bootstrap CER delta interval (percentage points): {page_ci['bootstrap_95_percent_interval_pp'] if page_ci else 'not available'}. This is descriptive uncertainty on a small, reused development set.",'', 'Recommendation: '+('The candidate improves all prespecified checks; use it as a candidate for broader real-book validation.' if positive else 'The candidate does not pass every prespecified real-book check. Preserve the original as the general-book default; do not infer book accuracy from synthetic gains.'),'','## Training and verification','', '- Two learning-rate candidates, original-model KL distillation and weight anchoring, frozen BatchNorm statistics, scan-style augmentation and balanced source sampling.','- Maximum three epochs per candidate, with a bounded total training-time budget. Actual candidate/epoch counts are in training-history.json.','- Checkpoint selected on validation domains before evaluating the new test set.','- Cached image hashes and text/document/known-font split assertions checked during loading. ONNX parity checked before training and after export.','- A routing-only development ablation did not materially improve modern-page accuracy and was not adopted.','', '## Artifacts','', '- [Private larger dataset](https://www.kaggle.com/datasets/sharifi76/persian-ocr-large-mixed-v3)','- [GPU experiment and model download](https://www.kaggle.com/code/sharifi76/indra-persian-large-train)','- Portable ONNX bundle: v3/results/indra-persian-large-model.zip','- Dataset revisions, transformations, source terms and rejected rows: data-manifest.json and DATA-ATTRIBUTION.md','- Datalab model weights retain the original modified OpenRAIL-M license.',''])
insert=lines.index('## Training and verification')
lines[insert:insert]=['## Full mixed-page comparison','',f"All 100 external pages: CER {pct(before['cer'])} -> {pct(after['cer'])}; WER {pct(before['wer'])} -> {pct(after['wer'])}. This broader set also includes handwriting and scene text beyond the requested printed-book domain.",'']
lines.extend(['## Execution trials and final selection','',
    'An initial batch-128 run and an optimized batch-256 run used the same two learning-rate configurations. Both retained the original as a control. The final trial was chosen by validation score before inspecting test accuracy.',
    'Replica-local loss reduced GPU communication. Actual-crop tests matched losses and gradients; the batch-256 probe processed about 405 samples/second versus about 120 in the original execution path.',
    f"Chosen execution trial: {selection['winner']}. Validation scores: "+', '.join(f"{k} {v['validation_score']:.6f}" for k,v in selection['trials'].items())+'.',
    f"Distinct text groups in effective splits: {result['unique_text_groups']}. Real source-document groups: {result['real_document_groups']}.",
    f"Optimizer sample draws in the chosen trial (including repeated augmented samples): {result['optimizer_examples']}.",''])
chosen_url='https://www.kaggle.com/code/'+selection['trials'][selection['winner']]['kernel']
lines=[line.replace('https://www.kaggle.com/code/sharifi76/indra-persian-large-train)',chosen_url+')') for line in lines]
(ROOT/'v3/REPORT.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps({'candidate':summary['selected_candidate'],'modern_pages':summary['modern_external_pages'],'recommendation_gate':positive}))
