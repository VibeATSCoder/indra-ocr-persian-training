# New dataset contract

The reusable trainer learns line recognition. Provide a correct transcription for each line image, or for a rectangular line box within a page image. A PDF without matching line transcriptions is not supervised OCR training data.

## Input JSONL

One JSON object per labeled line:

```json
{"image":"images/page-001.png","bbox":[40,110,710,160],"text":"متن منطقی فارسی","document_id":"book-a","split":"train","domain":"custom"}
```

| Field | Meaning |
| --- | --- |
| `image` | Required image path. Relative paths are resolved from the JSONL's directory. |
| `text` | Required logical-order Unicode transcription for that line. Do not reverse or reshape it yourself. |
| `document_id` | Required group identifying the entire source document/book or synthetic transaction. All its variants must share the group. |
| `bbox` | Optional `[x0, y0, x1, y1]` in pixels. Omit it for an already cropped line; visible ink is detected. |
| `split` | Optional `train`, `validation`, or `test`. Without it, document-group hashing assigns 80/10/10 by default. |
| `domain` | Optional sampling/evaluation domain; defaults to `custom`. |

For photographs with strong perspective, rectify the page/card before annotating line boxes. An axis-aligned box is not a perspective-correction algorithm.

## Preparation and checks

`indra-train prepare` applies the pinned Indra crop/padding rule, makes a 32-pixel-high ink map, generates visual RTL labels, and checks CTC sequence capacity. Its report records rejected images and labels that cannot fit the available timesteps. Unknown vocabulary characters are handled when loading the starting model: optional vocabulary expansion uses **training labels only**. Unseen characters in validation/test remain evaluation errors.

The preparer refuses document overlap, identical cropped images across splits, and conflicting labels for an identical image. By default, normalized repeated text across splits is refused too. Tiny datasets should set splits explicitly because hashing a few documents can leave validation empty.

Repeated field labels such as «مبلغ» occur across synthetic receipt templates. If shared labels are intentional, use `--allow-text-overlap`. Fixed template lines can also produce identical crops: `--drop-overlapping-images` removes those copies from lower-priority splits, retaining test before validation before train. Without that flag, cross-split identical images are refused. Conflicting labels and document-group overlap are always refused. Hold out full font families or layout families separately when evaluating unseen fonts/layouts; a document-group split alone does not establish unseen-font performance.

Preparation writes cached ink images, one JSONL per split, and `manifest.json`. `validate-data` rechecks label and image hashes. Use a new output directory after editing labels, images, or preprocessing settings.

## Convert our synthetic receipts

```bash
cd tools/synthetic_receipts
python build_templates.py
python make_dataset.py --count 1000 --out generated
cd ../..

python scripts/synthetic_to_lines.py \
  --input tools/synthetic_receipts/generated/labels.jsonl \
  --output data/receipt-lines.jsonl

indra-train prepare --input data/receipt-lines.jsonl \
  --output data/prepared-receipts --allow-text-overlap --drop-overlapping-images
```

The converter preserves `document_group` across quality variants and skips provenance-only text. It uses each generated line's bounding box. Keep the resulting prepared data private if you provide your own real records to the generator.

## Existing corpora and evaluation boundaries

Original preparation code for Garshasp, Persian-OCR-230k, Arshasb, mixed document lines, synthetic fonts, receipts, and MIDV-UP is under `legacy/original/`. Original Kaggle source attachments are recorded in `configs/historical/kaggle-kernels.json`.

The old `receipts-eval` corpus was a sealed evaluation set and must not be added to training or candidate selection. The reusable Kaggle builder refuses that source name. Its presence under a different name cannot be detected without the private fingerprint registry, which is deliberately absent from this code repository. Keep your own test images, labels, and exclusion registry outside Git and restore the original holdout registry only in an authorized local evaluation environment.
