# Synthetic receipt generation

Recovered vector-template and Persian text-rendering code from the earlier OCR work. The 25 layout recipes were based on style measurements from the approved tuning set; original receipt pixels and customer-field values are not included.

Install `requirements.txt` in an environment with Pillow/libraqm support. Three OFL fonts and their license files are included.

```bash
# From this directory.
python build_templates.py
python make_dataset.py --count 1000 --out generated
```

The default produces five quality variants per invented transaction. `generated/labels.jsonl` includes logical Persian text, line boxes, polygons, font information, and a `document_group` shared by all variants.

```bash
python make_dataset.py --count 1000 --out varied \
  --rotate 6 --perspective 0.025 \
  --fonts fonts/Vazirmatn.ttf fonts/NotoSansArabic.ttf fonts/NotoNaskhArabic.ttf
```

Use a fresh output directory for each generation. To supply your own invented record overrides, edit `records.example.jsonl` and pass `--records records.example.jsonl`. Automatic records are marked `fully_synthetic`; supplied records are reproduced as provided and marked `user_records`.

The root repository's `scripts/synthetic_to_lines.py` converts these outputs to the training JSONL contract. See `docs/datasets.md` for conversion and split preparation. Identical fixed-label crops must be removed from lower-priority splits with `--drop-overlapping-images`; augmented variants of one transaction must remain grouped.

`audit_dataset.py` is a separate local inspection tool and is also imported for its value-fingerprint helper. Running the generator does not open or require the original real receipts. `verify.py` is the original project-specific verification tool; its full historical audit requires the original local exclusion registry and specified sample counts. The recovered source hashes are in `docs/source-manifest.json` at the repository root.

Templates and generated images are ignored by Git and can be rebuilt from code and the supplied style measurements. Source glyph helpers retain Indra's Apache-2.0 license; the included fonts retain SIL OFL notices.
