# Recovered training history

Recovered from `/var/services/tune-ocr` on 2026-11-21. Parameter JSON files are extracted from the actual source declarations and saved result files, rather than reconstructed from chat descriptions. Original kernels are retained because later scripts were edited after earlier runs.

## Document models

| Stage | Code | Main settings and changes |
| --- | --- | --- |
| Initial short-line run | `legacy/original/train.py`, `legacy/kaggle_snapshots/training/` | Seed 20261101, batch 64, peak LR 2e-4, up to 8 epochs and a 5,400-second budget by default. |
| Book-oriented run | `legacy/kaggle_snapshots/training_books/` | Captured environment overrides: 5 epochs, LR 5e-5, early-backbone freeze, `balanced_book` selection, neighboring book-line augmentation. |
| Large mixed v3 | `legacy/original/v3/` | Larger mixed corpus and regularized trials. The full `experiment-plan.json` is preserved in this directory; submitted variants are in `configs/historical/kaggle-kernels.json`. |
| v4 | `legacy/original/v4/train.py`, `RUN=v4` | `adapt`: LR 3e-5, anchor 1e-3, KL 0.15; `conservative`: LR 1e-5, anchor 3e-3, KL 0.30. Each candidate: 4,200 seconds. Selected `adapt`. |
| v5 | Same trainer, `RUN=v5` | Four candidates crossing LR {3e-5, 8e-5} and regularization {(1e-3, 0.15), (2e-4, 0.05)}. Each: 4,500 seconds. Selected `fast-loose`. |

The v4/v5 sampler requested two million samples per epoch with replacement, using width buckets. Sampling mass: real 0.15, font 0.12, bookfont 0.25, Arshasb 0.12, Persian230k 0.14, Garshasp 0.22. Validation weights: real 0.35, bookfont 0.30, font 0.25, Garshasp 0.04, Arshasb 0.03, Persian230k 0.03. Selection combined 70% CER and 30% canonical CER, with eligible-domain minimum counts in the original script.

The recorded v5 training counts are 3,435 real-document lines, 339,516 Garshasp lines, 144,219 Persian230k lines, 111,745 Arshasb lines, 39,951 font lines, and 125,992 book-font lines after filtering. Some real-document labels were machine-generated and filtered, so their results are not human-ground-truth evidence. Original data reports and source checksums are required when reconstructing the full corpus.

## Receipt and ID models

`legacy/original/receipts/train.py` retains the run-2 through run-6 recipe table. Earlier run-1 code is preserved in `legacy/kaggle_snapshots/receipts/kernel-train/`. These models are a separate lineage from document v5.

| Run | Starting recognizer | Main change |
| --- | --- | --- |
| 1 | Document v5 | Synthetic receipt adaptation; see the original submitted kernel. |
| 2 | Document v5 `fast-loose` | Added approved real receipt line data and a four-candidate grid. |
| 3 | Run 2 `model-bold` | More low-quality examples; `steady` and `gentle` candidates. |
| 4 | Run 3 `model-steady` | Added ID-card lines; the recorded run crashed and was not the published winner. |
| 5 | Run 3 `model-steady` | Synthetic receipt/ID and MIDV-UP line mix; no real receipt lines in this run's input. |
| 6 / v5.1 | Run 3 `model-steady` | Approved real tuning-receipt lines added; `bold` selected: LR 1e-4, anchor 2e-5, KL 0.01, 3,600 seconds. |

The receipt recipes also normalize Arabic-Indic digits to Persian digits, suppress the Arabic-Indic output classes, and use receipt-specific low-quality augmentation and validation/domain mixtures. Those behaviors are preserved in the original trainer. `configs/receipts-v51-bold.json` carries the optimizer settings as a starting point for the new general trainer; it does not enable all legacy receipt policies automatically.

Detector training is in `legacy/original/receipts/det/`. The MIDV-UP card-segmentation trainer is in `legacy/original/receipts/ids/train_card.py`. The reusable `indra-train train` command trains the recognizer; use those separate scripts when changing detector/card-localization behavior.

## Reproduce an old run

1. Select the original Kaggle snapshot and its `kernel-metadata.json`. The corresponding source and captured environment settings are in the history directory.
2. Restore the original private Kaggle datasets, upstream model revision, and referenced kernel outputs. Keep dataset versions and content hashes fixed.
3. Use the original trainer/candidate schedule and evaluate validation before opening test metrics.
4. Compare the regenerated config, source hashes, split counts, vocabulary, and ONNX parity with the saved records.

Example builder commands, from `legacy/original/`:

```bash
# Document v5, original two-T4 candidate schedule.
python v4/build.py train --run v5

# Receipt run 6. Requires its original starting kernel output and private datasets.
python receipts/build.py train --run 6
```

The original builder metadata references account `sharifi76`. Change the owner and dataset references for another account. Some original data builders use a private holdout fingerprint registry; restore it separately. Network source locations, GPU hardware, dependency resolution, and dataset contents can change. Not every historical package version or dataset revision was logged, so the archived scripts alone do not guarantee bitwise reproduction.

## New-data runs

The reusable trainer accepts the documented JSONL contract, performs grouped split checks, and records the resolved config and environment. It runs candidates sequentially on the configured device and reports validation metrics for checkpoint selection. It retains a baseline candidate, so an exported run may intentionally keep the starting model if training does not improve validation.

The new implementation preserves the recognizer architecture and the CTC/KL/L2 objective but uses a new data interface and configurable sampling. Reuse historical optimizer settings as a starting point, then choose settings using a representative validation set. Do not describe old model scores as results for a new dataset.
