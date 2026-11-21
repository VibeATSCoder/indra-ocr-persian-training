# Persian Indra OCR training

Source code, recovered parameters, and reusable commands for our Persian Indra OCR fine-tuning work. The reusable trainer updates the **RTL line recognizer** and exports it into Indra's original ONNX graph. The detector, layout analysis, and other recognizers remain in the supplied model bundle.

The repository includes both the original experiment code and a configurable path for a new dataset. Document OCR and receipt/ID OCR have separate training histories and starting checkpoints.

## What is included

| Directory | Contents |
| --- | --- |
| `src/indra_training/` | Configurable preparation, training, validation, ONNX export, and Kaggle packaging. |
| `configs/` | Starting configurations for document and receipt adaptation; a two-step CPU smoke configuration. |
| `configs/historical/` | Parameters and selected aggregate results recovered from the original files, including complete candidate grids. |
| `legacy/original/` | Original preparation and training scripts, preserving their layout and bytes. |
| `legacy/kaggle_snapshots/` | 17 original Kaggle kernel snapshots and their metadata. |
| `vendor/indra/` | Upstream Indra source pinned to the commit used in our experiments. |
| `tools/synthetic_receipts/` | Recovered synthetic receipt generator, vector-template builder, and three OFL fonts. |
| `docs/` | Training history, dataset contract, validation evidence, and source checksums. |

Datasets, real customer images/labels, model weights, tokens, and generated training outputs are excluded from Git. Restore model bundles and datasets separately.

## Train on a new dataset

Use Linux or WSL. Python 3.11+ is recommended. On Kaggle, keep the runtime's CUDA-enabled PyTorch installation. On another GPU machine, install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) first.

```bash
git clone https://github.com/VibeATSCoder/indra-ocr-persian-training.git
cd indra-ocr-persian-training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,datasets,dev]'

# Original Indra model at the immutable revision used in our experiments.
indra-train download-base --output models/base

# Supply labeled line images, or page images with a bbox for each labeled line.
indra-train prepare --input /path/to/new-dataset/labels.jsonl --output data/prepared
indra-train validate-data data/prepared

# Edit the config before running. This preset uses one GPU for 4,500 seconds.
indra-train train --config configs/documents-v5-fast-loose.json

# Run test evaluation only after choosing the configuration/checkpoint.
indra-train evaluate --model runs/documents-new-data/model \
  --dataset data/prepared --split test \
  --output runs/documents-new-data/test-metrics.json
```

The input JSONL uses **logical Persian text**, not reversed text:

```json
{"image":"images/line-001.png","text":"متن همان خط را اینجا بنویسید","document_id":"book-001","split":"train","domain":"custom"}
```

Keep all lines from one book/document in one split. With no `split`, the preparer assigns the whole `document_id` deterministically. The [dataset guide](docs/datasets.md) explains page boxes, repeated template labels, validation/test grouping, and conversion from our synthetic receipt generator.

## Parameters we actually used

These are recovered historical settings, not accuracy claims for a new dataset.

| Setting | Document v5 winner | Receipt v5.1 winner |
| --- | --- | --- |
| Candidate | `fast-loose` | `bold`, receipt run 6 |
| Learning rate | `8e-5` | `1e-4` |
| Batch size | `192` | `192` |
| L2 anchor weight | `2e-4` | `2e-5` |
| Teacher KL weight | `0.05` | `0.01` |
| KL temperature | `2` | `2` |
| Training budget per candidate | `4,500 seconds` | `3,600 seconds` |
| Validation interval | `1,100 seconds` | `750 seconds` |
| Seed | `20261104` | `20261116` |
| Initialization | Original pinned Indra RTL model | Receipt run 3's `model-steady` |

Both used AdamW with zero weight decay, gradient clipping at 5, frozen BatchNorm statistics, 3% warmup, and cosine decay to 2% of the peak LR. Original candidate runs used one process per T4; the two T4s evaluated different candidates rather than splitting one batch.

The reusable presets copy the winner's optimizer settings. **The exact old data mixing, receipt-specific digit policy, augmentation, and selection details are preserved in the original scripts.** The new trainer is a configurable adaptation for new data; it does not claim bitwise reproduction of an old Kaggle run. See [training history and reproduction](docs/training-history.md).

## Start from an existing fine-tune

To adapt our private document model rather than original Indra, use your Hugging Face access token through the environment or local login, then download the desired commit/tag:

```bash
indra-train download-base --repo mehdigozin/indra-ocr-persian \
  --revision YOUR_COMMIT_OR_TAG --output models/persian-start

indra-train train --config configs/documents-v5-fast-loose.json \
  --base-model models/persian-start --output runs/new-adaptation
```

The downloader resolves the requested revision to an immutable commit and records it in `source.json`. For receipt adaptation, supply the correct receipt starting bundle and use `configs/receipts-v51-bold.json`. An existing run's exported `model/` directory can also be used as `--base-model` for a new run with a fresh optimizer.

## Run a small local check

```bash
python scripts/make_demo_dataset.py --output data/demo
indra-train prepare --input data/demo/labels.jsonl --output data/demo-prepared
indra-train train --config configs/smoke.json --base-model models/base
indra-train evaluate --model runs/smoke/model --dataset data/demo-prepared \
  --output runs/smoke/test-metrics.json
python -m unittest discover -s tests -v
# Deeper check: actual weight updates, expanded-vocabulary export, and kernel packaging.
python scripts/smoke_test.py --base-model models/base --report runs/smoke-report.json
```

The demo contains ten invented Persian lines and only tests that the pipeline works. Its scores do not measure production quality. A fresh dataset/run directory is required to avoid overwriting previous experiments.

## Run on Kaggle

Prepare the dataset locally or in a CPU notebook. Attach its prepared files and exactly one complete base-model bundle as **private Kaggle datasets**. Then build an upload-ready kernel:

```bash
indra-train kaggle-build --config configs/documents-v5-fast-loose.json \
  --owner YOUR_KAGGLE_USER --slug indra-new-dataset-train \
  --dataset YOUR_KAGGLE_USER/prepared-lines \
  --dataset YOUR_KAGGLE_USER/indra-base-model \
  --output runs/kaggle-new-dataset

# Authenticate the official Kaggle CLI separately; no token belongs in run.py.
kaggle kernels push -p runs/kaggle-new-dataset --accelerator NvidiaTeslaT4
kaggle kernels status YOUR_KAGGLE_USER/indra-new-dataset-train
kaggle kernels output YOUR_KAGGLE_USER/indra-new-dataset-train -p runs/kaggle-download
```

`kaggle-build` creates `run.py` and private kernel metadata; it does not upload or start training. The kernel discovers the prepared dataset and model in `/kaggle/input`, trains on `cuda:0`, and writes results under `/kaggle/working/run`. Candidates in the reusable trainer run sequentially on that device. Use the archived builders for the original two-GPU candidate scheduling.

## What a run saves

- `resolved-config.json`, dataset fingerprint, package versions, and Git revision.
- Baseline validation, per-candidate history, and best PyTorch checkpoints.
- `selection.json`, chosen using validation scores with the starting model eligible.
- `model/`, containing the selected ONNX recognizer and the supplied bundle's other model files.
- Numerical parity results for logits, output IDs, and sequence lengths.

Test metrics are produced by a separate `evaluate` command. A checkpoint contains optimizer state for inspection; the current CLI supports warm-starting from ONNX rather than exact mid-epoch optimizer/RNG resumption.

To export a particular candidate instead of the validation-selected bundle:

```bash
indra-train export-checkpoint --checkpoint runs/documents-new-data/best-fast-loose.pt \
  --base-model models/base --output runs/documents-new-data/candidate-model
```

## Base source and licenses

- Indra code: [`datalab-to/indra`, commit `e20355319b5aa5705a876a8d2b40aca606d7df5f`](https://github.com/datalab-to/indra/tree/e20355319b5aa5705a876a8d2b40aca606d7df5f).
- Original model: [`datalab-to/indra-ocr`, revision `bd02167fdf2005a19962cc15188f8496a2c87d3b`](https://huggingface.co/datalab-to/indra-ocr/tree/bd02167fdf2005a19962cc15188f8496a2c87d3b).
- Our PyTorch reconstruction: `src/indra_training/model.py`; original copy: `legacy/original/indra_rtl_trainable.py`.
- Recovered source checksums: [`docs/source-manifest.json`](docs/source-manifest.json).

Apache-2.0 notices for upstream code and SIL OFL notices for the included fonts are retained. Model weights and source datasets have separate terms; they are not distributed in this repository. See [third-party notices](docs/third-party-notices.md).
