# Recovery and validation

Checked on 2026-11-21. This is a source-recovery and functional-validation record, not a new full GPU fine-tuning benchmark.

| Check | Result |
| --- | --- |
| Unit tests | 11 passed: configuration validation, RTL/digit round trip, document/text overlap, image integrity, held-out duplicate priority, masked KL gradients, and private Kaggle packaging. |
| CPU training smoke | Ten invented Persian line images: 6 train, 2 validation, 2 test. Two optimizer updates completed with a nonzero parameter change. |
| Candidate selection | Validation-only; baseline remained selected on the easy smoke data. Test evaluation ran separately afterward. |
| Selected-model ONNX export | Logits, output IDs, and sequence lengths matched the PyTorch reconstruction. |
| Trained-candidate ONNX export | Passed the same parity checks; maximum logit difference below 0.00005. |
| Expanded vocabulary | A one-character head expansion passed ONNX/PyTorch parity. |
| Synthetic receipt tools | Rebuilt 25 vector layouts; generated 3 invented transactions in 2 quality levels, producing 6 images and 132 text regions. |
| Synthetic-data adapter | Skipped 6 provenance-only regions and converted 126 line labels. Preparation retained 103 crops after removing 23 cross-split duplicates with held-out priority. |
| Kaggle packaging | Private kernel metadata generated and its self-contained Python code compiled. No GPU kernel was submitted during recovery. |
| Python package | Wheel built, installed into a fresh Windows virtual environment without dependencies, and the installed CLI help executed. Training dependencies were tested in the existing WSL CPU environment. |
| Formatting | Ruff formatting and lint checks passed for the maintained trainer, tests, and scripts. Synthetic generator copies were reformatted with AST equivalence checked against the originals. |
| Original source | 113 archived legacy files verified byte-identical; the manifest records hashes for all recovered source copies. |
| Publication contents | No model weights, raw datasets, real receipt/ID images, customer labels, or matching credential patterns in the publication file list. |

Machine-readable CPU results and package versions are in `validation.json`. `source-manifest.json` records source and destination checksums, including derived copies whose formatting or README changed. Git attributes preserve recovered source bytes; maintained files use LF line endings.

To repeat the checks after installing the project:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m unittest discover -s tests -v
python scripts/smoke_test.py --base-model models/base --report runs/validation.json
```

CUDA execution, a new full training run, and accuracy on a new real dataset still require the intended GPU runtime and that dataset. The historical parameters and kernels are included so those runs can be launched with their provenance intact.
