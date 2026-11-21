# Third-party source notices

## Indra

`vendor/indra/`, the corresponding original-code snapshot, and the two preprocessing modules in `src/indra_training/_vendor/` derive from `datalab-to/indra` at commit `e20355319b5aa5705a876a8d2b40aca606d7df5f`.

The upstream Apache-2.0 license is retained at `licenses/Indra-Apache-2.0.txt`, in the vendored source, and in the installed preprocessing package. The PyTorch recognizer reconstruction was recovered from our existing `indra_rtl_trainable.py`; its original copy and checksum are preserved.

Indra model files use their own weight license. Model bundles are downloaded separately with their `LICENSE` retained. This repository does not grant additional rights to third-party weights or datasets.

## Fonts

The synthetic receipt tool includes Vazirmatn, Noto Sans Arabic, and Noto Naskh Arabic. Their SIL Open Font License texts are retained under `tools/synthetic_receipts/licenses/`. The unverified private font collection is not included.

## Training data

Original preparation code references Garshasp (CC BY 4.0), Persian-OCR-230k (MIT), Arshasb (MIT), and MIDV-UP (CC BY-SA 2.5), according to the recovered attribution records. Downloaded datasets require their own attribution and usage checks. No source dataset, private customer record, test-label set, or real ID-card image is included here.
