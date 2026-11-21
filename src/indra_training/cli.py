"""Command-line tools; training libraries are imported only by training commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Prepare logical-text line/page annotations")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=20261104)
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--test-fraction", type=float, default=0.1)
    prepare.add_argument("--max-width", type=int, default=2048)
    prepare.add_argument("--allow-text-overlap", action="store_true")
    prepare.add_argument(
        "--drop-overlapping-images",
        action="store_true",
        help="Remove duplicate lines from lower-priority splits; test > validation > train",
    )
    validate = commands.add_parser(
        "validate-data", help="Check prepared data integrity and split overlap"
    )
    validate.add_argument("dataset", type=Path)
    train = commands.add_parser("train", help="Fine-tune, select using validation, and export ONNX")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--base-model", dest="base_model_dir", type=Path)
    train.add_argument("--dataset", dest="dataset_dir", type=Path)
    train.add_argument("--output", dest="output_dir", type=Path)
    train.add_argument("--device")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--train-seconds", type=float)
    evaluate = commands.add_parser(
        "evaluate", help="Explicit evaluation after validation selection"
    )
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True, type=Path)
    check = commands.add_parser(
        "check-model", help="Check ONNX/PyTorch logits, IDs and sequence lengths"
    )
    check.add_argument("model", type=Path)
    export = commands.add_parser(
        "export-checkpoint", help="Export a candidate checkpoint and verify ONNX parity"
    )
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--base-model", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    download = commands.add_parser("download-base", help="Download a pinned source-model bundle")
    download.add_argument("--repo", default="datalab-to/indra-ocr")
    download.add_argument(
        "--revision", help="Immutable commit, tag, or branch; resolved to a commit"
    )
    download.add_argument("--output", required=True, type=Path)
    kernel = commands.add_parser(
        "kaggle-build", help="Build a private Kaggle kernel; does not submit it"
    )
    kernel.add_argument("--config", required=True, type=Path)
    kernel.add_argument("--owner", required=True)
    kernel.add_argument("--slug", required=True)
    kernel.add_argument("--dataset", action="append", required=True, dest="datasets")
    kernel.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .data import prepare_dataset

            print_json(
                prepare_dataset(
                    args.input,
                    args.output,
                    args.seed,
                    args.validation_fraction,
                    args.test_fraction,
                    args.max_width,
                    args.allow_text_overlap,
                    args.drop_overlapping_images,
                )
            )
        elif args.command == "validate-data":
            from .data import load_prepared

            manifest, _ = load_prepared(args.dataset)
            print_json(manifest)
        elif args.command == "train":
            from .config import load_config
            from .trainer import run_training

            keys = (
                "base_model_dir",
                "dataset_dir",
                "output_dir",
                "device",
                "max_steps",
                "train_seconds",
            )
            config = load_config(args.config, {key: getattr(args, key) for key in keys})
            print_json(run_training(config))
        elif args.command == "evaluate":
            from .trainer import evaluate_bundle

            if args.output.exists():
                raise FileExistsError(f"Evaluation output already exists: {args.output}")
            result = evaluate_bundle(args.model, args.dataset, args.split, device=args.device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print_json(result)
        elif args.command == "check-model":
            import torch

            from .model import verify_parity

            torch.set_num_threads(2)
            path = args.model / "rec_rtl.onnx" if args.model.is_dir() else args.model
            print_json(verify_parity(path))
        elif args.command == "export-checkpoint":
            from .trainer import export_checkpoint

            print_json(export_checkpoint(args.checkpoint, args.base_model, args.output))
        elif args.command == "download-base":
            from huggingface_hub import model_info, snapshot_download

            if args.output.exists():
                raise FileExistsError(f"Model output already exists: {args.output}")
            revision = args.revision
            if not revision:
                if args.repo != "datalab-to/indra-ocr":
                    raise ValueError("Set --revision explicitly for a custom/private model")
                revision = "bd02167fdf2005a19962cc15188f8496a2c87d3b"
            resolved = model_info(args.repo, revision=revision).sha
            snapshot_download(
                args.repo,
                revision=resolved,
                local_dir=args.output,
                allow_patterns=["*.onnx", "*.npz", "meta.json", "math_meta.json", "LICENSE"],
            )
            (args.output / "source.json").write_text(
                json.dumps({"repo": args.repo, "revision": resolved}, indent=2) + "\n",
                encoding="utf-8",
            )
            print_json({"repo": args.repo, "revision": resolved, "output": str(args.output)})
        elif args.command == "kaggle-build":
            from .kaggle import build_kernel

            root = Path.cwd()
            if not (root / "src/indra_training").is_dir():
                root = Path(__file__).resolve().parents[2]
            if not (root / "pyproject.toml").is_file():
                raise ValueError("Run kaggle-build from the cloned repository checkout")
            print_json(
                build_kernel(root, args.output, args.owner, args.slug, args.datasets, args.config)
            )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
