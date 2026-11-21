"""Exercise new-data training and both selected/candidate ONNX export on CPU."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from make_demo_dataset import make_demo

from indra_training.config import load_config
from indra_training.data import prepare_dataset
from indra_training.kaggle import build_kernel
from indra_training.model import IndraRTL, verify_parity
from indra_training.trainer import (
    evaluate_bundle,
    export_checkpoint,
    run_training,
    vocabulary,
)


def smoke(base: Path, root: Path):
    with tempfile.TemporaryDirectory(prefix="indra-smoke-") as temporary:
        work = Path(temporary)
        count = make_demo(work / "raw", root / "tools/synthetic_receipts/fonts/Vazirmatn.ttf")
        data = prepare_dataset(work / "raw/labels.jsonl", work / "prepared")
        config = load_config(
            root / "configs/smoke.json",
            {
                "base_model_dir": base,
                "dataset_dir": work / "prepared",
                "output_dir": work / "run",
            },
        )
        result = run_training(config)
        assert result["training"][0]["steps"] == 2
        assert result["test_evaluated"] is False
        checkpoint = torch.load(work / "run/best-smoke.pt", weights_only=True, map_location="cpu")
        starting = IndraRTL.from_onnx(base / "rec_rtl.onnx")
        delta = max(
            float((value - starting.state_dict()[name]).abs().max())
            for name, value in checkpoint["state_dict"].items()
            if value.is_floating_point()
        )
        assert delta > 0, "Training did not update any floating-point parameters"
        candidate_export = export_checkpoint(work / "run/best-smoke.pt", base, work / "candidate")
        # Exercise the expanded-head export contract independently of OCR accuracy.
        original_chars = vocabulary(base)
        extra = next(char for char in ("\u200c", "\u2603", "\u262f") if char not in original_chars)
        extended = IndraRTL.from_onnx(base / "rec_rtl.onnx")
        extended.add_characters([extra], original_chars.index(" ") + 1)
        extended.write_onnx(base / "rec_rtl.onnx", work / "extended.onnx")
        expanded_export = verify_parity(work / "extended.onnx", extended, atol=5e-4)
        test = evaluate_bundle(work / "run/model", work / "prepared")
        metadata = build_kernel(
            root,
            work / "kernel",
            "example",
            "indra-smoke",
            ["example/prepared", "example/model"],
            root / "configs/documents-v5-fast-loose.json",
        )
        compile((work / "kernel/run.py").read_text(), "run.py", "exec")
        assert metadata["is_private"] is True
        return {
            "scope": "Functional CPU smoke test on invented text; not a production accuracy benchmark",
            "synthetic_lines": count,
            "prepared_counts": data["counts"],
            "dataset_sha256": data["data_sha256"],
            "base_onnx_sha256": hashlib.sha256((base / "rec_rtl.onnx").read_bytes()).hexdigest(),
            "training_steps": 2,
            "maximum_parameter_change": delta,
            "selected_model": result["selected"]["name"],
            "test_used_for_selection": False,
            "selected_export_parity": result["export_parity"],
            "trained_candidate_export_parity": candidate_export,
            "expanded_vocabulary_export_parity": expanded_export,
            "post_selection_test_samples": test["metrics"]["all"]["samples"],
            "kaggle_kernel_compiled": True,
            "kaggle_kernel_private": True,
            "kaggle_gpu_training_rerun": False,
            "environment": json.loads((work / "run/environment.json").read_text()),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = smoke(args.base_model.resolve(), Path(__file__).resolve().parents[1])
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
