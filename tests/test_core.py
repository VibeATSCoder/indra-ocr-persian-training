import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from indra_training._vendor.rtl import flip, visual_label
from indra_training.config import Candidate, TrainConfig, load_config
from indra_training.data import check_disjoint, clean, group_split, load_prepared, prepare_dataset


class ConfigTests(unittest.TestCase):
    def test_rejects_nonfinite_and_duplicate_candidates(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            TrainConfig(train_seconds=float("nan")).validate()
        with self.assertRaisesRegex(ValueError, "unique"):
            TrainConfig(candidates=[Candidate(), Candidate()]).validate()

    def test_resolves_paths_and_rejects_unknown_options(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"base_model_dir": "models"}))
            self.assertEqual(load_config(path).base_model_dir, str(Path(folder) / "models"))
            path.write_text(json.dumps({"learning_rate_typo": 1}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path)


class DataTests(unittest.TestCase):
    def test_rtl_digits_and_latin_text_roundtrip(self):
        value = clean("سلام ۱۲۳، فایل ABC 456 آماده است")
        self.assertEqual(clean(flip(visual_label(value))), value)

    def test_document_split_is_stable(self):
        self.assertEqual(group_split("book-a", 123, 0.1, 0.1), group_split("book-a", 123, 0.1, 0.1))

    def test_document_and_label_overlap_fail(self):
        row = {"document_id": "a", "image_sha256": "one", "text_group": "text"}
        with self.assertRaisesRegex(ValueError, "document_id"):
            check_disjoint({"train": [row], "test": [dict(row)]})
        other = {"document_id": "b", "image_sha256": "two", "text_group": "text"}
        with self.assertRaisesRegex(ValueError, "text_group"):
            check_disjoint({"train": [row], "validation": [other]})
        check_disjoint({"train": [row], "validation": [other]}, allow_text_overlap=True)

    def make_input(self, root):
        records = []
        for index, split in enumerate(("train", "validation", "test")):
            image = Image.new("L", (300, 60), 255)
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 15, 200 + index * 12, 40), fill=0)
            draw.rectangle((40 + index * 20, 5, 50 + index * 20, 45), fill=0)
            image.save(root / f"{split}.png")
            records.append(
                {
                    "image": f"{split}.png",
                    "text": f"sample {index}",
                    "document_id": f"document-{index}",
                    "split": split,
                }
            )
        source = root / "labels.jsonl"
        source.write_text("".join(json.dumps(r) + "\n" for r in records))
        return source

    def test_prepare_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = self.make_input(root), root / "prepared"
            report = prepare_dataset(source, output)
            manifest, rows = load_prepared(output)
            self.assertEqual(manifest["data_sha256"], report["data_sha256"])
            self.assertEqual([len(rows[s]) for s in ("train", "validation", "test")], [1, 1, 1])
            with self.assertRaises(FileExistsError):
                prepare_dataset(source, output)
            image_path = output / rows["train"][0]["image"]
            image = np.asarray(Image.open(image_path)).copy()
            image[0, 0] ^= 255
            Image.fromarray(image).save(image_path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_prepared(output)

    def test_conflicting_explicit_document_split_fails_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_input(root)
            records = [json.loads(line) for line in source.read_text().splitlines()]
            records[1]["document_id"] = records[0]["document_id"]
            source.write_text("".join(json.dumps(row) + "\n" for row in records))
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "document_id"):
                prepare_dataset(source, output)
            self.assertFalse(output.exists())

    def test_template_duplicate_is_removed_from_training_not_heldout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = self.make_input(root)
            records = [json.loads(line) for line in source.read_text().splitlines()]
            image = Image.new("L", (300, 60), 255)
            ImageDraw.Draw(image).line((15, 10, 260, 45), fill=0, width=8)
            image.save(root / "extra.png")
            records.append(
                {
                    "image": "extra.png",
                    "text": "extra example",
                    "document_id": "extra-train",
                    "split": "train",
                }
            )
            duplicate = {**records[0], "document_id": "heldout-document", "split": "validation"}
            records.append(duplicate)
            source.write_text("".join(json.dumps(row) + "\n" for row in records))
            report = prepare_dataset(source, root / "prepared", drop_overlapping_images=True)
            _, rows = load_prepared(root / "prepared")
            self.assertEqual(report["cross_split_duplicate_images_removed"], {"train": 1})
            self.assertEqual([row["document_id"] for row in rows["train"]], ["extra-train"])
            self.assertIn("heldout-document", [row["document_id"] for row in rows["validation"]])


class ObjectiveTests(unittest.TestCase):
    def test_padding_does_not_change_teacher_kl_or_receive_gradients(self):
        import torch

        from indra_training.objectives import teacher_kl

        torch.manual_seed(3)
        logits = torch.randn(2, 5, 4, requires_grad=True)
        teacher = torch.randn(2, 5, 4)
        lengths = torch.tensor([2, 3])
        original = teacher_kl(logits, teacher, lengths)
        changed = logits.detach().clone()
        changed[0, 2:] += 300
        changed[1, 3:] -= 200
        torch.testing.assert_close(original.detach(), teacher_kl(changed, teacher, lengths))
        original.backward()
        self.assertEqual(float(logits.grad[0, 2:].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[1, 3:].abs().sum()), 0.0)


class KaggleTests(unittest.TestCase):
    def test_sealed_dataset_cannot_be_built_as_a_training_source(self):
        from indra_training.kaggle import build_kernel

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "kernel"
            with self.assertRaisesRegex(ValueError, "receipts-eval"):
                build_kernel(
                    root,
                    output,
                    "example",
                    "train",
                    ["example/receipts-eval"],
                    root / "configs/smoke.json",
                )
            self.assertFalse(output.exists())

    def test_kernel_contains_compilable_code_and_private_metadata(self):
        from indra_training.kaggle import build_kernel

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "kernel"
            metadata = build_kernel(
                root,
                output,
                "example",
                "train",
                ["example/prepared", "example/model"],
                root / "configs/smoke.json",
            )
            compile((output / "run.py").read_text(), "run.py", "exec")
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["dataset_sources"], ["example/prepared", "example/model"])


if __name__ == "__main__":
    unittest.main()
