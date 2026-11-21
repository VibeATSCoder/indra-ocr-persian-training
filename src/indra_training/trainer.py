"""Configurable recognizer fine-tuning; validation selects, explicit evaluation tests."""

from __future__ import annotations

import collections
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ._vendor.rec_preproc import pad_batch
from ._vendor.rtl import flip
from .config import TrainConfig, config_dict
from .data import canonical, clean, load_prepared
from .model import IndraRTL, verify_parity
from .objectives import anchored_loss, teacher_kl


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def vocabulary(directory: Path) -> list[str]:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    vocab = meta["vocab_rtl"]
    vocab = json.loads(vocab) if isinstance(vocab, str) else vocab
    chars = list(vocab["chars"])
    if vocab.get("blank", 0) != 0 or len(set(chars)) != len(chars):
        raise ValueError("Expected an Indra vocabulary with unique characters and blank ID 0")
    return chars


def augment(ink):
    """Light scan-like perturbations, preserving the line's text and width."""
    ink = ink.copy()
    if random.random() < 0.4:
        scale = random.uniform(0.5, 1.0)
        small = cv2.resize(ink, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ink = cv2.resize(small, (ink.shape[1], 32), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.35:
        ink = cv2.GaussianBlur(ink, (3, 3), random.uniform(0.2, 0.9))
    if random.random() < 0.15:
        op = cv2.erode if random.random() < 0.5 else cv2.dilate
        ink = op(ink, np.ones((2, 1), np.uint8))
    if random.random() < 0.35:
        ink = np.clip(
            ink.astype(np.float32) * random.uniform(0.6, 1.15)
            + np.random.normal(0, random.uniform(0, 6), ink.shape),
            0,
            255,
        ).astype(np.uint8)
    if random.random() < 0.2:
        ok, encoded = cv2.imencode(
            ".jpg", 255 - ink, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)]
        )
        if ok:
            ink = 255 - cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return ink


class Lines(Dataset):
    def __init__(self, root, rows, training=False):
        self.root, self.rows, self.training = root, rows, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        ink = np.asarray(Image.open(self.root / row["image"])).copy()
        if self.training:
            ink = augment(ink)
        return ink, row


def collate(items):
    x, widths = pad_batch([ink.astype(np.float32) / 255 for ink, _ in items])
    rows = [row for _, row in items]
    labels = torch.tensor(
        [token for row in rows for token in row.get("target", [])], dtype=torch.long
    )
    sizes = torch.tensor([len(row.get("target", [])) for row in rows], dtype=torch.long)
    return torch.from_numpy(x), torch.from_numpy(widths), labels, sizes, rows


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def batches(rows, config, epoch, train=False):
    rng = np.random.default_rng(config.seed + epoch)
    if train:
        counts = collections.Counter(row["domain"] for row in rows)
        mass = config.domain_sampling_mass or {domain: 1.0 for domain in counts}
        if set(counts) - set(mass):
            raise ValueError(
                f"Missing domain_sampling_mass entries: {sorted(set(counts) - set(mass))}"
            )
        weights = np.asarray([mass[row["domain"]] / counts[row["domain"]] for row in rows])
        if weights.sum() <= 0:
            raise ValueError("No positive sampling mass for the training domains")
        order = rng.choice(
            len(rows),
            config.samples_per_epoch or len(rows),
            replace=True,
            p=weights / weights.sum(),
        )
    else:
        order = np.arange(len(rows))
    result = []
    for start in range(0, len(order), 8192):
        pool = sorted(order[start : start + 8192].tolist(), key=lambda i: rows[i]["width"])
        result.extend(
            pool[i : i + config.batch_size] for i in range(0, len(pool), config.batch_size)
        )
    if train:
        rng.shuffle(result)
    return result


def loader(root, rows, config, epoch=0, train=False):
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        Lines(root, rows, training=train and config.augmentation),
        batch_sampler=batches(rows, config, epoch, train),
        collate_fn=collate,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def decode(logits, lengths, characters):
    decoded = []
    for values, length in zip(logits.argmax(-1).cpu().tolist(), lengths.cpu().tolist()):
        previous, text = -1, []
        for token in values[:length]:
            if token and token != previous:
                text.append(characters[token - 1])
            previous = token
        decoded.append(clean(flip("".join(text))))
    return decoded


@torch.inference_mode()
def evaluate_model(model, data, characters, device):
    model.eval()
    totals = collections.defaultdict(collections.Counter)
    for x, widths, _, _, rows in data:
        logits, lengths = model(x.to(device), widths.to(device))
        for row, prediction in zip(rows, decode(logits, lengths, characters)):
            text = row["text"]
            values = {
                "characters": len(text),
                "character_edits": Levenshtein.distance(text, prediction),
                "words": len(text.split()),
                "word_edits": Levenshtein.distance(text.split(), prediction.split()),
                "canonical_characters": len(canonical(text)),
                "canonical_edits": Levenshtein.distance(canonical(text), canonical(prediction)),
                "samples": 1,
                "exact": int(text == prediction),
            }
            for domain in {"all", row["domain"]}:
                totals[domain].update(values)
    return {
        domain: {
            **dict(v),
            "cer": v["character_edits"] / max(v["characters"], 1),
            "wer": v["word_edits"] / max(v["words"], 1),
            "canonical_cer": v["canonical_edits"] / max(v["canonical_characters"], 1),
        }
        for domain, v in totals.items()
    }


def validation_score(metrics, weights):
    weights = weights or {domain: 1.0 for domain in metrics if domain != "all"}
    valid = {
        domain: weight for domain, weight in weights.items() if domain in metrics and weight > 0
    }
    if not valid:
        raise ValueError("No validation domains have a positive selection weight")
    return sum(
        weight * (0.7 * metrics[d]["cer"] + 0.3 * metrics[d]["canonical_cer"])
        for d, weight in valid.items()
    ) / sum(valid.values())


def learning_rate(peak, fraction, config):
    if config.warmup_fraction and fraction < config.warmup_fraction:
        return peak * (0.1 + 0.9 * fraction / config.warmup_fraction)
    progress = (fraction - config.warmup_fraction) / (1 - config.warmup_fraction)
    return peak * (
        config.minimum_lr_ratio
        + (1 - config.minimum_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))
    )


def export_bundle(model, base, destination, characters):
    destination.mkdir()
    for path in base.iterdir():
        if path.is_file() and (
            path.suffix in {".onnx", ".npz"}
            or path.name in {"meta.json", "LICENSE", "math_meta.json"}
        ):
            if path.name not in {"smoke-updated.onnx", "roundtrip.onnx"}:
                shutil.copyfile(path, destination / path.name)
    model = model.cpu().eval()
    model.write_onnx(base / "rec_rtl.onnx", destination / "rec_rtl.onnx")
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    meta["vocab_rtl"] = json.dumps({"blank": 0, "chars": characters}, ensure_ascii=False)
    write_json(destination / "meta.json", meta)
    return verify_parity(destination / "rec_rtl.onnx", model, atol=5e-4)


def run_training(config: TrainConfig) -> dict:
    config.validate()
    base, dataset, output = map(
        Path, (config.base_model_dir, config.dataset_dir, config.output_dir)
    )
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory, not {output}")
    for name in ("rec_rtl.onnx", "meta.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(f"Missing base-model file: {base / name}")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a GPU runtime or explicitly set device=cpu")
    device = torch.device(config.device)
    torch.set_num_threads(config.num_threads)
    cv2.setNumThreads(1)
    manifest, rows = load_prepared(dataset)
    original_chars = vocabulary(base)
    # Vocabulary discovery reads training labels only, not validation/test labels.
    added = sorted({c for row in rows["train"] for c in row["visual"]} - set(original_chars))
    if added and not config.allow_vocab_extension:
        raise ValueError(f"Training labels have unsupported characters: {added!r}")
    characters = original_chars + added
    token = {c: i + 1 for i, c in enumerate(characters)}
    rejected = collections.Counter()
    for split in rows:
        accepted = []
        for row in rows[split]:
            if set(row["visual"]) - set(characters):
                if split == "train":
                    raise ValueError("Training contains unknown characters")
                # Keep such evaluation rows: they count as recognition errors.
                row["target"] = []
                rejected[f"{split}_rows_with_unseen_characters"] += 1
            else:
                row["target"] = [token[c] for c in row["visual"]]
            accepted.append(row)
        rows[split] = accepted
    output.mkdir(parents=True)
    write_json(output / "resolved-config.json", config_dict(config))
    packages = {}
    for package in (
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
        "Pillow",
        "opencv-python-headless",
        "rapidfuzz",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unavailable"
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
    except FileNotFoundError:
        revision = None
    base_sha256 = hashlib.sha256((base / "rec_rtl.onnx").read_bytes()).hexdigest()
    write_json(
        output / "environment.json",
        {
            "packages": packages,
            "git_commit": revision,
            "cuda": torch.version.cuda,
            "device": str(device),
            "base_onnx_sha256": base_sha256,
        },
    )
    write_json(output / "dataset-manifest.json", manifest)
    pretrained_parity = verify_parity(base / "rec_rtl.onnx")
    teacher = IndraRTL.from_onnx(base / "rec_rtl.onnx").to(device).eval().requires_grad_(False)
    val_loader = loader(dataset, rows["validation"], config)
    baseline = evaluate_model(teacher, val_loader, original_chars, device)
    write_json(output / "baseline-validation.json", baseline)
    baseline_score = validation_score(baseline, config.validation_weights)
    options = [{"name": "baseline", "score": baseline_score}]
    summaries = []
    for candidate in config.candidates:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        student = IndraRTL.from_onnx(base / "rec_rtl.onnx")
        if added:
            student.add_characters(added, original_chars.index(" ") + 1)
        student.to(device)
        anchors = {name: value.detach().clone() for name, value in student.named_parameters()}
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=candidate.learning_rate, weight_decay=0
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        best, step, history = float("inf"), 0, []
        start = last_eval = time.monotonic()
        for epoch in range(config.max_epochs):
            training_loader = loader(dataset, rows["train"], config, epoch, train=True)
            for batch_index, (x, widths, labels, sizes, _) in enumerate(training_loader):
                student.train()
                x, widths = x.to(device), widths.to(device)
                elapsed = time.monotonic() - start
                fraction = max(
                    step / config.max_steps if config.max_steps else 0,
                    elapsed / config.train_seconds,
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(candidate.learning_rate, fraction, config)
                with torch.autocast(
                    device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    logits, lengths = student(x, widths)
                    with torch.no_grad():
                        old_logits, _ = teacher(x, widths)
                ctc = F.ctc_loss(
                    logits.float().log_softmax(-1).transpose(0, 1),
                    labels.to(device),
                    lengths,
                    sizes.to(device),
                    blank=0,
                    zero_infinity=False,
                )
                kd = teacher_kl(logits, old_logits, lengths, config.temperature)
                anchor = anchored_loss(student, anchors)
                loss = ctc + candidate.distillation_weight * kd + candidate.anchor_weight * anchor
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Nonfinite loss: inspect the dataset and lower the learning rate"
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), config.gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
                step += 1
                elapsed = time.monotonic() - start
                final = (
                    config.max_steps is not None and step >= config.max_steps
                ) or elapsed >= config.train_seconds
                final = final or (
                    epoch == config.max_epochs - 1 and batch_index + 1 == len(training_loader)
                )
                due = (
                    config.eval_every_steps > 0 and step % config.eval_every_steps == 0
                ) or time.monotonic() - last_eval >= config.eval_every_seconds
                if due or final:
                    metrics = evaluate_model(student, val_loader, characters, device)
                    score = validation_score(metrics, config.validation_weights)
                    entry = {
                        "step": step,
                        "epoch": epoch,
                        "seconds": elapsed,
                        "loss": float(loss.detach()),
                        "ctc": float(ctc.detach()),
                        "kl": float(kd.detach()),
                        "anchor": float(anchor.detach()),
                        "gradient_norm": float(gradient_norm),
                        "score": score,
                        "validation": metrics,
                    }
                    history.append(entry)
                    if score < best:
                        best = score
                        checkpoint_path = output / f"best-{candidate.name}.pt"
                        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
                        torch.save(
                            {
                                "state_dict": student.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "characters": characters,
                                "candidate": candidate.__dict__,
                                "step": step,
                                "epoch": epoch,
                                "validation_score": score,
                                "dataset_sha256": manifest["data_sha256"],
                                "config": config_dict(config),
                            },
                            temporary_path,
                        )
                        temporary_path.replace(checkpoint_path)
                    write_json(output / f"history-{candidate.name}.json", history)
                    print(
                        json.dumps(
                            {
                                "candidate": candidate.name,
                                "step": step,
                                "score": score,
                                "loss": entry["loss"],
                            }
                        ),
                        flush=True,
                    )
                    last_eval = time.monotonic()
                if final:
                    break
            if final:
                break
        options.append({"name": candidate.name, "score": best})
        summaries.append(
            {"name": candidate.name, "steps": step, "seconds": time.monotonic() - start}
        )
        del student, optimizer, anchors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = min(options, key=lambda option: option["score"])
    # Commit selection before exporting; this command never evaluates the test split.
    write_json(
        output / "selection.json",
        {"selected": selected, "candidates": options, "split": "validation"},
    )
    selected_model = IndraRTL.from_onnx(base / "rec_rtl.onnx")
    selected_chars = original_chars
    if selected["name"] != "baseline":
        if added:
            selected_model.add_characters(added, original_chars.index(" ") + 1)
        checkpoint = torch.load(
            output / f"best-{selected['name']}.pt", map_location="cpu", weights_only=True
        )
        selected_model.load_state_dict(checkpoint["state_dict"])
        selected_chars = characters
    parity = export_bundle(selected_model, base, output / "model", selected_chars)
    report = {
        "selected": selected,
        "candidates": options,
        "training": summaries,
        "dataset_sha256": manifest["data_sha256"],
        "base_onnx_sha256": base_sha256,
        "added_training_characters": added,
        "evaluation_character_coverage": dict(rejected),
        "pretrained_parity": pretrained_parity,
        "export_parity": parity,
        "test_evaluated": False,
    }
    write_json(output / "results.json", report)
    write_json(output / "model/training.json", report)
    return report


def export_checkpoint(checkpoint_path, base_model_dir, destination):
    base, destination = Path(base_model_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Export output already exists: {destination}")
    torch.set_num_threads(2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    original = vocabulary(base)
    characters = checkpoint["characters"]
    if characters[: len(original)] != original:
        raise ValueError("Checkpoint vocabulary does not extend this base model's vocabulary")
    model = IndraRTL.from_onnx(base / "rec_rtl.onnx")
    if len(characters) > len(original):
        model.add_characters(characters[len(original) :], original.index(" ") + 1)
    model.load_state_dict(checkpoint["state_dict"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    parity = export_bundle(model, base, destination, characters)
    write_json(
        destination / "training.json",
        {
            "checkpoint_step": checkpoint["step"],
            "candidate": checkpoint["candidate"],
            "dataset_sha256": checkpoint["dataset_sha256"],
            "export_parity": parity,
        },
    )
    return parity


def evaluate_bundle(model_dir, dataset_dir, split="test", batch_size=32, device="cpu"):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    manifest, rows = load_prepared(dataset_dir)
    if not rows[split]:
        raise ValueError(f"Dataset has no {split} examples")
    config = TrainConfig(batch_size=batch_size, device=device, num_workers=0)
    torch.set_num_threads(config.num_threads)
    model_dir = Path(model_dir)
    model = IndraRTL.from_onnx(model_dir / "rec_rtl.onnx").to(device)
    return {
        "split": split,
        "dataset_sha256": manifest["data_sha256"],
        "metrics": evaluate_model(
            model,
            loader(Path(dataset_dir), rows[split], config),
            vocabulary(model_dir),
            torch.device(device),
        ),
    }
