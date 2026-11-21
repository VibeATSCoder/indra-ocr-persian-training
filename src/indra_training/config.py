"""Validated training settings; paths in JSON are relative to the config file."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Candidate:
    name: str = "fast-loose"
    learning_rate: float = 8e-5
    anchor_weight: float = 2e-4
    distillation_weight: float = 0.05


@dataclass
class TrainConfig:
    base_model_dir: str = "../models/base"
    dataset_dir: str = "../data/prepared"
    output_dir: str = "../runs/new-dataset"
    seed: int = 20261104
    batch_size: int = 192
    device: str = "cuda:0"
    max_steps: int | None = None
    train_seconds: float = 4500
    max_epochs: int = 1000
    eval_every_steps: int = 0
    eval_every_seconds: float = 1100
    samples_per_epoch: int | None = None
    num_threads: int = 2
    num_workers: int = 2
    warmup_fraction: float = 0.03
    minimum_lr_ratio: float = 0.02
    gradient_clip: float = 5.0
    temperature: float = 2.0
    allow_vocab_extension: bool = True
    augmentation: bool = True
    domain_sampling_mass: dict[str, float] = field(default_factory=dict)
    validation_weights: dict[str, float] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=lambda: [Candidate()])

    def validate(self):
        scalars = (
            self.train_seconds,
            self.eval_every_seconds,
            self.warmup_fraction,
            self.minimum_lr_ratio,
            self.gradient_clip,
            self.temperature,
        )
        scalars += tuple(
            value
            for candidate in self.candidates
            for value in (
                candidate.learning_rate,
                candidate.anchor_weight,
                candidate.distillation_weight,
            )
        )
        scalars += tuple(self.domain_sampling_mass.values()) + tuple(
            self.validation_weights.values()
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("All numerical training settings must be finite")
        if self.batch_size < 1 or self.max_epochs < 1 or self.num_threads < 1:
            raise ValueError("batch_size, max_epochs and num_threads must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.train_seconds <= 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("train_seconds must be positive; warmup_fraction must be in [0, 1)")
        if not 0 < self.minimum_lr_ratio <= 1 or self.temperature <= 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid LR floor, temperature or gradient clip")
        if self.eval_every_steps < 0 or self.eval_every_seconds <= 0:
            raise ValueError("Invalid validation interval")
        if self.samples_per_epoch is not None and self.samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive or null")
        names = [c.name for c in self.candidates]
        if not names or len(set(names)) != len(names):
            raise ValueError("Candidate names must be present and unique")
        for candidate in self.candidates:
            if not candidate.name or any(
                c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in candidate.name
            ):
                raise ValueError("Candidate names may contain lowercase letters, digits, - and _")
            if (
                candidate.learning_rate <= 0
                or min(candidate.anchor_weight, candidate.distillation_weight) < 0
            ):
                raise ValueError("Invalid candidate learning rate or regularization")
        for mapping in (self.domain_sampling_mass, self.validation_weights):
            if mapping and (any(w < 0 for w in mapping.values()) or sum(mapping.values()) <= 0):
                raise ValueError("Domain weights must be nonnegative and have a positive sum")
        return self


def load_config(path: str | Path, overrides: dict | None = None) -> TrainConfig:
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(values) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    raw_candidates = values.pop("candidates", None)
    config = TrainConfig(**values)
    if raw_candidates is not None:
        config.candidates = [Candidate(**row) for row in raw_candidates]
    for key in ("base_model_dir", "dataset_dir", "output_dir"):
        value = Path(getattr(config, key))
        setattr(config, key, str((path.parent / value).resolve()))
    for key, value in (overrides or {}).items():
        if value is not None:
            if key.endswith("_dir"):
                value = str(Path(value).resolve())
            setattr(config, key, value)
    return config.validate()


def config_dict(config: TrainConfig) -> dict:
    return asdict(config)
