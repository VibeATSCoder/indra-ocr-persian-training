"""The historical CTC + teacher KL + L2-anchor objective."""

import torch
from torch.nn import functional as F


def teacher_kl(logits, teacher, lengths, temperature=2.0):
    """Exclude padded timesteps and compare only the teacher's existing classes."""
    classes = teacher.shape[-1]
    values = F.kl_div(
        F.log_softmax(logits[:, :, :classes].float() / temperature, -1),
        F.softmax(teacher.float() / temperature, -1),
        reduction="none",
    ).sum(-1)
    mask = torch.arange(logits.shape[1], device=logits.device)[None, :] < lengths[:, None]
    return (values * mask).sum() / mask.sum().clamp_min(1) * temperature**2


def anchored_loss(model, anchors):
    return sum((value - anchors[name]).square().sum() for name, value in model.named_parameters())
