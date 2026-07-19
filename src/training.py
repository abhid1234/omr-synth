"""Pure helpers for configuring training."""
from __future__ import annotations

import math


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    """Return the warmup/cosine learning rate for an optimizer step."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
