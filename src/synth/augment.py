"""Seeded scan/manuscript-like image degradations."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def degrade(path: Path, seed: int, level: int) -> None:
    if level not in (0, 1, 2, 3):
        raise ValueError("curriculum level must be 0, 1, 2, or 3")
    if level == 0:
        return
    rng = random.Random(seed)
    image = Image.open(path).convert("L")
    image = Image.new("L", (image.width + 24, image.height + 24), 255)
    source = Image.open(path).convert("L")
    image.paste(source, (12, 12))
    angle = rng.uniform(-1.2, 1.2) * min(level, 2.5)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    if level >= 2:
        skew = rng.uniform(1.5, 4.0)
        image = image.transform(image.size, Image.Transform.QUAD,
                                (skew, 0, 0, image.height - skew, image.width, image.height,
                                 image.width - skew, skew), resample=Image.Resampling.BICUBIC,
                                fillcolor=255)
    arr = np.asarray(image, dtype=np.float32)
    # Row displacement imitates inconsistent pen/staff alignment without changing labels.
    if level >= 2:
        shifted = np.empty_like(arr)
        phase = rng.random() * 6.28
        for row in range(arr.shape[0]):
            shifted[row] = np.roll(arr[row], int(2.2 * np.sin(row / 31 + phase)))
        arr = shifted
    if level >= 3:
        # Low-frequency paper mottling and stronger local baseline wander.
        yy, xx = np.indices(arr.shape)
        phase_x, phase_y = rng.random() * 6.28, rng.random() * 6.28
        texture = (5.5 * np.sin(xx / 73 + phase_x) + 4.0 * np.sin(yy / 41 + phase_y))
        arr = np.clip(arr + texture, 0, 255)
    paper = np.random.default_rng(seed).normal(0, 2.2 * level, arr.shape)
    arr = np.clip(arr + paper, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    if level >= 2:
        # MinFilter spreads dark ink; selective pale rows resemble faded staff segments.
        image = image.filter(ImageFilter.MinFilter(3))
        arr = np.asarray(image).copy()
        for row in range(rng.randrange(8, 17), arr.shape[0], rng.randrange(34, 54)):
            arr[row:row + 1] = np.maximum(arr[row:row + 1], rng.randint(150, 220))
        image = Image.fromarray(arr)
    if level >= 3:
        # Deterministic broken/faint ink patches and variable stroke width.
        arr = np.asarray(image).copy()
        for _ in range(max(5, image.width // 180)):
            x = rng.randrange(0, max(1, image.width - 12))
            y = rng.randrange(0, max(1, image.height - 5))
            w, h = rng.randrange(5, 24), rng.randrange(1, 4)
            arr[y:y+h, x:x+w] = np.maximum(arr[y:y+h, x:x+w], rng.randrange(175, 235))
        image = Image.fromarray(arr)
        if rng.random() < 0.65:
            image = image.filter(ImageFilter.MinFilter(3))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.15 + 0.18 * level))
    image = ImageEnhance.Contrast(image).enhance(0.92 + rng.random() * 0.16)
    image.save(path, optimize=True)
