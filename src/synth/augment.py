"""Seeded scan/manuscript-like image degradations."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def degrade(path: Path, seed: int, level: int) -> None:
    if level == 0:
        return
    rng = random.Random(seed)
    image = Image.open(path).convert("L")
    image = Image.new("L", (image.width + 24, image.height + 24), 255)
    source = Image.open(path).convert("L")
    image.paste(source, (12, 12))
    angle = rng.uniform(-1.2, 1.2) * level
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
    image = image.filter(ImageFilter.GaussianBlur(radius=0.15 + 0.18 * level))
    image = ImageEnhance.Contrast(image).enhance(0.92 + rng.random() * 0.16)
    image.save(path, optimize=True)
