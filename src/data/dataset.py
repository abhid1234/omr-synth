"""Framework-neutral manifest dataset; Torch wrapping belongs in training code."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.data.split import split_name
from src.vocab.dsl import validate


@dataclass(frozen=True)
class ManifestItem:
    id: str
    image_path: Path
    target_path: Path
    curriculum: int
    split: str


class ManifestDataset:
    def __init__(self, manifest: Path, split: str | None = None, max_curriculum: int = 2) -> None:
        self.root = manifest.parent
        self.items: list[ManifestItem] = []
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            assigned = split_name(row["id"])
            item = ManifestItem(row["id"], self.root / row["image"], self.root / row["target"],
                                int(row["curriculum"]), assigned)
            if item.curriculum <= max_curriculum and (split is None or assigned == split):
                if not item.image_path.is_file() or not item.target_path.is_file():
                    raise FileNotFoundError(f"manifest line {line_no} points to a missing file")
                validate(item.target_path.read_text(encoding="utf-8"))
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, ManifestItem]:
        item = self.items[index]
        image = Image.open(item.image_path).convert("L")
        target = item.target_path.read_text(encoding="utf-8").strip()
        return image, target, item
