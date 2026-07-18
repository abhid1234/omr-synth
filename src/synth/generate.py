"""CLI for creating image/OMRDSL pairs and a JSONL manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.synth.augment import degrade
from src.synth.render import VerovioRenderer
from src.synth.scores import generate_score
from src.vocab.dsl import serialize


def generate_dataset(output_dir: Path, count: int = 24, seed: int = 1729) -> Path:
    if count < 1:
        raise ValueError("count must be positive")
    images = output_dir / "images"
    targets = output_dir / "targets"
    images.mkdir(parents=True, exist_ok=True)
    targets.mkdir(parents=True, exist_ok=True)
    renderer = VerovioRenderer()
    manifest_path = output_dir / "manifest.jsonl"
    lines = []
    for index in range(count):
        example_seed = seed + index
        level = index % 3
        example_id = f"synth-{example_seed:08d}"
        image_path = images / f"{example_id}.png"
        target_path = targets / f"{example_id}.omrdsl"
        score, record = generate_score(example_seed, level)
        renderer.render(score, image_path)
        degrade(image_path, example_seed, level)
        target_path.write_text(serialize(record) + "\n", encoding="utf-8")
        with Image.open(image_path) as image:
            width, height = image.size
        lines.append(json.dumps({"id": example_id, "image": str(image_path.relative_to(output_dir)),
                                 "target": str(target_path.relative_to(output_dir)), "seed": example_seed,
                                 "curriculum": level, "width": width, "height": height,
                                 "renderer": "verovio", "format": "OMRDSL-v1"}, sort_keys=True))
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    manifest = generate_dataset(args.output, args.count, args.seed)
    print(f"generated {args.count} pairs in {manifest.parent}")


if __name__ == "__main__":
    main()
