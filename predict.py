"""Deferred checkpoint inference entry point; Torch is loaded only for real prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.vocab.dsl import parse, to_music21
from src.vocab.tokenizer import Tokenizer


def preprocess_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Match training's grayscale, aspect-preserving pad and ink-positive scaling."""
    with Image.open(path) as source:
        image = ImageOps.pad(source.convert("L"), image_size, color=255)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return (1.0 - pixels)[None, :, :]


def decode_prediction(ids: list[int], vocabulary: list[str]) -> str:
    """Decode one generated sequence using the vocabulary stored in its checkpoint."""
    return Tokenizer(vocabulary).decode(ids)


def write_round_trip(dsl_text: str, musicxml: Path, render: Path | None = None) -> None:
    record, _ = parse(dsl_text)
    score = to_music21(record)
    musicxml.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(musicxml))
    if render is not None:
        from src.synth.render import VerovioRenderer
        VerovioRenderer().render(score, record, render)


def predict(args: argparse.Namespace) -> str:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Prediction is deferred: install the optional Torch stack in a separate environment.") from exc
    from src.model.omr import OMRTransformer

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    tokenizer = Tokenizer(vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = OMRTransformer(len(vocabulary), tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    pixels = preprocess_image(args.image, (args.width, args.height))
    images = torch.from_numpy(pixels).unsqueeze(0).to(device)
    ids = model.generate(images, tokenizer.token_to_id["<bos>"],
                         tokenizer.token_to_id["<eos>"], args.max_length)[0].tolist()
    result = decode_prediction(ids, vocabulary)
    print(result)
    if args.musicxml:
        write_round_trip(result, args.musicxml, args.render)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="trained .pt checkpoint")
    parser.add_argument("--image", type=Path, help="input notation image")
    parser.add_argument("--width", type=int, default=1024); parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--musicxml", type=Path, help="optional reconstructed MusicXML output")
    parser.add_argument("--render", type=Path, help="optional Verovio PNG (requires --musicxml)")
    parser.add_argument("--dry-run", action="store_true", help="validate arguments without importing Torch")
    return parser


def main() -> None:
    parser = build_parser(); args = parser.parse_args()
    if args.render and not args.musicxml:
        parser.error("--render requires --musicxml")
    if args.dry_run:
        print(json.dumps({"status": "deferred", "torch_imported": False,
                          "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                          "image": str(args.image) if args.image else None,
                          "paid_resources": False}))
        return
    if not args.checkpoint or not args.image:
        parser.error("--checkpoint and --image are required unless --dry-run is used")
    predict(args)


if __name__ == "__main__": main()
