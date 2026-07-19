"""Export genuine held-out OMR model predictions for the web demo."""
from __future__ import annotations

import argparse
import base64
import json
import math
import random
from pathlib import Path

from evaluate import _decode_generated
from predict import preprocess_image
from src.data.dataset import ManifestDataset, ManifestItem
from src.eval.metrics import corpus_metrics, edit_distance
from src.vocab.dsl import parse
from src.vocab.tokenizer import Tokenizer


NOTATIONS = ("western", "jianpu", "sargam")
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pitch_hz(pitch: str) -> float:
    """Convert the OMRDSL pitch spelling to Hz, with C4 at semitone zero."""
    if len(pitch) < 2 or pitch[0] not in _SEMITONES:
        raise ValueError(f"invalid pitch: {pitch}")
    octave = int(pitch[-1])
    accidental = pitch[1:-1]
    alteration = {"": 0, "s": 1, "b": -1}[accidental]
    semitone = _SEMITONES[pitch[0]] + alteration + 12 * (octave - 4)
    return 440.0 * 2.0 ** ((semitone - 9) / 12.0)


def omrdsl_to_notes(text: str) -> tuple[list[dict[str, object]], bool]:
    """Parse OMRDSL into playback events; malformed input is an honest failure."""
    try:
        record, _ = parse(text)
        notes: list[dict[str, object]] = []
        for measure in record.measures:
            for voice in measure.voices:
                for event in voice:
                    beats = event.duration / 4.0
                    if event.kind == "REST":
                        notes.append({"rest": True, "beats": beats})
                    else:
                        for pitch in event.pitches:
                            notes.append({"pitch": pitch, "hz": _pitch_hz(pitch),
                                          "beats": beats})
        return notes, True
    except (IndexError, KeyError, TypeError, ValueError):
        return [], False


def _data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def select_examples(dataset: ManifestDataset, per_notation: int, seed: int) -> list[ManifestItem]:
    """Select a stable seeded sample independently for every notation."""
    rng = random.Random(seed)
    selected: list[ManifestItem] = []
    for notation in NOTATIONS:
        candidates = sorted((item for item in dataset.items if item.notation == notation),
                            key=lambda item: item.id)
        rng.shuffle(candidates)
        selected.extend(candidates[:per_notation])
    return selected


def _summarize(pairs: list[tuple[str, str]]) -> dict[str, int | float]:
    metrics = corpus_metrics(pairs)
    return {"count": len(pairs), "exact_match": metrics["exact_match"],
            "mean_token_accuracy": metrics["token_accuracy"]}


def build_summary(checkpoint: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in NOTATIONS}
    all_pairs: list[tuple[str, str]] = []
    for row in rows:
        pair = (str(row["target"]), str(row["prediction"]))
        grouped[str(row["notation"])].append(pair)
        all_pairs.append(pair)
    return {
        "checkpoint": checkpoint.name,
        "count": len(rows),
        "counts": {name: len(grouped[name]) for name in NOTATIONS},
        "overall": _summarize(all_pairs),
        "by_notation": {name: _summarize(grouped[name]) for name in NOTATIONS},
    }


def _print_summary(summary: dict[str, object]) -> None:
    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"{'group':<10} {'count':>6} {'token acc':>11} {'exact':>9}")
    rows = [("overall", summary["overall"]), *summary["by_notation"].items()]
    for name, row in rows:
        print(f"{name:<10} {row['count']:>6d} {row['mean_token_accuracy']:>11.4f} "
              f"{row['exact_match']:>9.4f}")


def export_predictions(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Demo prediction export is deferred: install the optional Torch stack in a separate environment."
        ) from exc
    from src.model.omr import OMRTransformer

    torch.manual_seed(args.seed)
    dataset = ManifestDataset(args.manifest, args.split, args.max_curriculum)
    items = select_examples(dataset, args.per_notation, args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    tokenizer = Tokenizer(vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = OMRTransformer(len(vocabulary), tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for item in items:
            pixels = preprocess_image(item.image_path, (args.width, args.height))
            image = torch.from_numpy(pixels).unsqueeze(0).to(device)
            ids = model.generate(image, tokenizer.token_to_id["<bos>"],
                                 tokenizer.token_to_id["<eos>"])[0].cpu().tolist()
            prediction = _decode_generated(ids, tokenizer)
            target = item.target_path.read_text(encoding="utf-8").strip()
            metrics = corpus_metrics([(target, prediction)])
            notes, parse_ok = omrdsl_to_notes(prediction)
            rows.append({
                "notation": item.notation, "curriculum": item.curriculum, "id": item.id,
                "image": _data_uri(item.image_path), "target": target,
                "prediction": prediction, "token_accuracy": metrics["token_accuracy"],
                "exact_match": target == prediction,
                "token_edit_distance": edit_distance(target.split(), prediction.split()),
                "notes": notes, "parse_ok": parse_ok,
            })

    result = {"summary": build_summary(args.checkpoint, rows), "predictions": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print_summary(result["summary"])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("samples/manifest.jsonl"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--per-notation", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-curriculum", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("demo_predictions.json"))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.per_notation < 0:
        parser.error("--per-notation must be non-negative")
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if not 0 <= args.max_curriculum <= 3:
        parser.error("--max-curriculum must be between 0 and 3")
    export_predictions(args)


if __name__ == "__main__":
    main()
