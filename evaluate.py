"""Evaluate an OMR checkpoint on a deterministic manifest split."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from predict import preprocess_image
from src.data.dataset import ManifestDataset
from src.eval.metrics import corpus_metrics, edit_distance
from src.vocab.tokenizer import Tokenizer


NOTATIONS = ("western", "jianpu", "sargam")


def aggregate_predictions(
    examples: Iterable[tuple[str, str, str]],
) -> dict[str, object]:
    """Aggregate (notation, reference, hypothesis) triples with existing metrics."""
    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in NOTATIONS}
    all_pairs: list[tuple[str, str]] = []
    for notation, reference, hypothesis in examples:
        grouped.setdefault(notation, []).append((reference, hypothesis))
        all_pairs.append((reference, hypothesis))

    def summarize(pairs: list[tuple[str, str]]) -> dict[str, int | float]:
        metrics = corpus_metrics(pairs)
        edits = sum(edit_distance(ref.split(), hyp.split()) for ref, hyp in pairs)
        target_tokens = sum(len(ref.split()) for ref, _ in pairs)
        predicted_tokens = sum(len(hyp.split()) for _, hyp in pairs)
        count = len(pairs)
        return {
            "count": count,
            "token_accuracy": metrics["token_accuracy"],
            "token_edit_distance": edits,
            "symbol_error_rate": metrics["ser"],
            "exact_match_rate": metrics["exact_match"],
            "mean_predicted_length": predicted_tokens / count if count else 0.0,
            "mean_target_length": target_tokens / count if count else 0.0,
        }

    return {
        "overall": summarize(all_pairs),
        "by_notation": {name: summarize(grouped[name]) for name in NOTATIONS},
    }


def _decode_generated(ids: list[int], tokenizer: Tokenizer) -> str:
    eos_id = tokenizer.token_to_id["<eos>"]
    if eos_id in ids:
        ids = ids[:ids.index(eos_id) + 1]
    return tokenizer.decode(ids, validate_result=False)


def _print_table(summary: dict[str, object]) -> None:
    rows = [("overall", summary["overall"])]
    rows.extend(summary["by_notation"].items())
    print("\nEvaluation metrics", file=sys.stderr)
    print(f"{'group':<10} {'count':>6} {'tok acc':>9} {'edit':>7} {'SER':>9} "
          f"{'exact':>9} {'pred len':>10} {'tgt len':>9}", file=sys.stderr)
    for name, row in rows:
        print(f"{name:<10} {row['count']:>6d} {row['token_accuracy']:>9.4f} "
              f"{row['token_edit_distance']:>7d} {row['symbol_error_rate']:>9.4f} "
              f"{row['exact_match_rate']:>9.4f} {row['mean_predicted_length']:>10.2f} "
              f"{row['mean_target_length']:>9.2f}", file=sys.stderr)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Evaluation is deferred: install the optional Torch stack in a separate environment."
        ) from exc
    from src.model.omr import OMRTransformer

    torch.manual_seed(0)
    dataset = ManifestDataset(args.manifest, args.split, args.max_curriculum)
    items = dataset.items[:args.limit] if args.limit is not None else dataset.items
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocabulary = checkpoint["vocabulary"]
    tokenizer = Tokenizer(vocabulary)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = OMRTransformer(len(vocabulary), tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions: list[tuple[str, str, str]] = []
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            arrays = [preprocess_image(item.image_path, (args.width, args.height))
                      for item in batch]
            images = torch.stack([torch.from_numpy(array) for array in arrays]).to(device)
            generated = model.generate(
                images, tokenizer.token_to_id["<bos>"], tokenizer.token_to_id["<eos>"]
            ).cpu().tolist()
            for item, ids in zip(batch, generated):
                reference = item.target_path.read_text(encoding="utf-8").strip()
                predictions.append((item.notation, reference,
                                    _decode_generated(ids, tokenizer)))

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "max_curriculum": args.max_curriculum,
        "image_size": {"width": args.width, "height": args.height},
        **aggregate_predictions(predictions),
    }
    _print_table(summary)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("samples/manifest.jsonl"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-curriculum", type=int, default=3)
    parser.add_argument("--limit", type=int, help="maximum number of split examples")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, help="optional JSON summary path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("--width and --height must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if not 0 <= args.max_curriculum <= 3:
        parser.error("--max-curriculum must be between 0 and 3")
    evaluate(args)


if __name__ == "__main__":
    main()
