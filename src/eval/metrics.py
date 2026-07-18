"""Token-level OMR evaluation with no ML dependencies."""
from __future__ import annotations

from collections.abc import Iterable


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for col, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def symbol_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def corpus_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    total_edits = total_ref = aligned_correct = aligned_total = exact = count = 0
    for reference, hypothesis in pairs:
        ref, hyp = reference.split(), hypothesis.split()
        total_edits += edit_distance(ref, hyp)
        total_ref += len(ref)
        aligned_correct += sum(a == b for a, b in zip(ref, hyp))
        aligned_total += max(len(ref), len(hyp))
        exact += reference.strip() == hypothesis.strip()
        count += 1
    return {"ser": total_edits / total_ref if total_ref else 0.0,
            "token_accuracy": aligned_correct / aligned_total if aligned_total else 1.0,
            "exact_match": exact / count if count else 0.0}
