"""Stable split assignment independent of manifest ordering."""
import hashlib


def split_name(example_id: str, train: int = 80, val: int = 10) -> str:
    if train < 1 or val < 0 or train + val >= 100:
        raise ValueError("invalid split percentages")
    bucket = int.from_bytes(hashlib.sha256(example_id.encode()).digest()[:4], "big") % 100
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"
