"""Fixed vocabulary tokenizer for OMRDSL v1."""
from __future__ import annotations

from dataclasses import dataclass

from .dsl import validate


def default_vocabulary() -> list[str]:
    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    structural = ["VERSION_1", "PART_BEGIN", "PART_END", "BAR_END", "REST",
                  "CHORD_BEGIN", "CHORD_END", "CLEF_G2"]
    keys = [f"KEY_{n}" for n in range(-4, 5)]
    times = ["TIME_2_4", "TIME_3_4", "TIME_4_4", "TIME_6_8"]
    bars = [f"BAR_{n}" for n in range(1, 17)]
    voices = ["VOICE_1", "VOICE_2"]
    durations = [f"DUR_{n}" for n in (1, 2, 3, 4, 6, 8, 12, 16)]
    accidentals = ("", "s", "b")
    pitches = [f"{prefix}_{letter}{acc}{octave}"
               for prefix in ("NOTE", "PITCH")
               for octave in range(2, 7)
               for letter in "CDEFGAB" for acc in accidentals]
    return special + structural + keys + times + bars + voices + durations + pitches


@dataclass
class Tokenizer:
    vocabulary: list[str] | None = None

    def __post_init__(self) -> None:
        self.vocabulary = self.vocabulary or default_vocabulary()
        if len(self.vocabulary) != len(set(self.vocabulary)):
            raise ValueError("duplicate vocabulary tokens")
        self.token_to_id = {token: i for i, token in enumerate(self.vocabulary)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    def encode(self, text: str, strict: bool = True) -> list[int]:
        validate(text)
        if strict:
            unknown = [t for t in text.split() if t not in self.token_to_id]
            if unknown:
                raise ValueError(f"unknown tokens: {unknown}")
        unk = self.token_to_id["<unk>"]
        return [self.token_to_id.get(t, unk) for t in text.split()]

    def decode(self, ids: list[int], validate_result: bool = True) -> str:
        try:
            text = " ".join(self.vocabulary[i] for i in ids if i != self.pad_id)
        except IndexError as exc:
            raise ValueError("token ID outside vocabulary") from exc
        if validate_result:
            validate(text)
        return text
