"""Canonical, deliberately small event serialization for the initial curriculum."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

DSL_VERSION = "VERSION_1"


@dataclass(frozen=True)
class Event:
    kind: str
    duration: int
    pitches: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasureRecord:
    number: int
    voices: tuple[tuple[Event, ...], ...]


@dataclass(frozen=True)
class ScoreRecord:
    clef: str
    key_fifths: int
    time: tuple[int, int]
    measures: tuple[MeasureRecord, ...]


def serialize(record: ScoreRecord, notation: str = "western") -> str:
    """Serialize semantic music plus the visual notation used for its image."""
    notation_token = f"NOTATION_{notation.upper()}"
    if notation_token not in {"NOTATION_WESTERN", "NOTATION_JIANPU"}:
        raise ValueError(f"unsupported notation: {notation}")
    tokens = ["<bos>", DSL_VERSION, "PART_BEGIN", notation_token, f"CLEF_{record.clef}",
              f"KEY_{record.key_fifths}", f"TIME_{record.time[0]}_{record.time[1]}"]
    for measure in record.measures:
        tokens.append(f"BAR_{measure.number}")
        for voice_no, events in enumerate(measure.voices, 1):
            tokens.append(f"VOICE_{voice_no}")
            for event in events:
                if event.kind == "REST":
                    tokens.extend(("REST", f"DUR_{event.duration}"))
                elif len(event.pitches) == 1:
                    tokens.extend((f"NOTE_{event.pitches[0]}", f"DUR_{event.duration}"))
                else:
                    tokens.append("CHORD_BEGIN")
                    tokens.extend(f"PITCH_{p}" for p in sorted(event.pitches))
                    tokens.extend(("CHORD_END", f"DUR_{event.duration}"))
        tokens.append("BAR_END")
    tokens.extend(("PART_END", "<eos>"))
    target = " ".join(tokens)
    validate(target)
    return target


def tokens(text: str) -> list[str]:
    return text.strip().split()


def validate(text: str) -> None:
    ts = tokens(text)
    if len(ts) < 9 or ts[:3] != ["<bos>", DSL_VERSION, "PART_BEGIN"]:
        raise ValueError("invalid OMRDSL header")
    if ts[-2:] != ["PART_END", "<eos>"]:
        raise ValueError("invalid OMRDSL trailer")
    notation = [t for t in ts if t.startswith("NOTATION_")]
    if notation and (len(notation) != 1 or notation[0] not in
                     {"NOTATION_WESTERN", "NOTATION_JIANPU"}):
        raise ValueError("invalid notation marker")
    if not any(t.startswith("BAR_") and t != "BAR_END" for t in ts):
        raise ValueError("score has no measures")
    if ts.count("CHORD_BEGIN") != ts.count("CHORD_END"):
        raise ValueError("unbalanced chord markers")
    for index, token in enumerate(ts):
        if token in {"REST"} or token.startswith("NOTE_") or token == "CHORD_END":
            if index + 1 >= len(ts) or not ts[index + 1].startswith("DUR_"):
                raise ValueError(f"missing duration after {token}")


def iter_tokens(records: Iterable[ScoreRecord]) -> Iterable[str]:
    for record in records:
        yield from tokens(serialize(record))
