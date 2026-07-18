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


def parse(text: str) -> tuple[ScoreRecord, str]:
    """Parse validated OMRDSL into its canonical record and notation name."""
    validate(text)
    ts = tokens(text)
    notation = ts[3].removeprefix("NOTATION_").lower()
    clef = ts[4].removeprefix("CLEF_")
    key_fifths = int(ts[5].removeprefix("KEY_"))
    numerator, denominator = map(int, ts[6].removeprefix("TIME_").split("_"))
    measures: list[MeasureRecord] = []
    index = 7
    while ts[index] != "PART_END":
        if not ts[index].startswith("BAR_") or ts[index] == "BAR_END":
            raise ValueError(f"expected measure, got {ts[index]}")
        number = int(ts[index].removeprefix("BAR_")); index += 1
        voices: list[tuple[Event, ...]] = []
        while ts[index] != "BAR_END":
            expected_voice = f"VOICE_{len(voices) + 1}"
            if ts[index] != expected_voice:
                raise ValueError(f"expected {expected_voice}, got {ts[index]}")
            index += 1
            events: list[Event] = []
            while not ts[index].startswith("VOICE_") and ts[index] != "BAR_END":
                token = ts[index]
                if token == "REST":
                    events.append(Event("REST", int(ts[index + 1].removeprefix("DUR_"))))
                    index += 2
                elif token.startswith("NOTE_"):
                    events.append(Event("NOTE", int(ts[index + 1].removeprefix("DUR_")),
                                        (token.removeprefix("NOTE_"),)))
                    index += 2
                elif token == "CHORD_BEGIN":
                    index += 1; pitches: list[str] = []
                    while ts[index] != "CHORD_END":
                        if not ts[index].startswith("PITCH_"):
                            raise ValueError(f"expected chord pitch, got {ts[index]}")
                        pitches.append(ts[index].removeprefix("PITCH_")); index += 1
                    duration = int(ts[index + 1].removeprefix("DUR_"))
                    events.append(Event("CHORD", duration, tuple(pitches))); index += 2
                else:
                    raise ValueError(f"unexpected event token: {token}")
            voices.append(tuple(events))
        measures.append(MeasureRecord(number, tuple(voices))); index += 1
    return ScoreRecord(clef, key_fifths, (numerator, denominator), tuple(measures)), notation


def to_music21(record: ScoreRecord):
    """Build a music21 score from an OMRDSL semantic record."""
    from fractions import Fraction
    from music21 import chord, clef, key, meter, note, stream

    score = stream.Score()
    part = stream.Part(id="P1")
    if record.clef != "G2":
        raise ValueError(f"unsupported clef: {record.clef}")
    part.insert(0, clef.TrebleClef())
    part.insert(0, key.KeySignature(record.key_fifths))
    part.insert(0, meter.TimeSignature(f"{record.time[0]}/{record.time[1]}"))
    for measure_record in record.measures:
        measure = stream.Measure(number=measure_record.number)
        containers = ([measure] if len(measure_record.voices) == 1 else
                      [stream.Voice(id=str(i)) for i in range(1, len(measure_record.voices) + 1)])
        for container, events in zip(containers, measure_record.voices):
            for event in events:
                obj = (note.Rest() if event.kind == "REST" else
                       note.Note(event.pitches[0]) if len(event.pitches) == 1 else
                       chord.Chord(event.pitches))
                obj.duration.quarterLength = Fraction(event.duration, 4)
                container.append(obj)
            if container is not measure:
                measure.insert(0, container)
        part.append(measure)
    score.append(part)
    return score


def iter_tokens(records: Iterable[ScoreRecord]) -> Iterable[str]:
    for record in records:
        yield from tokens(serialize(record))
