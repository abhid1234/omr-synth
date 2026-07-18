"""Procedural symbolic scores and matching semantic records."""
from __future__ import annotations

import random
from fractions import Fraction

from music21 import clef, key, meter, note, stream

from src.vocab.dsl import Event, MeasureRecord, ScoreRecord

DURATIONS = ((1, 0.25), (2, 0.5), (4, 1.0), (8, 2.0))
PITCHES = ("C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5")


def _events_for_bar(rng: random.Random, units: int, rests: bool) -> list[Event]:
    result: list[Event] = []
    remaining = units
    while remaining:
        choices = [(u, q) for u, q in DURATIONS if u <= remaining]
        duration, _ = rng.choice(choices)
        if rests and rng.random() < 0.18:
            result.append(Event("REST", duration))
        else:
            result.append(Event("NOTE", duration, (rng.choice(PITCHES),)))
        remaining -= duration
    return result


def _append_voice(container: stream.Stream, events: list[Event]) -> None:
    for event in events:
        quarter_length = Fraction(event.duration, 4)
        obj = note.Rest() if event.kind == "REST" else note.Note(event.pitches[0])
        obj.duration.quarterLength = quarter_length
        container.append(obj)


def generate_score(seed: int, level: int = 1, measures: int = 2) -> tuple[stream.Score, ScoreRecord]:
    if level not in (0, 1, 2):
        raise ValueError("curriculum level must be 0, 1, or 2")
    rng = random.Random(seed)
    time_choices = [(4, 4)] if level == 0 else [(2, 4), (3, 4), (4, 4), (6, 8)]
    numerator, denominator = rng.choice(time_choices)
    units_per_bar = numerator * (16 // denominator)
    fifths = 0 if level == 0 else rng.randint(-2, 2)

    score = stream.Score(id=f"synthetic-{seed}")
    part = stream.Part(id="P1")
    part.insert(0, clef.TrebleClef())
    part.insert(0, key.KeySignature(fifths))
    part.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
    records: list[MeasureRecord] = []
    for number in range(1, measures + 1):
        measure = stream.Measure(number=number)
        first = _events_for_bar(rng, units_per_bar, rests=level >= 1)
        voice_records = [tuple(first)]
        if level == 2:
            v1, v2 = stream.Voice(id="1"), stream.Voice(id="2")
            _append_voice(v1, first)
            second = _events_for_bar(rng, units_per_bar, rests=True)
            for i, event in enumerate(second):
                if event.kind == "NOTE":
                    pitch = PITCHES[max(0, PITCHES.index(event.pitches[0]) - 4)]
                    second[i] = Event("NOTE", event.duration, (pitch,))
            _append_voice(v2, second)
            measure.insert(0, v1)
            measure.insert(0, v2)
            voice_records.append(tuple(second))
        else:
            _append_voice(measure, first)
        part.append(measure)
        records.append(MeasureRecord(number, tuple(voice_records)))
    score.append(part)
    return score, ScoreRecord("G2", fifths, (numerator, denominator), tuple(records))
