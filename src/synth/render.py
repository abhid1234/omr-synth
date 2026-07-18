"""Swappable renderers for Western staff, Jianpu, and Sargam notation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import cairosvg
import verovio
from music21 import stream
from PIL import Image, ImageDraw, ImageFont

from src.vocab.dsl import Event, ScoreRecord


class Renderer(Protocol):
    notation: str

    def render(self, score: stream.Score, record: ScoreRecord, output: Path) -> None: ...


class VerovioRenderer:
    notation = "western"

    def __init__(self, scale: int = 42) -> None:
        self.scale = scale

    def render(self, score: stream.Score, record: ScoreRecord, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="omr-synth-") as temp_dir:
            xml_path = Path(temp_dir) / "score.musicxml"
            score.write("musicxml", fp=str(xml_path))
            toolkit = verovio.toolkit()
            toolkit.setOptions({"pageWidth": 1800, "pageHeight": 900, "scale": self.scale,
                                "adjustPageHeight": True, "breaks": "none", "footer": "none",
                                "header": "none"})
            if not toolkit.loadFile(str(xml_path)):
                raise RuntimeError("Verovio could not load generated MusicXML")
            svg = toolkit.renderToSVG(1)
            cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(output),
                             background_color="#ffffff")


class JianpuRenderer:
    """Draw Jianpu directly from the canonical semantic record using Pillow."""

    notation = "jianpu"
    _tonics = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E",
               -1: "F", -2: "B", -3: "E", -4: "A"}
    _tonic_labels = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E",
                     -1: "F", -2: "B♭", -3: "E♭", -4: "A♭"}
    _letters = "CDEFGAB"

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 25 * scale)
            self.small_font = ImageFont.truetype("DejaVuSans.ttf", 12 * scale)
        except OSError:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def _degree(self, pitch: str, fifths: int) -> tuple[str, int]:
        letter, octave = pitch[0], int(pitch[-1])
        tonic = self._tonics[fifths]
        degree = (self._letters.index(letter) - self._letters.index(tonic)) % 7 + 1
        # Middle Jianpu register is centered around the tonic nearest C4.
        absolute = octave * 7 + self._letters.index(letter)
        tonic_absolute = 4 * 7 + self._letters.index(tonic)
        register = (absolute - tonic_absolute) // 7
        return str(degree), register

    def _draw_event(self, draw: ImageDraw.ImageDraw, event: Event, x: int, y: int,
                    fifths: int) -> int:
        symbol, register = ("0", 0) if event.kind == "REST" else self._degree(event.pitches[0], fifths)
        draw.text((x, y), symbol, font=self.font, fill=20, anchor="mm")
        dot_y = y - 27 * self.scale if register > 0 else y + 27 * self.scale
        for offset in range(abs(register)):
            radius = 2 * self.scale
            cy = dot_y + offset * (-7 if register > 0 else 7) * self.scale
            draw.ellipse((x-radius, cy-radius, x+radius, cy+radius), fill=20)
        underlines = 2 if event.duration == 1 else 1 if event.duration in (2, 3) else 0
        for line in range(underlines):
            yy = y + (20 + line * 5) * self.scale
            draw.line((x-10*self.scale, yy, x+10*self.scale, yy), fill=20, width=self.scale)
        if event.duration in (3, 6, 12):
            draw.ellipse((x+13*self.scale, y-2*self.scale, x+16*self.scale, y+self.scale), fill=20)
        dashes = max(0, event.duration // 4 - 1)
        for dash in range(dashes):
            dx = x + (22 + dash * 16) * self.scale
            draw.line((dx, y, dx+9*self.scale, y), fill=20, width=2*self.scale)
        return (34 + dashes * 16) * self.scale

    def render(self, score: stream.Score, record: ScoreRecord, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        width = 900 * self.scale
        voice_rows = sum(len(measure.voices) for measure in record.measures)
        height = max(180, 90 + voice_rows * 65) * self.scale
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)
        tonic = self._tonic_labels[record.key_fifths]
        draw.text((28*self.scale, 22*self.scale),
                  f"1={tonic}   {record.time[0]}/{record.time[1]}",
                  font=self.small_font, fill=15)
        x, y = 35*self.scale, 80*self.scale
        for measure in record.measures:
            for voice_no, events in enumerate(measure.voices):
                if x > width - 250*self.scale:
                    x, y = 35*self.scale, y + 65*self.scale
                if len(measure.voices) > 1:
                    draw.text((x, y), f"{voice_no + 1}:", font=self.small_font, fill=70, anchor="mm")
                    x += 22*self.scale
                for event in events:
                    if x > width - 70*self.scale:
                        x, y = 35*self.scale, y + 65*self.scale
                    x += self._draw_event(draw, event, x, y, record.key_fifths)
                draw.line((x, y-24*self.scale, x, y+25*self.scale), fill=20, width=2*self.scale)
                x += 18*self.scale
        ink_bbox = Image.eval(image, lambda value: 255 - value).getbbox()
        if ink_bbox:
            padding = 24 * self.scale
            left, top, right, bottom = ink_bbox
            box = (max(0, left-padding), max(0, top-padding),
                   min(image.width, right+padding), min(image.height, bottom+padding))
            image = image.crop(box)
        image.save(output, optimize=True)


class SargamRenderer:
    """Draw tonic-relative Sargam directly from the canonical event record."""

    notation = "sargam"
    _tonic_pcs = {0: 0, 1: 7, 2: 2, 3: 9, 4: 4,
                  -1: 5, -2: 10, -3: 3, -4: 8}
    _tonic_labels = {0: "C", 1: "G", 2: "D", 3: "A", 4: "E",
                     -1: "F", -2: "B♭", -3: "E♭", -4: "A♭"}
    # (letter, komal, tivra) for the twelve semitones above Sa.
    _degrees = (("S", False, False), ("R", True, False), ("R", False, False),
                ("G", True, False), ("G", False, False), ("M", False, False),
                ("M", False, True), ("P", False, False), ("D", True, False),
                ("D", False, False), ("N", True, False), ("N", False, False))
    _natural_pcs = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 25 * scale)
            self.small_font = ImageFont.truetype("DejaVuSans.ttf", 12 * scale)
        except OSError:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    @classmethod
    def _pitch_number(cls, pitch: str) -> int:
        accidental = pitch[1:-1]
        alteration = accidental.count("s") + accidental.count("#")
        alteration -= accidental.count("b") + accidental.count("-")
        return (int(pitch[-1]) + 1) * 12 + cls._natural_pcs[pitch[0]] + alteration

    def _degree(self, pitch: str, fifths: int) -> tuple[str, bool, bool, int]:
        pitch_number = self._pitch_number(pitch)
        tonic_pc = self._tonic_pcs[fifths]
        relative = (pitch_number - tonic_pc) % 12
        # Madhya Sa is the tonic at or immediately above C4's octave boundary.
        tonic_number = 60 + tonic_pc
        if tonic_number >= 72:
            tonic_number -= 12
        register = (pitch_number - tonic_number) // 12
        letter, komal, tivra = self._degrees[relative]
        return letter, komal, tivra, register

    def _draw_symbol(self, draw: ImageDraw.ImageDraw, pitch: str, x: int, y: int,
                     fifths: int) -> None:
        symbol, komal, tivra, register = self._degree(pitch, fifths)
        draw.text((x, y), symbol, font=self.font, fill=20, anchor="mm")
        if komal:
            draw.line((x-8*self.scale, y+17*self.scale, x+8*self.scale, y+17*self.scale),
                      fill=20, width=max(1, self.scale))
        if tivra:
            draw.line((x, y-24*self.scale, x, y-17*self.scale), fill=20,
                      width=max(1, self.scale))
        dot_y = y - 28*self.scale if register > 0 else y + 28*self.scale
        for offset in range(abs(register)):
            radius = 2*self.scale
            cy = dot_y + offset * (-7 if register > 0 else 7) * self.scale
            draw.ellipse((x-radius, cy-radius, x+radius, cy+radius), fill=20)

    def _draw_event(self, draw: ImageDraw.ImageDraw, event: Event, x: int, y: int,
                    fifths: int) -> int:
        if event.kind == "REST":
            draw.text((x, y), "–", font=self.font, fill=20, anchor="mm")
        else:
            # Chord tones share one time position and are stacked vertically.
            for index, pitch in enumerate(event.pitches):
                self._draw_symbol(draw, pitch, x, y-index*24*self.scale, fifths)
        underlines = 2 if event.duration == 1 else 1 if event.duration in (2, 3) else 0
        for line in range(underlines):
            yy = y + (22 + line*5)*self.scale
            draw.line((x-10*self.scale, yy, x+10*self.scale, yy), fill=20, width=self.scale)
        if event.duration in (3, 6, 12):
            draw.ellipse((x+13*self.scale, y-2*self.scale, x+16*self.scale, y+self.scale), fill=20)
        dashes = max(0, event.duration // 4 - 1)
        for dash in range(dashes):
            dx = x + (22 + dash*16)*self.scale
            draw.line((dx, y, dx+9*self.scale, y), fill=20, width=2*self.scale)
        return (36 + dashes*16)*self.scale

    def render(self, score: stream.Score, record: ScoreRecord, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        width = 900*self.scale
        voice_rows = sum(len(measure.voices) for measure in record.measures)
        height = max(180, 90 + voice_rows*70)*self.scale
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)
        draw.text((28*self.scale, 22*self.scale),
                  f"Sa = {self._tonic_labels[record.key_fifths]}   Tala {record.time[0]}/{record.time[1]}",
                  font=self.small_font, fill=15)
        x, y = 35*self.scale, 82*self.scale
        for measure in record.measures:
            for voice_no, events in enumerate(measure.voices):
                if x > width - 250*self.scale:
                    x, y = 35*self.scale, y + 70*self.scale
                if len(measure.voices) > 1:
                    draw.text((x, y), f"{voice_no + 1}:", font=self.small_font, fill=70, anchor="mm")
                    x += 22*self.scale
                for event in events:
                    if x > width - 70*self.scale:
                        x, y = 35*self.scale, y + 70*self.scale
                    x += self._draw_event(draw, event, x, y, record.key_fifths)
                draw.line((x, y-30*self.scale, x, y+27*self.scale), fill=20, width=2*self.scale)
                x += 18*self.scale
        ink_bbox = Image.eval(image, lambda value: 255-value).getbbox()
        if ink_bbox:
            padding = 24*self.scale
            left, top, right, bottom = ink_bbox
            image = image.crop((max(0, left-padding), max(0, top-padding),
                                min(image.width, right+padding), min(image.height, bottom+padding)))
        image.save(output, optimize=True)
