"""MusicXML -> Verovio SVG -> PNG renderer."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol

os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import cairosvg
import verovio
from music21 import stream


class Renderer(Protocol):
    def render(self, score: stream.Score, output: Path) -> None: ...


class VerovioRenderer:
    def __init__(self, scale: int = 42) -> None:
        self.scale = scale

    def render(self, score: stream.Score, output: Path) -> None:
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
