import tempfile
import unittest
from pathlib import Path

import numpy as np
from music21 import converter
from PIL import Image

from predict import decode_prediction, preprocess_image, write_round_trip
from src.synth.scores import generate_score
from src.vocab.dsl import parse, serialize
from src.vocab.tokenizer import Tokenizer


class PredictHelpersTest(unittest.TestCase):
    def test_preprocess_is_padded_ink_positive_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            image = Image.new("L", (4, 2), 255)
            image.putpixel((0, 0), 0); image.save(image_path)
            pixels = preprocess_image(image_path, (8, 8))
            self.assertEqual(pixels.shape, (1, 8, 8))
            self.assertEqual(pixels.dtype, np.float32)
            self.assertEqual(float(pixels.min()), 0.0)
            self.assertEqual(float(pixels.max()), 1.0)

    def test_tokenizer_dsl_musicxml_round_trip(self):
        _, record = generate_score(81, level=2, measures=2)
        target = serialize(record, "jianpu")
        tokenizer = Tokenizer()
        decoded = decode_prediction(tokenizer.encode(target), tokenizer.vocabulary)
        parsed_record, notation = parse(decoded)
        self.assertEqual(notation, "jianpu")
        self.assertEqual(serialize(parsed_record, notation), target)
        with tempfile.TemporaryDirectory() as directory:
            musicxml = Path(directory) / "round-trip.musicxml"
            write_round_trip(decoded, musicxml)
            loaded = converter.parse(musicxml)
            self.assertEqual(len(loaded.parts), 1)
            self.assertGreater(len(loaded.parts[0].getElementsByClass("Measure")), 0)


if __name__ == "__main__": unittest.main()
