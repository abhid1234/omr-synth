import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dataset import ManifestDataset
from src.synth.generate import generate_dataset


class SynthesisTest(unittest.TestCase):
    def test_real_rendered_pair_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_dataset(root, count=1, seed=90)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            image_path = root / rows[0]["image"]
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width * image.height, 10_000)
                extrema = image.convert("L").getextrema()
                self.assertLess(extrema[0], 100, "render should contain dark ink")
                self.assertGreater(extrema[1], 240, "render should contain light paper")
            dataset = ManifestDataset(manifest)
            self.assertEqual(len(dataset), 1)
            self.assertTrue(dataset[0][1].startswith("<bos> VERSION_1"))


if __name__ == "__main__": unittest.main()
