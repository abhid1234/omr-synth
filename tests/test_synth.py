import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dataset import ManifestDataset
from src.synth.generate import generate_dataset
from src.synth.augment import degrade
from src.synth.render import JianpuRenderer, SargamRenderer
from src.synth.scores import generate_score
from src.vocab.dsl import parse, serialize


class SynthesisTest(unittest.TestCase):
    def test_real_rendered_pair_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_dataset(root, count=3, seed=90)
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual({row["notation"] for row in rows}, {"western", "jianpu", "sargam"})
            for row in rows:
                image_path = root / row["image"]
                with Image.open(image_path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreater(image.width * image.height, 10_000)
                    extrema = image.convert("L").getextrema()
                    self.assertLess(extrema[0], 100, "render should contain dark ink")
                    self.assertGreater(extrema[1], 240, "render should contain light paper")
                target = (root / row["target"]).read_text()
                self.assertIn(f"NOTATION_{row['notation'].upper()}", target)
            dataset = ManifestDataset(manifest)
            self.assertEqual(len(dataset), 3)
            self.assertTrue(dataset[0][1].startswith("<bos> VERSION_1"))

    def test_jianpu_is_drawn_from_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "jianpu.png"
            score, record = generate_score(44, level=3, measures=2)
            JianpuRenderer(scale=1).render(score, record, output)
            with Image.open(output) as image:
                self.assertGreater(image.width, 500)
                self.assertLess(image.convert("L").getextrema()[0], 50)

    def test_hard_curriculum_is_seeded(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.png", Path(directory) / "b.png"
            Image.new("L", (320, 100), 255).save(first)
            Image.new("L", (320, 100), 255).save(second)
            degrade(first, 123, 3)
            degrade(second, 123, 3)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_sargam_render_and_round_trip_share_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sargam.png"
            score, record = generate_score(51, level=3, measures=2)
            renderer = SargamRenderer(scale=1)
            renderer.render(score, record, output)
            with Image.open(output) as image:
                self.assertGreater(image.width, 500)
                self.assertLess(image.convert("L").getextrema()[0], 50)
            targets = {name: serialize(record, notation=name)
                       for name in ("western", "jianpu", "sargam")}
            for name, target in targets.items():
                parsed_record, notation = parse(target)
                self.assertEqual(parsed_record, record)
                self.assertEqual(notation, name)
            semantic_targets = {target.replace(f" NOTATION_{name.upper()}", "")
                                for name, target in targets.items()}
            self.assertEqual(len(semantic_targets), 1)

    def test_sargam_chromatic_degree_conventions(self):
        renderer = SargamRenderer(scale=1)
        expected = (("S", False, False), ("R", True, False), ("R", False, False),
                    ("G", True, False), ("G", False, False), ("M", False, False),
                    ("M", False, True), ("P", False, False), ("D", True, False),
                    ("D", False, False), ("N", True, False), ("N", False, False))
        pitches = ("C4", "Cs4", "D4", "Ds4", "E4", "F4", "Fs4", "G4",
                   "Gs4", "A4", "As4", "B4")
        self.assertEqual(tuple(renderer._degree(pitch, 0)[:3] for pitch in pitches), expected)


if __name__ == "__main__": unittest.main()
