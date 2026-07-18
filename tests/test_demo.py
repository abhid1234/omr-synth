import tempfile
import unittest
from pathlib import Path

from src.demo.build_demo import CURATED_SAMPLES, build_demo


class DemoTest(unittest.TestCase):
    def test_demo_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            self.assertEqual(build_demo(output), output)
            document = output.read_text(encoding="utf-8")
            self.assertTrue(document.startswith("<!doctype html>"))
            self.assertEqual(document.count("data:image/png;base64,"), 3 * len(CURATED_SAMPLES))
            self.assertIn("One score. Three notations. One musical target.", document)
            self.assertIn("Sargam solfège notation", document)
            self.assertIn("VERSION_1", document)
            self.assertIn("NOTE_", document)
            self.assertIn("curriculum level 3", document)
            self.assertNotIn("http://", document)
            self.assertNotIn("https://", document)
            self.assertNotIn("<script", document.lower())


if __name__ == "__main__":
    unittest.main()
