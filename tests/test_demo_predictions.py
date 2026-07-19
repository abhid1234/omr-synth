import math
import unittest

from demo_predictions import omrdsl_to_notes


class DemoPredictionNotesTests(unittest.TestCase):
    def test_notes_rests_and_pitch_frequency(self):
        text = ("<bos> VERSION_1 PART_BEGIN NOTATION_WESTERN CLEF_G2 KEY_0 TIME_4_4 "
                "BAR_1 VOICE_1 NOTE_C4 DUR_4 REST DUR_2 NOTE_A4 DUR_8 BAR_END "
                "PART_END <eos>")
        notes, parse_ok = omrdsl_to_notes(text)
        self.assertTrue(parse_ok)
        self.assertEqual(notes[0]["pitch"], "C4")
        self.assertTrue(math.isclose(notes[0]["hz"], 261.625565, rel_tol=1e-6))
        self.assertEqual(notes[0]["beats"], 1.0)
        self.assertEqual(notes[1], {"rest": True, "beats": 0.5})
        self.assertEqual(notes[2], {"pitch": "A4", "hz": 440.0, "beats": 2.0})

    def test_malformed_prediction_is_not_dropped(self):
        notes, parse_ok = omrdsl_to_notes("not OMRDSL")
        self.assertEqual(notes, [])
        self.assertFalse(parse_ok)


if __name__ == "__main__":
    unittest.main()
