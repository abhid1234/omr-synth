import unittest

from src.synth.scores import generate_score
from src.vocab.dsl import serialize
from src.vocab.tokenizer import Tokenizer


class TokenizerTest(unittest.TestCase):
    def test_round_trip_all_curricula(self):
        tokenizer = Tokenizer()
        for level in range(3):
            _, record = generate_score(100 + level, level)
            target = serialize(record)
            self.assertEqual(tokenizer.decode(tokenizer.encode(target)), target)

    def test_unknown_is_rejected_in_strict_mode(self):
        tokenizer = Tokenizer()
        _, record = generate_score(5, 0)
        target = serialize(record).replace("NOTE_C4", "NOTE_X9")
        with self.assertRaises(ValueError):
            tokenizer.encode(target)


if __name__ == "__main__": unittest.main()
