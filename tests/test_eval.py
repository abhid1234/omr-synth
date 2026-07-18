import unittest

from src.eval.metrics import corpus_metrics, edit_distance, symbol_error_rate


class MetricsTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("a b c".split(), "a x c d".split()), 2)

    def test_ser(self):
        self.assertEqual(symbol_error_rate("a b c d", "a x c"), 0.5)

    def test_corpus_identity(self):
        self.assertEqual(corpus_metrics([("a b", "a b")]),
                         {"ser": 0.0, "token_accuracy": 1.0, "exact_match": 1.0})


if __name__ == "__main__": unittest.main()
