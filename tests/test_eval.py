import unittest

from evaluate import aggregate_predictions
from src.eval.metrics import corpus_metrics, edit_distance, symbol_error_rate


class MetricsTest(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("a b c".split(), "a x c d".split()), 2)

    def test_ser(self):
        self.assertEqual(symbol_error_rate("a b c d", "a x c"), 0.5)

    def test_corpus_identity(self):
        self.assertEqual(corpus_metrics([("a b", "a b")]),
                         {"ser": 0.0, "token_accuracy": 1.0, "exact_match": 1.0})

    def test_aggregation_overall_and_by_notation(self):
        summary = aggregate_predictions([
            ("western", "a b c", "a b c"),
            ("jianpu", "a b", "a x"),
            ("western", "a", "a z"),
        ])
        self.assertEqual(summary["overall"]["count"], 3)
        self.assertEqual(summary["overall"]["token_edit_distance"], 2)
        self.assertEqual(summary["by_notation"]["western"]["count"], 2)
        self.assertEqual(summary["by_notation"]["jianpu"]["count"], 1)
        self.assertEqual(summary["by_notation"]["sargam"]["count"], 0)
        self.assertAlmostEqual(summary["by_notation"]["jianpu"]["symbol_error_rate"], 0.5)

    def test_aggregation_reports_lengths_and_exact_match(self):
        summary = aggregate_predictions([("sargam", "a b", "a b c")])
        row = summary["by_notation"]["sargam"]
        self.assertEqual(row["mean_target_length"], 2.0)
        self.assertEqual(row["mean_predicted_length"], 3.0)
        self.assertEqual(row["exact_match_rate"], 0.0)


if __name__ == "__main__": unittest.main()
