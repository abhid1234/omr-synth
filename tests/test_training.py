import unittest

from src.training import lr_at


class LearningRateScheduleTest(unittest.TestCase):
    def test_linear_warmup_ramps_from_zero_to_base(self):
        self.assertEqual(lr_at(0, 3e-4, 10, 100, 3e-5), 0.0)
        self.assertAlmostEqual(lr_at(5, 3e-4, 10, 100, 3e-5), 1.5e-4)
        self.assertAlmostEqual(lr_at(10, 3e-4, 10, 100, 3e-5), 3e-4)

    def test_cosine_decay_reaches_minimum(self):
        midpoint = lr_at(55, 3e-4, 10, 100, 3e-5)
        self.assertAlmostEqual(midpoint, (3e-4 + 3e-5) / 2)
        self.assertAlmostEqual(lr_at(100, 3e-4, 10, 100, 3e-5), 3e-5)


if __name__ == "__main__":
    unittest.main()
