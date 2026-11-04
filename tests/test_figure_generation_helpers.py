"""Check a supplementary figure audit calculation."""

import unittest

from figure_generation.audit_supplementary_examples import accepted_count


class FigureGenerationHelperTests(unittest.TestCase):
    def test_short_gap_count_is_not_always_spike_loss_count(self):
        self.assertEqual(accepted_count([0, 15, 30]), 2)


if __name__ == "__main__":
    unittest.main()
