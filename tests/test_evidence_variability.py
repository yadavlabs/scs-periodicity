import unittest
import numpy as np
import pandas as pd
from population.figure_generation.evidence_variability import evidence_summary, SPECS


class EvidenceVariabilityTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([dict(freq=100, representative_regime='I_50', cv=.5, rep_index=i,
                                  **{column: value for column, *_ in SPECS})
                             for i, value in enumerate([-1., 0., 0., 1.])])

    def test_summary_separates_spread_and_mean_uncertainty(self):
        summary = evidence_summary(self.rows())
        self.assertEqual(len(summary), 4)
        self.assertTrue(summary['median'].eq(0).all())
        self.assertTrue(summary.q25.eq(-.25).all())
        self.assertTrue(summary.q75.eq(.25).all())
        self.assertTrue(np.allclose(summary['sem'], np.std([-1., 0., 0., 1.], ddof=1)/2))
        self.assertTrue(summary.n.eq(4).all())

    def test_duplicate_aliases_are_not_pooled_as_repetitions(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            evidence_summary(pd.concat([rows, rows.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
