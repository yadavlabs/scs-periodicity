import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from figure_generation.fig1_sampling import sampling_source, sampling_component


class SamplingComponentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.index = pd.read_csv(self.root / 'data/detection/reference/index_axon.csv',
                                 usecols=['index_axon_id'])

    def tearDown(self):
        plt.close('all')

    def test_manifest_coverage_and_index_half(self):
        frame, index, diameters = sampling_source(SimpleNamespace(root=self.root, index=self.index))
        self.assertEqual(len(frame), 420)
        self.assertEqual(frame.cutline_id.nunique(), 35)
        self.assertEqual(diameters, [5.7, 8.7, 10, 11.5, 12.8, 14])
        selected = frame[frame.is_index_axon].iloc[0]
        self.assertEqual(selected.display_half, 'left')
        self.assertEqual(selected.ring_order_inner_to_outer, 6)
        self.assertEqual(index.axon_id, 'S01_140um_ph1')

    def test_vector_component_exports_and_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            exports = []
            v2 = SimpleNamespace(root=self.root, output=Path(directory), index=self.index,
                                 table=lambda *args: None,
                                 figure=lambda fig, stem: exports.append(stem) or fig)
            fig = sampling_component(v2)
            self.assertEqual(exports, ['fig1b_population_sampling_key'])
            self.assertEqual(len(fig.axes[1].patches), 12)
            svg = (Path(directory) / 'fig1b_population_sampling_key.svg').read_text(encoding='utf-8')
            self.assertIn('<text', svg)
            self.assertNotIn('<image', svg)
            self.assertIn('Phase 1', svg)
            notes = (Path(directory) / 'fig1b_population_sampling_key_caption_notes.txt').read_text(encoding='utf-8')
            self.assertIn('not nested axons or myelin layers', notes)


if __name__ == '__main__':
    unittest.main()
