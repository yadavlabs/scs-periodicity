import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from population.detection.detection_core import population_trajectory_condition_jobs
from population.space_time_v.spacetime_core import (
    POPULATION_DIAMETERS_UM,
    discover_mrg_workers,
    make_axon_id,
    population_axon_jobs,
    population_phase_assignments,
)


class PopulationPhaseCohortTests(unittest.TestCase):
    def test_phase_assignment_is_deterministic_balanced_and_antithetic(self):
        cutlines = tuple(f"S{i:02d}" for i in range(1, 14))
        first = population_phase_assignments(cutlines)
        second = population_phase_assignments(reversed(cutlines))
        self.assertEqual(first, second)
        for cutline_id in cutlines:
            ph0, ph1 = first[cutline_id]
            self.assertAlmostEqual(ph1["phase_fraction"] - ph0["phase_fraction"], 0.5)
            self.assertEqual((ph0["phase_fraction"], ph1["phase_fraction"]), (0.25, 0.75))

    def test_production_job_manifest_has_420_phase_specific_axons(self):
        jobs = population_axon_jobs(REPO_ROOT)
        self.assertEqual(len(jobs), 35 * len(POPULATION_DIAMETERS_UM) * 2)
        self.assertEqual(len({job["axon_id"] for job in jobs}), len(jobs))
        self.assertTrue(all(job["axon_id"].endswith(("_ph0", "_ph1")) for job in jobs))

    def test_worker_discovery_separates_node_count_geometries(self):
        with tempfile.TemporaryDirectory() as temporary:
            basis_dir = Path(temporary)
            for phase_id, n_nodes in (("ph0", 18), ("ph1", 19)):
                axon_id = make_axon_id("S01", 14.0, phase_id)
                np.savez_compressed(
                    basis_dir / f"{axon_id}_1uA_VeBasis.npz",
                    axon_id=np.asarray(axon_id), diameter_um=np.asarray(14.0),
                    n_valid_nodes=np.asarray(n_nodes),
                    readout_node_number=np.asarray(n_nodes - 1),
                )
            workers = discover_mrg_workers(basis_dir)
            self.assertEqual([(w.axonnodes, w.readout_node_number) for w in workers], [(18, 17), (19, 18)])

    def test_recruitment_pruning_groups_each_phase_separately(self):
        jobs = population_trajectory_condition_jobs(REPO_ROOT)
        self.assertEqual(len(jobs), 35 * 2 * 55)
        for job in jobs:
            self.assertEqual(len(job["targets"]), len(POPULATION_DIAMETERS_UM))
            self.assertEqual({target["target_phase_id"] for target in job["targets"]}, {job["phase_id"]})


if __name__ == "__main__":
    unittest.main()
