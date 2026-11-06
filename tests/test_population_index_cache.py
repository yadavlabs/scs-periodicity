from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from hpc_tools.prepare_population_index_cache import prepare_cache


def _write(path: Path, index_id: str, protocol: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["index_axon_id", "value"]
        if protocol is not None:
            fields.append("detection_protocol_version")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row = {"index_axon_id": index_id, "value": 1}
        if protocol is not None:
            row["detection_protocol_version"] = protocol
        writer.writerow(row)


class PopulationIndexCacheTests(unittest.TestCase):
    def test_archives_only_incompatible_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "data/detection/partials/index-regimes"
            stale = partial / "stale.csv"
            current = partial / "current.csv"
            _write(stale, "S01_140um")
            _write(current, "S01_128um")
            stale.with_suffix(".audit.json").write_text("{}", encoding="utf-8")

            dry = prepare_cache(
                root=root, expected_index_axon_id="S01_128um", apply=False,
                archive_tag="test",
            )
            self.assertEqual(dry["incompatible_csv_count"], 1)
            self.assertTrue(stale.is_file())

            applied = prepare_cache(
                root=root, expected_index_axon_id="S01_128um", apply=True,
                archive_tag="test",
            )
            self.assertEqual(applied["files_moved_or_planned"], 2)
            self.assertFalse(stale.exists())
            self.assertTrue(current.is_file())
            archive = Path(applied["archive_root"])
            self.assertTrue((archive / stale.relative_to(root)).is_file())
            self.assertTrue((archive / stale.with_suffix(".audit.json").relative_to(root)).is_file())

    def test_archives_same_index_when_protocol_is_stale_or_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/detection"
            current = data / "index_regime_amplitudes.csv"
            stale = data / "index_cv0_threshold_max.csv"
            legacy = data / "population_responses.csv"
            _write(current, "S01_140um_ph1", "canonical-cv0_tol0p01_v1")
            _write(stale, "S01_140um_ph1", "old-protocol")
            _write(legacy, "S01_140um_ph1")

            report = prepare_cache(
                root=root,
                expected_index_axon_id="S01_140um_ph1",
                expected_protocol_version="canonical-cv0_tol0p01_v1",
                apply=True,
                archive_tag="protocol-test",
            )
            self.assertEqual(report["incompatible_csv_count"], 2)
            self.assertTrue(current.is_file())
            self.assertFalse(stale.exists())
            self.assertFalse(legacy.exists())


if __name__ == "__main__":
    unittest.main()
