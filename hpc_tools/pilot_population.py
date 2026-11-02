"""Run a small, isolated NEURON check for a released population axon.

Each invocation loads one axon geometry in a fresh process. Results are saved
under results/population/pilot and never overwrite production simulation data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from population.detection.detection_core import (
    PopulationMRGRunner,
    _config_path,
    _reference_trace,
    load_population_basis,
    prepend_zeros_ms,
)
from population.space_time_v.spacetime_core import prepare_mrg_workers


ROOT = Path(__file__).resolve().parents[1]


def reference_threshold(root: Path, axon_id: str) -> float:
    path = root / "data/population/reference/index-screen/index_axon_thresholds.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["axon_id"] == axon_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one reference threshold for {axon_id}, found {len(rows)}")
    return float(rows[0]["I_first_uA"])


def train_trace(root: Path, basis: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    path = root / "data/time-patterns/population/cv0.0_f20_waveforms.npz"
    with np.load(path, allow_pickle=False) as waveforms:
        label = next((name for name in waveforms.files if "regular" in name.lower()), waveforms.files[0])
        current = np.asarray(waveforms[label], dtype=np.float32)
    time_ms = np.arange(current.size, dtype=np.float32) * np.float32(1000.0 / 20000.0)
    voltage = np.zeros((current.size, int(basis["n_valid_nodes"])), dtype=np.float32)
    voltage[current < 0] = basis["V_neg_mV"]
    voltage[current > 0] = basis["V_pos_mV"]
    return time_ms, voltage


def run(root: Path, axon_id: str, mode: str) -> dict[str, object]:
    basis_path = root / "data/space-time-V/population" / f"{axon_id}_1uA_VeBasis.npz"
    basis = load_population_basis(basis_path)
    prepare_mrg_workers(root=root)
    config = _config_path(root, basis["diameter_um"], basis["n_valid_nodes"])
    threshold = reference_threshold(root, axon_id)
    if mode == "impulse":
        time_ms, voltage = _reference_trace(basis)
    elif mode == "train":
        time_ms, voltage = train_trace(root, basis)
    else:
        raise ValueError(mode)
    time_ms, voltage = prepend_zeros_ms(time_ms, voltage)
    runner = PopulationMRGRunner(
        time_ms, voltage, config_path=config,
        readout_node_number=basis["readout_node_number"],
    )
    amplitude = threshold + 20.0
    spike_times, spike_count = runner.run_mrg_for_amp(amplitude)
    if mode == "impulse" and spike_count < 1:
        raise AssertionError(f"{axon_id} did not spike at reference threshold + 20 uA")
    if not np.all(np.isfinite(spike_times)):
        raise AssertionError("Non-finite spike time")
    return {
        "status": "passed",
        "mode": mode,
        "axon_id": axon_id,
        "diameter_um": basis["diameter_um"],
        "phase_id": basis["phase_id"],
        "n_valid_nodes": basis["n_valid_nodes"],
        "readout_node_number": basis["readout_node_number"],
        "reference_threshold_uA": threshold,
        "tested_amplitude_uA": amplitude,
        "spike_count": spike_count,
        "first_spike_ms": float(spike_times[0]) if spike_count else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--axon-id", required=True)
    parser.add_argument("--mode", choices=("impulse", "train"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, args.axon_id, args.mode)
    output_dir = root / "results/population/pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.axon_id}_{args.mode}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
