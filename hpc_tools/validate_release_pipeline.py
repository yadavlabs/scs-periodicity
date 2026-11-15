"""Check that every scheduled population analysis stage produced its public outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from detection.accepted_cv0 import INDEX_SHA256, SOURCE_SHA256
from detection.protocol import CV0_PROTOCOL_VERSION, DETECTION_PROTOCOL_VERSION


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "data/detection/index_axon_thresholds.csv",
    "data/detection/index_regime_amplitudes.csv",
    "data/detection/index_cv0_threshold_max.csv",
    "data/detection/population_responses.csv",
    "data/detection/population_cv0_based_responses.csv",
    "data/detection/behavior_fit_inputs/index_regime_amplitudes_pre_imax.csv",
    "data/detection/population_recruitment_outcomes.csv",
    "data/detection/reference_phase_threshold_summary.csv",
    "data/detection/behavior_fit_inputs/provenance.json",
    "data/features/population_cv0_based_features.csv",
    "data/features/population_cv0_template_evidence.csv",
    "data/features/population_cv0_time_representations_v5.npz",
    "data/detection/behavioral_matching/run_manifest.json",
    "data/detection/true_population_trial_summary.csv",
    "data/discrimination/control/validation.json",
    "data/discrimination/scope/validation.json",
    "data/discrimination/locked/inputs/manifest.json",
    "data/discrimination/locked/followup/validation.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_protocol(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream), None)
    if not row or not row.get("detection_protocol_version"):
        raise ValueError(f"Missing detection protocol in {path}")
    return str(row["detection_protocol_version"])


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete release pipeline:\n" + "\n".join(missing))
    expected_protocols = {
        "data/detection/index_regime_amplitudes.csv": DETECTION_PROTOCOL_VERSION,
        "data/detection/population_responses.csv": DETECTION_PROTOCOL_VERSION,
        "data/detection/index_cv0_threshold_max.csv": CV0_PROTOCOL_VERSION,
        "data/detection/population_cv0_based_responses.csv": CV0_PROTOCOL_VERSION,
        "data/features/population_cv0_based_features.csv": CV0_PROTOCOL_VERSION,
        "data/features/population_cv0_template_evidence.csv": CV0_PROTOCOL_VERSION,
    }
    for name, expected in expected_protocols.items():
        if first_protocol(ROOT / name) != expected:
            raise ValueError(f"Unexpected detection protocol in {name}")
    for name in (
        "data/features/population_cv0_based_features.csv",
        "data/features/population_cv0_template_evidence.csv",
    ):
        with (ROOT / name).open(newline="", encoding="utf-8") as stream:
            if sum(1 for _ in stream) - 1 != 49_500:
                raise ValueError(f"Incomplete accepted CV0 output: {name}")
    source = ROOT / "data/detection/behavior_fit_inputs/index_regime_amplitudes_pre_imax.csv"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Accepted CV0 input fingerprint mismatch")
    index = ROOT / "data/detection/index_cv0_threshold_max.csv"
    if hashlib.sha256(index.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != INDEX_SHA256:
        raise ValueError("Accepted CV0 index table fingerprint mismatch")
    with np.load(ROOT / "data/features/population_cv0_time_representations_v5.npz", allow_pickle=False) as traces:
        if str(traces["detection_protocol_version"].item()) != CV0_PROTOCOL_VERSION:
            raise ValueError("Population trace archive has the wrong CV0 protocol")
    for name in (
        "data/discrimination/control/validation.json",
        "data/discrimination/scope/validation.json",
        "data/discrimination/locked/followup/validation.json",
    ):
        status = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if status.get("status") != "ready" or status.get("errors"):
            raise ValueError(f"Analysis validation failed: {name}")
    summary = json.loads((ROOT / "data/detection/true_population_trial_summary_manifest.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("source_rows") != 20_790_000
            or summary.get("trials") != 49_500):
        raise ValueError("Incomplete true-CV summary")
    if digest(ROOT / "data/detection/true_population_trial_summary.csv") != summary["summary_sha256"]:
        raise ValueError("True-CV summary fingerprint mismatch")
    from discrimination.locked_mixture_followup import verify
    verify(ROOT, ROOT / "data/discrimination/locked/inputs")
    report = {"status": "ready", "required_files": len(REQUIRED), "trials": 49_500}
    output = ROOT / "logs/pipeline/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
