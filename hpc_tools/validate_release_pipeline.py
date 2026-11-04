"""Check that every scheduled population analysis stage produced its public outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "data/detection/population/index_axon_thresholds.csv",
    "data/detection/population/index_regime_amplitudes.csv",
    "data/detection/population/population_responses.csv",
    "data/detection/population/population_cv0_based_responses.csv",
    "data/detection/population/population_recruitment_outcomes.csv",
    "data/detection/population/reference_phase_threshold_summary.csv",
    "data/detection/behavior_fit_inputs/provenance.json",
    "data/features/population_cv0_based_features.csv",
    "data/features/population_cv0_template_evidence.csv",
    "results/detection/behavioral_matching/run_manifest.json",
    "results/detection/true_population_trial_summary.csv",
    "results/discrimination/validation.json",
    "results/discrimination_v5_scope_hypothesis_full/validation.json",
    "results/discrimination_locked_mixture_inputs/manifest.json",
    "results/discrimination_locked_mixture_followup/validation.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete release pipeline:\n" + "\n".join(missing))
    for name in (
        "results/discrimination/validation.json",
        "results/discrimination_v5_scope_hypothesis_full/validation.json",
        "results/discrimination_locked_mixture_followup/validation.json",
    ):
        status = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if status.get("status") != "ready" or status.get("errors"):
            raise ValueError(f"Analysis validation failed: {name}")
    summary = json.loads((ROOT / "results/detection/true_population_trial_summary_manifest.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "complete" or summary.get("source_rows") != 20_790_000
            or summary.get("trials") != 49_500):
        raise ValueError("Incomplete true-CV summary")
    if digest(ROOT / "results/detection/true_population_trial_summary.csv") != summary["summary_sha256"]:
        raise ValueError("True-CV summary fingerprint mismatch")
    from population.discrimination.locked_mixture_followup import verify
    verify(ROOT, ROOT / "results/discrimination_locked_mixture_inputs")
    report = {"status": "ready", "required_files": len(REQUIRED), "trials": 49_500}
    output = ROOT / "results/pipeline/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
