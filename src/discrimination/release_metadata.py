"""Validate regenerated release outputs and write discrimination metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_ROOT = REPO_ROOT / "data" / "discrimination"

PER_BLOCK_MODELS = ("hybrid_combined", "rate_combined", "timing_combined")
PER_POINT_MODELS = ("hybrid_combined",)

EXPECTED_STAGE05_FILES = (
    "merged_results.csv",
    "merged_results_mc_summary.csv",
    "merged_curves.csv",
    "merged_curves_mc_summary.csv",
    "merged_weights_mc_summary.csv",
    "merged_params.csv",
    "merged_params.jsonl",
    "merged_params_mc_summary.jsonl",
)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_config(target_root: Path, *, n_mc: int, run_label: str) -> dict[str, str]:
    config = {
        "description": "Release-facing stage-05 discrimination data.",
        "run_label": str(run_label),
        "per_block": {
            "analysis": "per-block mixtures",
            "model_bank": list(PER_BLOCK_MODELS),
            "threshold_weights": "one shared threshold mixture per standard block",
        },
        "per_point": {
            "analysis": "per-point hybrid comparison",
            "model_bank": list(PER_POINT_MODELS),
            "threshold_weights": "CV-specific threshold mixtures within each frequency-by-block panel",
        },
        "fixed_pipeline_settings": {
            "bins_ms": [1.0, 10.0],
            "primary_bin_ms": 10.0,
            "aux_bin_ms": 1.0,
            "n_mc": int(n_mc),
            "train_frac": 0.8,
            "jnd_k": 2.5,
            "entmax_alpha": 1.1,
            "loss_reg_l2": 0.0001,
            "weight_kl": 0.01,
            "allow_zero_gains": False,
            "rate_gate_mode_hybrid": "delta_cv",
            "rate_gate_reg_hybrid": 0.002,
            "rate_gate_mode_rate_timing": "none",
            "weight_entropy_floor_k": 3.0,
            "weight_entropy_floor_lambda": 0.01,
            "weight_max_cap": 0.65,
            "weight_max_cap_lambda": 0.01,
            "weight_smooth_lambda_per_point": 0.001,
        },
    }
    path = target_root / "run_config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"target": display_path(path)}


def build_manifest(target_root: Path) -> dict[str, object]:
    manifest: dict[str, object] = {"stage05": {}, "slack": {}}
    for folder in ("per-block", "per-point"):
        folder_path = target_root / folder
        rows = []
        for name in EXPECTED_STAGE05_FILES:
            path = folder_path / name
            if not path.exists():
                raise FileNotFoundError(f"Missing expected discrimination output: {display_path(path)}")
            rows.append({"path": display_path(path), "bytes": int(path.stat().st_size)})
        manifest["stage05"][folder] = rows

    slack_dir = target_root / "slack"
    if not slack_dir.exists():
        raise FileNotFoundError(f"Missing Slack behavior directory: {display_path(slack_dir)}")
    slack_files = sorted(slack_dir.glob("*.csv"))
    if not slack_files:
        raise FileNotFoundError(f"No Slack behavior CSV files found in {display_path(slack_dir)}")
    manifest["slack"] = {
        "path": display_path(slack_dir),
        "files": len(slack_files),
    }
    return manifest


def finalize_metadata(target_root: Path, *, n_mc: int, run_label: str) -> dict[str, object]:
    target_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(target_root)
    manifest["config"] = write_config(target_root, n_mc=int(n_mc), run_label=str(run_label))
    manifest_path = target_root / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize release discrimination metadata.")
    parser.add_argument("--target", default=str(TARGET_ROOT), help="Target data/discrimination directory.")
    parser.add_argument("--n-mc", type=int, default=200, help="Monte Carlo count used by the regenerated run.")
    parser.add_argument("--run-label", default="release", help="Short label stored in run_config.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_root = Path(args.target).resolve()
    manifest = finalize_metadata(target_root, n_mc=int(args.n_mc), run_label=str(args.run_label))
    print(f"Finalized release discrimination data in {display_path(target_root)}")
    print(f"  per-block files: {len(manifest['stage05']['per-block'])}")
    print(f"  per-point files: {len(manifest['stage05']['per-point'])}")
    print(f"  Slack CSV files: {manifest['slack']['files']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
