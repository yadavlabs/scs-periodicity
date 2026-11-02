"""CLI and shard merger for the population discrimination observer."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _sem(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return 0.0 if x.size <= 1 else float(np.std(x, ddof=1) / np.sqrt(x.size))


def _merge_csv(partial_root: Path, out_root: Path, stem: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(partial_root / "**" / f"{stem}.csv"), recursive=True))
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
            frame["partial_path"] = os.path.relpath(os.path.dirname(path), partial_root)
            frames.append(frame)
        except Exception as exc:
            print(f"[merge] skipping {path}: {exc}", flush=True)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    out = out_root / f"merged_{stem}.csv"
    merged.to_csv(out, index=False)
    print(f"[merge] wrote {out} (rows={len(merged)})", flush=True)
    return merged


def _write_summary(df: pd.DataFrame, path: Path, extra_keys: list[str]) -> None:
    if df.empty or "mc" not in df:
        return
    base_keys = [
        "weight_scope", "readout_scope", "state_mode", "model_family",
        "rate_gate_mode", "mixture_semantics", "split", "rate_gate_reg",
        "rate_gate_prior_floor", "rate_gate_prior_center", "rate_gate_prior_slope",
        "loss_reg_l2", "weight_kl", "weight_entropy_floor_k",
        "weight_entropy_floor_lambda", "weight_max_cap", "weight_max_cap_lambda",
        "n_parameters", "n_weight_parameters", "n_readout_parameters",
    ]
    keys = [c for c in base_keys + extra_keys if c in df.columns]
    numeric = [
        c for c in df.columns
        if c not in keys + ["mc", "partial_path"]
        and pd.api.types.is_numeric_dtype(df[c])
        and c not in {"rep_index"}
    ]
    if not keys or not numeric:
        return
    agg = {f"{c}_mean": (c, "mean") for c in numeric}
    agg.update({f"{c}_sem": (c, _sem) for c in numeric})
    summary = df.groupby(keys, as_index=False, dropna=False).agg(
        n_rows=("mc", "size"),
        n_mc=("mc", lambda s: int(pd.to_numeric(s, errors="coerce").nunique())),
        **agg,
    )
    summary.to_csv(path, index=False)
    print(f"[merge] wrote {path} (rows={len(summary)})", flush=True)


def merge_outputs(partial_root: str, out_root: str) -> int:
    src = Path(partial_root)
    dst = Path(out_root)
    dst.mkdir(parents=True, exist_ok=True)
    results = _merge_csv(src, dst, "results")
    curves = _merge_csv(src, dst, "curves")
    weights = _merge_csv(src, dst, "weights")
    effective_weights = _merge_csv(src, dst, "effective_weights")
    _merge_csv(src, dst, "split_membership")
    if results.empty and curves.empty and weights.empty and effective_weights.empty:
        raise SystemExit(f"No population observer shards found under {src}")
    _write_summary(results, dst / "merged_results_mc_summary.csv", ["freq", "block"])
    _write_summary(curves, dst / "merged_curves_mc_summary.csv", ["freq", "block", "cv"])
    _write_summary(
        weights,
        dst / "merged_weights_mc_summary.csv",
        [
            "freq", "block", "thresh_type", "canonical_thresh_type",
            "equivalent_regimes", "regime_active",
        ],
    )
    _write_summary(
        effective_weights,
        dst / "merged_effective_weights_mc_summary.csv",
        ["freq", "block", "canonical_thresh_type", "equivalent_regimes"],
    )

    # Stable, manuscript-facing aliases used by the population notebooks.
    result_summary = dst / "merged_results_mc_summary.csv"
    curve_summary = dst / "merged_curves_mc_summary.csv"
    weight_summary = dst / "merged_weights_mc_summary.csv"
    if result_summary.exists():
        pd.read_csv(result_summary).to_csv(dst / "population_observer_rmse.csv", index=False)
    if curve_summary.exists():
        curve_df = pd.read_csv(curve_summary)
        curve_df.to_csv(dst / "population_discrimination_curves.csv", index=False)
        share_cols = [
            c for c in curve_df.columns
            if c in {
                "weight_scope", "readout_scope", "state_mode", "model_family",
                "rate_gate_mode", "mixture_semantics", "split", "freq", "block", "cv",
                "loss_reg_l2", "weight_kl", "weight_entropy_floor_k",
                "weight_entropy_floor_lambda", "weight_max_cap", "weight_max_cap_lambda",
                "n_parameters", "n_weight_parameters", "n_readout_parameters",
                "rate_gate_reg", "rate_gate_prior_floor", "rate_gate_prior_center",
                "rate_gate_prior_slope",
                "effective_rate_share_mean", "effective_rate_share_sem", "n_mc",
            }
        ]
        curve_df[share_cols].to_csv(dst / "population_effective_rate_share.csv", index=False)
    if weight_summary.exists():
        pd.read_csv(weight_summary).to_csv(dst / "population_observer_weights.csv", index=False)
    effective_weight_summary = dst / "merged_effective_weights_mc_summary.csv"
    if effective_weight_summary.exists():
        pd.read_csv(effective_weight_summary).to_csv(
            dst / "population_observer_effective_weights.csv", index=False
        )

    jsonl_files = sorted(glob.glob(str(src / "**" / "params.jsonl"), recursive=True))
    if jsonl_files:
        out_jsonl = dst / "merged_params.jsonl"
        count = 0
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for path in jsonl_files:
                rel = os.path.relpath(os.path.dirname(path), src)
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        row["partial_path"] = rel
                        fout.write(json.dumps(row) + "\n")
                        count += 1
        print(f"[merge] wrote {out_jsonl} (rows={count})", flush=True)
        try:
            params = pd.read_json(out_jsonl, lines=True)
            params.to_csv(dst / "merged_params.csv", index=False)
            params.to_csv(dst / "population_observer_readout.csv", index=False)
        except Exception as exc:
            print(f"[merge] parameter CSV export failed: {exc}", flush=True)
    return 0


def validate_outputs(out_root: str, expected_mc: int, expected_configurations: int) -> int:
    root = Path(out_root)
    results = pd.read_csv(root / "merged_results.csv")
    weights = pd.read_csv(root / "merged_weights.csv")
    effective_weight_path = root / "merged_effective_weights.csv"
    effective_weights = (
        pd.read_csv(effective_weight_path) if effective_weight_path.is_file() else pd.DataFrame()
    )
    curves = pd.read_csv(root / "merged_curves.csv")
    config = [
        "weight_scope", "readout_scope", "state_mode", "model_family",
        "rate_gate_mode", "mixture_semantics",
    ]
    regularization = [
        "loss_reg_l2", "weight_kl", "weight_entropy_floor_k",
        "weight_entropy_floor_lambda", "weight_max_cap", "weight_max_cap_lambda",
        "rate_gate_reg", "rate_gate_prior_floor", "rate_gate_prior_center",
        "rate_gate_prior_slope",
    ]
    model_dimensions = [
        "n_parameters", "n_weight_parameters", "n_readout_parameters",
    ]
    config.extend(
        column for column in regularization + model_dimensions if column in results.columns
    )
    n_configs = int(results[config].drop_duplicates().shape[0])
    errors: list[str] = []
    if n_configs != int(expected_configurations):
        errors.append(f"expected {expected_configurations} configurations, found {n_configs}")
    counts = results.groupby(config + ["split", "freq", "block"], dropna=False)["mc"].nunique()
    bad_counts = counts[counts != int(expected_mc)]
    if not bad_counts.empty:
        errors.append(f"{len(bad_counts)} result groups do not contain {expected_mc} MC splits")
    weight_sums = weights.groupby(config + ["mc", "weight_key"], dropna=False)["weight"].sum()
    bad_weights = weight_sums[~np.isclose(weight_sums.to_numpy(float), 1.0, atol=1e-7)]
    if not bad_weights.empty:
        errors.append(f"{len(bad_weights)} mixture vectors do not sum to one")
    if not effective_weights.empty:
        effective_sums = effective_weights.groupby(
            config + ["mc", "weight_key", "freq", "block"], dropna=False
        )["weight"].sum()
        bad_effective = effective_sums[
            ~np.isclose(effective_sums.to_numpy(float), 1.0, atol=1e-7)
        ]
        if not bad_effective.empty:
            errors.append(
                f"{len(bad_effective)} frequency-specific collapsed mixture vectors "
                "do not sum to one"
            )
    curve_keys = config + ["mc", "split", "freq", "block", "cv"]
    duplicate_curves = int(curves.duplicated(curve_keys).sum())
    if duplicate_curves:
        errors.append(f"{duplicate_curves} duplicate curve rows")
    # Count each fitted configuration/MC split once.  A merged run is not
    # considered ready when even one optimizer reports formal failure.
    fit_rows = results[
        results["freq"].astype(str).eq("ALL")
        & results["block"].astype(str).eq("ALL")
        & results["split"].astype(str).eq("train")
    ].drop_duplicates(config + ["mc"])
    success = fit_rows["optimization_success"].astype(str).str.strip().str.lower().eq("true")
    failed_fits = int((~success).sum())
    if failed_fits:
        errors.append(f"{failed_fits} of {len(fit_rows)} observer fits did not converge formally")
    success_rate = float(success.mean()) if len(success) else 0.0
    assignments = {"rate_timing": {"std0": "rate", "std1": "timing"},
                   "timing_rate": {"std0": "timing", "std1": "rate"}}
    followups = set(results.model_family) & set(assignments)
    if followups:
        params = pd.read_csv(root / "merged_params.csv")
        membership = pd.read_csv(root / "merged_split_membership.csv")
        for family in followups:
            f = curves[curves.model_family == family]
            expected_keys = {(mc, split, freq, block, round(cv, 1))
                             for mc in range(expected_mc) for split in ("train", "test")
                             for freq in (20., 50., 100., 200.) for block in ("std0", "std1")
                             for cv in np.linspace(0, 1, 11)}
            keys = set(zip(f.mc, f.split, f.freq.astype(float), f.block, f.cv.round(1)))
            if keys != expected_keys or len(f) != len(expected_keys):
                errors.append(f"{family}: incomplete or duplicate frequency/block/CV/MC coverage")
            r = results[results.model_family == family]
            w = weights[weights.model_family == family]
            if not w.block.eq("ALL").all() or w.groupby("mc").weight_key.nunique().ne(4).any():
                errors.append(f"{family}: mixtures are not shared per frequency across blocks")
            if not ((r.n_parameters == 29) & (r.n_weight_parameters == 23) & (r.n_readout_parameters == 6)).all():
                errors.append(f"{family}: expected 23 mixture + 6 evidence parameters")
            for block, cue in assignments[family].items():
                p = params[(params.model_family == family) & (params.block == block)]
                inactive = "timing" if cue == "rate" else "rate"
                numeric = p[["bias", "gain_rate_1ms", "gain_rate_10ms", "gain_timing_1ms", "gain_timing_10ms"]]
                if not np.isfinite(numeric.to_numpy()).all():
                    errors.append(f"{family}/{block}: nonfinite evidence parameters")
                if len(p) != expected_mc or p.mc.nunique() != expected_mc or not np.all(
                    p[[f"gain_{inactive}_1ms", f"gain_{inactive}_10ms"]].to_numpy() == 0
                ):
                    errors.append(f"{family}/{block}: invalid inactive-channel gains or parameter coverage")
                if not np.all(f.loc[f.block == block, f"{inactive}_drive"].to_numpy() == 0):
                    errors.append(f"{family}/{block}: inactive cue affected predictions")
        m = membership[membership.model_family.isin(followups)]
        if (len(m) != expected_mc * len(followups) or m.duplicated(["model_family", "mc"]).any()
                or set(m.mc) != set(range(expected_mc))):
            errors.append("Invalid block-cue split membership coverage")
        for _, group in m.groupby("mc"):
            if group.train_reps.nunique() != 1 or group.test_reps.nunique() != 1:
                errors.append("Opposing assignments did not use matched simulation splits")
            for row in group.itertuples():
                train_reps, test_reps = set(json.loads(row.train_reps)), set(json.loads(row.test_reps))
                if (len(train_reps) != 80 or len(test_reps) != 20 or train_reps & test_reps
                        or train_reps | test_reps != set(range(100))):
                    errors.append("Expected disjoint 80/20 simulation splits")
        provenance = root / "followup_inputs.json"
        if provenance.exists():
            from population.discrimination.block_cue_followup import verify
            try:
                verify(provenance, root=Path.cwd())
            except (ValueError, FileNotFoundError) as exc:
                errors.append(str(exc))
    for name, frame, columns in (
        ("results", results, [
            "objective_train", "data_nll_train", "rmse", "mae", "nll_per_trial",
            "readout_l2_penalty", "weight_kl_penalty",
            "weight_entropy_floor_penalty", "weight_max_cap_penalty", "rate_gate_penalty",
        ]),
        ("curves", curves, ["p_predicted", "rate_drive", "timing_drive", "gate", "effective_rate_share"]),
        ("weights", weights, [
            "weight", "weight_entropy", "effective_regime_count",
            "effective_regime_target", "max_regime_weight",
        ]),
        ("effective_weights", effective_weights, [
            "weight", "weight_entropy", "effective_regime_count",
            "effective_regime_target", "max_regime_weight",
        ]),
    ):
        present = [column for column in columns if column in frame]
        if present and not np.isfinite(frame[present].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all():
            errors.append(f"{name} contain non-finite numeric outputs")
    diversity: dict[str, float | int | None] = {
        "mixture_vectors": 0,
        "effective_regime_count_min": None,
        "effective_regime_count_mean": None,
        "fraction_effective_k_ge_target": None,
        "max_regime_weight_max": None,
        "fraction_max_weight_le_cap": None,
    }
    diversity_source = effective_weights if not effective_weights.empty else weights
    if {"effective_regime_count", "max_regime_weight"}.issubset(diversity_source.columns):
        diversity_keys = config + ["mc", "weight_key"]
        diversity_keys.extend(
            column for column in ("freq", "block")
            if column in diversity_source.columns
        )
        mixture = diversity_source[
            diversity_source["state_mode"].astype(str).eq("mixture")
        ].drop_duplicates(
            diversity_keys
        )
        if len(mixture):
            k_eff = pd.to_numeric(mixture["effective_regime_count"], errors="coerce")
            max_weight = pd.to_numeric(mixture["max_regime_weight"], errors="coerce")
            target_column = (
                "effective_regime_target"
                if "effective_regime_target" in mixture.columns
                else "weight_entropy_floor_k"
            )
            k_target = pd.to_numeric(mixture[target_column], errors="coerce")
            weight_cap = pd.to_numeric(mixture["weight_max_cap"], errors="coerce")
            diversity = {
                "mixture_vectors": int(len(mixture)),
                "effective_regime_count_min": float(k_eff.min()),
                "effective_regime_count_mean": float(k_eff.mean()),
                "fraction_effective_k_ge_target": float((k_eff >= k_target - 1e-6).mean()),
                "max_regime_weight_max": float(max_weight.max()),
                "fraction_max_weight_le_cap": float((max_weight <= weight_cap + 1e-6).mean()),
            }
    report = {
        "status": "failed" if errors else "ready",
        "expected_mc": int(expected_mc),
        "configurations": n_configs,
        "result_rows": int(len(results)),
        "curve_rows": int(len(curves)),
        "weight_rows": int(len(weights)),
        "effective_weight_rows": int(len(effective_weights)),
        "observer_fits": int(len(fit_rows)),
        "failed_observer_fits": failed_fits,
        "optimization_success_fraction": success_rate,
        "mixture_diversity": diversity,
        "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit("Population observer validation failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--partial-root", required=True)
    merge.add_argument("--out-root", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--out-root", required=True)
    validate.add_argument("--expected-mc", required=True, type=int)
    validate.add_argument("--expected-configurations", required=True, type=int)
    args = parser.parse_args()
    if args.command == "merge":
        return merge_outputs(args.partial_root, args.out_root)
    if args.command == "validate":
        return validate_outputs(args.out_root, args.expected_mc, args.expected_configurations)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
