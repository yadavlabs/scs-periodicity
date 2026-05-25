"""Command-line entrypoint for release discrimination fits and shard merging."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:  # package import when called as python -m discrimination.discrimination_CLI
    from .discrimination_core import (
        BLOCKS,
        FREQUENCIES,
        RELEASE_MODEL_BANKS,
        RELEASE_PIPELINE_SETTINGS,
        canonical_threshold_type,
        run_release_fit,
    )
except ImportError:  # direct script execution from the repository root
    from discrimination_core import (  # type: ignore
        BLOCKS,
        FREQUENCIES,
        RELEASE_MODEL_BANKS,
        RELEASE_PIPELINE_SETTINGS,
        canonical_threshold_type,
        run_release_fit,
    )


RELEASE_DROP_COLUMNS = {
    "partial_path",
    "scope",
    "scope_key",
    "loto_th",
    "drop_thresh",
    "is_loto",
    "fixed_thresh",
    "is_fixed_thresh",
    "base_run_id",
    "loss_reg_l2",
    "weight_kl",
    "allow_zero_gains",
    "rate_gate_reg",
    "rate_gate_prior_floor",
    "rate_gate_prior_center",
    "rate_gate_prior_slope",
    "shared_thresh_mode",
    "thresh_flex_mode",
    "weight_entropy_floor_k",
    "weight_entropy_floor_lambda",
    "weight_max_cap",
    "weight_max_cap_lambda",
    "weight_smooth_lambda",
}
RELEASE_PER_BLOCK_MODELS = {"rate_combined", "timing_combined", "hybrid_combined"}
RELEASE_PER_POINT_MODELS = {"hybrid_combined"}


def merge_discrimination_shards(args: argparse.Namespace) -> int:
    res_pattern = os.path.join(args.partial_root, "**", "results.csv")
    prm_pattern = os.path.join(args.partial_root, "**", "params.jsonl")
    curve_pattern = os.path.join(args.partial_root, "**", "curves.csv")
    res_files = sorted(glob.glob(res_pattern, recursive=True))
    prm_files = sorted(glob.glob(prm_pattern, recursive=True))
    curve_files = sorted(glob.glob(curve_pattern, recursive=True))
    if not res_files and not prm_files and not curve_files:
        raise SystemExit(f"No results.csv, params.jsonl, or curves.csv found under: {args.partial_root}")

    os.makedirs(args.out_root, exist_ok=True)

    # Merge results.csv shards
    if res_files:
        dfs = []
        for f in res_files:
            try:
                df = pd.read_csv(f)
                df["partial_path"] = os.path.relpath(os.path.dirname(f), args.partial_root)
                dfs.append(df)
            except Exception as e:
                print(f"[merge] skipping results {f}: {e}", flush=True)

        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(args.out_root, "merged_results.csv")
            out.to_csv(out_path, index=False)
            print(f"[merge] wrote: {out_path} (rows={len(out)})", flush=True)
            _merge_write_mc_summaries_from_results(out, args.out_root, prefix="merged_")
        else:
            print("[merge] no readable results.csv shards", flush=True)

    # Merge curves.csv shards
    if curve_files:
        dfs = []
        for f in curve_files:
            try:
                df = pd.read_csv(f)
                df["partial_path"] = os.path.relpath(os.path.dirname(f), args.partial_root)
                dfs.append(df)
            except Exception as e:
                print(f"[merge] skipping curves {f}: {e}", flush=True)
        if dfs:
            out = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(args.out_root, "merged_curves.csv")
            out.to_csv(out_path, index=False)
            print(f"[merge] wrote: {out_path} (rows={len(out)})", flush=True)
            _merge_write_curve_summaries_from_curves(out, args.out_root, prefix="merged_")
        else:
            print("[merge] no readable curves.csv shards", flush=True)

    # Merge params.jsonl shards
    if prm_files:
        out_jsonl = os.path.join(args.out_root, "merged_params.jsonl")
        n_rows = 0
        with open(out_jsonl, "w", encoding="utf-8") as fout:
            for f in prm_files:
                rel_dir = os.path.relpath(os.path.dirname(f), args.partial_root)
                try:
                    with open(f, "r", encoding="utf-8") as fin:
                        for ln in fin:
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                row = json.loads(ln)
                            except Exception:
                                # keep malformed shards from killing the full merge
                                continue
                            if isinstance(row, dict):
                                row["partial_path"] = rel_dir
                                fout.write(json.dumps(row) + "\n")
                                n_rows += 1
                except Exception as e:
                    print(f"[merge] skipping params {f}: {e}", flush=True)
        print(f"[merge] wrote: {out_jsonl} (rows={n_rows})", flush=True)

        # Convenience CSV mirror for quick notebook loading
        try:
            prm_df = pd.read_json(out_jsonl, lines=True)
            out_csv = os.path.join(args.out_root, "merged_params.csv")
            prm_df.to_csv(out_csv, index=False)
            print(f"[merge] wrote: {out_csv} (rows={len(prm_df)})", flush=True)
        except Exception as e:
            print(f"[merge] params csv export failed: {e}", flush=True)

    if args.release_kind:
        _merge_compact_release_outputs(args.out_root, args.release_kind)

    return 0


def _merge_sem_series(x: pd.Series) -> float:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    xv = xv[np.isfinite(xv)]
    if xv.size <= 1:
        return 0.0
    return float(np.std(xv, ddof=1) / np.sqrt(xv.size))


def _merge_parse_weight_dict_cell(x) -> dict:
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            try:
                fv = float(v)
            except Exception:
                continue
            if np.isfinite(fv):
                out[canonical_threshold_type(k)] = fv
        return out
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return {}
    s = str(x).strip()
    if not s or s.lower() in {"none", "null", "nan", "{}"}:
        return {}
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}
    for k, v in obj.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out[canonical_threshold_type(k)] = fv
    return out


def _merge_truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _merge_is_none_text(value) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "nan", "null"} or text == "NONE"


def _merge_release_keep_row(row: dict, release_kind: str) -> bool:
    get = row.get
    model_bank = str(get("model_bank", ""))
    scope = str(get("scope", ""))
    scope_key = str(get("scope_key", ""))
    drop_thresh = str(get("drop_thresh", "NONE"))
    fixed_thresh = get("fixed_thresh", "")
    thresh_flex_mode = str(get("thresh_flex_mode", ""))

    if release_kind == "per_block":
        if model_bank and model_bank not in RELEASE_PER_BLOCK_MODELS:
            return False
        if scope and scope != "per_block":
            return False
        if not scope and scope_key and scope_key not in {"block=std0", "block=std1"}:
            return False
        if drop_thresh != "NONE":
            return False
        if "fixed_thresh" in row and not _merge_is_none_text(fixed_thresh):
            return False
        if _merge_truthy(get("is_loto", False)) or _merge_truthy(get("is_fixed_thresh", False)):
            return False
        if thresh_flex_mode and thresh_flex_mode not in {"shared", "nan", "None"}:
            return False
        return True

    if release_kind == "per_point":
        if model_bank and model_bank not in RELEASE_PER_POINT_MODELS:
            return False
        if scope and scope != "per_freq_block":
            return False
        if not scope and scope_key and not scope_key.startswith("freq="):
            return False
        if drop_thresh != "NONE":
            return False
        if _merge_truthy(get("is_loto", False)) or _merge_truthy(get("is_fixed_thresh", False)):
            return False
        if thresh_flex_mode and thresh_flex_mode != "per_point":
            return False
        return True

    raise ValueError(f"Unknown release_kind: {release_kind}")


def _merge_compact_release_csv(path: str, release_kind: str) -> None:
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        out = df.loc[:, [c for c in df.columns if c not in RELEASE_DROP_COLUMNS]].copy()
    else:
        keep = df.apply(lambda row: _merge_release_keep_row(row.to_dict(), release_kind), axis=1)
        out = df.loc[keep, [c for c in df.columns if c not in RELEASE_DROP_COLUMNS]].reset_index(drop=True)
    out.to_csv(path, index=False)
    print(f"[merge] compacted release csv: {path} (rows={len(out)})", flush=True)


def _merge_compact_release_jsonl(path: str, release_kind: str) -> None:
    if not os.path.exists(path):
        return
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and _merge_release_keep_row(row, release_kind):
                rows.append({key: value for key, value in row.items() if key not in RELEASE_DROP_COLUMNS})
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[merge] compacted release jsonl: {path} (rows={len(rows)})", flush=True)


def _merge_compact_release_outputs(out_root: str, release_kind: str) -> None:
    for name in [
        "merged_results.csv",
        "merged_results_mc_summary.csv",
        "merged_curves.csv",
        "merged_curves_mc_summary.csv",
        "merged_weights_mc_summary.csv",
        "merged_params.csv",
    ]:
        _merge_compact_release_csv(os.path.join(out_root, name), release_kind)
    for name in ["merged_params.jsonl", "merged_params_mc_summary.jsonl"]:
        _merge_compact_release_jsonl(os.path.join(out_root, name), release_kind)


def _merge_json_scalar(x):
    if isinstance(x, np.generic):
        return x.item()
    return x


def _merge_write_mc_summaries_from_results(df_res: pd.DataFrame, out_root: str, prefix: str = "") -> None:
    if df_res is None or df_res.empty:
        return
    os.makedirs(out_root, exist_ok=True)

    key_cols = [
        c
        for c in [
            "scope",
            "scope_key",
            "model_bank",
            "freq",
            "block",
            "bin_ms",
            "aux_bin_ms",
            "drop_thresh",
            "is_loto",
            "fixed_thresh",
            "is_fixed_thresh",
            "n_thresh",
            "loss_reg_l2",
            "weight_kl",
            "allow_zero_gains",
            "rate_gate_mode",
            "rate_gate_reg",
            "rate_gate_prior_floor",
            "rate_gate_prior_center",
            "rate_gate_prior_slope",
            "shared_thresh_mode",
            "thresh_flex_mode",
            "weight_entropy_floor_k",
            "weight_entropy_floor_lambda",
            "weight_max_cap",
            "weight_max_cap_lambda",
            "weight_smooth_lambda",
        ]
        if c in df_res.columns
    ]
    if not key_cols:
        return

    work = df_res.copy()
    for c in [
        "rmse_train",
        "mae_train",
        "nll_train",
        "rmse_test",
        "mae_test",
        "gain_rate",
        "gain_timing",
        "bias",
        "temp",
        "gain_rate_aux",
        "gain_timing_aux",
        "rate_gate_floor",
        "rate_gate_center",
        "rate_gate_slope",
        "mc",
    ]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    agg_spec = {}
    for c in [
        "rmse_train",
        "mae_train",
        "nll_train",
        "rmse_test",
        "mae_test",
        "gain_rate",
        "gain_timing",
        "bias",
        "temp",
        "gain_rate_aux",
        "gain_timing_aux",
        "rate_gate_floor",
        "rate_gate_center",
        "rate_gate_slope",
    ]:
        if c in work.columns:
            agg_spec[f"{c}_mean"] = (c, "mean")
            agg_spec[f"{c}_sem"] = (c, _merge_sem_series)

    summary = work.groupby(key_cols, as_index=False, dropna=False).agg(
        n_rows=("mc", "size"),
        n_mc=("mc", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique()) if len(s) else 0),
        **agg_spec,
    )
    p_sum = os.path.join(out_root, f"{prefix}results_mc_summary.csv")
    summary.to_csv(p_sum, index=False)
    print(f"[merge] wrote: {p_sum} (rows={len(summary)})", flush=True)

    weight_records = []
    for _, row in work.iterrows():
        mb = str(row.get("model_bank", "")).strip().lower()
        w = _merge_parse_weight_dict_cell(row.get("weights_json"))
        if not w:
            if mb == "rate_combined":
                w = _merge_parse_weight_dict_cell(row.get("weights_rate_json"))
            elif mb == "timing_combined":
                w = _merge_parse_weight_dict_cell(row.get("weights_timing_json"))
            else:
                w = _merge_parse_weight_dict_cell(row.get("weights_shared_json"))
        if not w:
            continue
        rec_base = {k: row[k] for k in key_cols if k in row.index}
        rec_base["mc"] = row.get("mc", np.nan)
        for th, wt in w.items():
            rec = dict(rec_base)
            rec["thresh_type"] = str(th)
            rec["weight"] = float(wt)
            weight_records.append(rec)

    if not weight_records:
        return

    wdf = pd.DataFrame(weight_records)
    wsum = wdf.groupby(key_cols + ["thresh_type"], as_index=False, dropna=False).agg(
        weight_mean=("weight", "mean"),
        weight_sem=("weight", _merge_sem_series),
        n_rows=("weight", "size"),
        n_mc=("mc", lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique()) if len(s) else 0),
    )
    p_w = os.path.join(out_root, f"{prefix}weights_mc_summary.csv")
    wsum.to_csv(p_w, index=False)
    print(f"[merge] wrote: {p_w} (rows={len(wsum)})", flush=True)

    jrec = []
    for gk, g in wsum.groupby(key_cols, dropna=False):
        if not isinstance(gk, tuple):
            gk = (gk,)
        d = {k: _merge_json_scalar(v) for k, v in zip(key_cols, gk)}
        d["n_mc"] = int(pd.to_numeric(g["n_mc"], errors="coerce").max()) if not g.empty else 0
        d["weights_mean"] = {str(r["thresh_type"]): float(r["weight_mean"]) for _, r in g.iterrows()}
        d["weights_sem"] = {str(r["thresh_type"]): float(r["weight_sem"]) for _, r in g.iterrows()}
        gg = summary
        for k in key_cols:
            if pd.isna(d[k]):
                gg = gg[gg[k].isna()]
            else:
                gg = gg[gg[k] == d[k]]
        if not gg.empty:
            row_s = gg.iloc[0]
            param_fields = [
                "gain_rate",
                "gain_timing",
                "bias",
                "temp",
                "gain_rate_aux",
                "gain_timing_aux",
                "rate_gate_floor",
                "rate_gate_center",
                "rate_gate_slope",
            ]
            d["params_mean"] = {
                k: float(row_s.get(f"{k}_mean", np.nan))
                for k in param_fields
                if f"{k}_mean" in row_s.index
            }
            d["params_sem"] = {
                k: float(row_s.get(f"{k}_sem", np.nan))
                for k in param_fields
                if f"{k}_sem" in row_s.index
            }
        jrec.append(d)

    p_j = os.path.join(out_root, f"{prefix}params_mc_summary.jsonl")
    with open(p_j, "w", encoding="utf-8") as f:
        for r in jrec:
            f.write(json.dumps(r) + "\n")
    print(f"[merge] wrote: {p_j} (rows={len(jrec)})", flush=True)


def _merge_parse_json_float_list(x) -> list:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return [float(v) for v in arr.tolist()]
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"none", "null", "nan", "[]"}:
        return []
    try:
        obj = json.loads(s)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out = []
    for v in obj:
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(float(fv))
    return out


def _merge_write_curve_summaries_from_curves(df_curves: pd.DataFrame, out_root: str, prefix: str = "") -> None:
    if df_curves is None or df_curves.empty:
        return
    key_cols = [
        c
        for c in [
            "scope",
            "scope_key",
            "model_bank",
            "freq",
            "block",
            "bin_ms",
            "aux_bin_ms",
            "drop_thresh",
            "is_loto",
            "fixed_thresh",
            "is_fixed_thresh",
            "loss_reg_l2",
            "weight_kl",
            "allow_zero_gains",
            "rate_gate_mode",
            "rate_gate_reg",
            "rate_gate_prior_floor",
            "rate_gate_prior_center",
            "rate_gate_prior_slope",
            "shared_thresh_mode",
            "thresh_flex_mode",
            "weight_entropy_floor_k",
            "weight_entropy_floor_lambda",
            "weight_max_cap",
            "weight_max_cap_lambda",
            "weight_smooth_lambda",
        ]
        if c in df_curves.columns
    ]
    if not key_cols:
        return
    metric_fields = [
        ("p_obs_json", "p_obs"),
        ("p_pred_train_json", "p_pred_train"),
        ("p_pred_test_json", "p_pred_test"),
        ("rate_drive_train_json", "rate_drive_train"),
        ("timing_drive_train_json", "timing_drive_train"),
        ("rate_drive_test_json", "rate_drive_test"),
        ("timing_drive_test_json", "timing_drive_test"),
        ("rate_drive_aux_train_json", "rate_drive_aux_train"),
        ("timing_drive_aux_train_json", "timing_drive_aux_train"),
        ("rate_drive_aux_test_json", "rate_drive_aux_test"),
        ("timing_drive_aux_test_json", "timing_drive_aux_test"),
        ("gate_train_json", "gate_train"),
        ("gate_test_json", "gate_test"),
    ]
    rows = []
    for _, row in df_curves.iterrows():
        cv_vals = _merge_parse_json_float_list(row.get("cv_json"))
        if not cv_vals:
            continue
        payload = {}
        for col, name in metric_fields:
            vals = _merge_parse_json_float_list(row.get(col))
            if len(vals) == len(cv_vals):
                payload[name] = vals
        if not payload:
            continue
        base = {k: row[k] for k in key_cols if k in row.index}
        for i, cv in enumerate(cv_vals):
            rec = dict(base)
            rec["cv"] = float(cv)
            for name, vals in payload.items():
                rec[name] = float(vals[i])
            rows.append(rec)
    if not rows:
        return
    long_df = pd.DataFrame(rows)
    agg_map = {}
    for c in [
        "p_obs",
        "p_pred_train",
        "p_pred_test",
        "rate_drive_train",
        "timing_drive_train",
        "rate_drive_test",
        "timing_drive_test",
        "rate_drive_aux_train",
        "timing_drive_aux_train",
        "rate_drive_aux_test",
        "timing_drive_aux_test",
        "gate_train",
        "gate_test",
    ]:
        if c in long_df.columns:
            agg_map[f"{c}_mean"] = (c, "mean")
            agg_map[f"{c}_sem"] = (c, _merge_sem_series)
    if not agg_map:
        return
    out = long_df.groupby(key_cols + ["cv"], as_index=False, dropna=False).agg(
        n_rows=("cv", "size"),
        **agg_map,
    )
    p_out = os.path.join(out_root, f"{prefix}curves_mc_summary.csv")
    out.to_csv(p_out, index=False)
    print(f"[merge] wrote: {p_out} (rows={len(out)})", flush=True)



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run release-facing periodicity discrimination fits and merge SLURM shards."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="Run one release discrimination fit shard.")
    fit.add_argument("--run_kind", required=True, choices=["per_block", "per_point"])
    fit.add_argument("--sim_csv", required=True)
    fit.add_argument("--slack_dir", required=True)
    fit.add_argument("--out_dir", required=True)
    fit.add_argument("--scope_key", required=True)
    fit.add_argument("--model_bank", required=True, choices=list(RELEASE_MODEL_BANKS))
    fit.add_argument("--n_mc", type=int, default=int(RELEASE_PIPELINE_SETTINGS["n_mc"]))
    fit.add_argument("--mc_start", type=int, default=0)
    fit.add_argument("--mc_count", type=int, default=None)
    fit.add_argument("--train_frac", type=float, default=float(RELEASE_PIPELINE_SETTINGS["train_frac"]))
    fit.add_argument("--seed", type=int, default=int(RELEASE_PIPELINE_SETTINGS["seed"]))
    fit.set_defaults(func=_main_fit)

    merge = sub.add_parser("merge", help="Merge partial results, params, and curves shards.")
    merge.add_argument("--partial-root", required=True)
    merge.add_argument("--out-root", required=True)
    merge.add_argument(
        "--release-kind",
        choices=["per_block", "per_point"],
        default=None,
        help="Compact merged outputs for the release data lane.",
    )
    merge.set_defaults(func=merge_discrimination_shards)
    return parser


def _main_fit(args: argparse.Namespace) -> int:
    df, _ = run_release_fit(
        run_kind=args.run_kind,
        sim_csv=args.sim_csv,
        slack_dir=args.slack_dir,
        out_dir=args.out_dir,
        scope_key=args.scope_key,
        model_bank=args.model_bank,
        n_mc=args.n_mc,
        mc_start=args.mc_start,
        mc_count=args.mc_count,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(df.head(20).to_string(index=False))
    print(f"Wrote: {Path(args.out_dir) / 'results.csv'}")
    print(f"Wrote: {Path(args.out_dir) / 'params.jsonl'}")
    print(f"Wrote: {Path(args.out_dir) / 'curves.csv'}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    return int(args.func(args))



__all__ = ["build_argparser", "main", "merge_discrimination_shards"]


if __name__ == "__main__":
    raise SystemExit(main())
