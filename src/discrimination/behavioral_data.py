"""Slack behavioral parser retained from the original analysis for population use."""
from __future__ import annotations
import os
import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd

def parse_rat_name_from_filename(path: str) -> str:
    """Slack files: '23-Dec-2022_Olaf.csv' or '25-Aug-2022_Mars_a.csv'."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return parts[1]


def infer_block_from_file(df_file: pd.DataFrame) -> Optional[str]:
    """Infer std0 vs std1 for a Slack behavioral file.

    We look at mid CV trials (0<CV<1) and see whether Type=1 or Type=2 dominates.
      - if Type=1 dominates: mid CVs correspond to Type=1 => std0
      - if Type=2 dominates: mid CVs correspond to Type=2 => std1
    """
    if "CV" not in df_file.columns or "Type" not in df_file.columns:
        return None
    mid = df_file[(df_file["CV"] > 0.0) & (df_file["CV"] < 1.0)].copy()
    if mid.empty:
        return None
    vc = mid["Type"].value_counts()
    if vc.empty:
        return None
    top = int(vc.idxmax())
    return "std0" if top == 1 else "std1"


def load_slack_dir(
    slack_dir: str,
    *,
    include_templates: bool = True,
    keep_only_varying_type: bool = True,
    aggregate_sessions_by_rat: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all Slack behavioral files in a directory.

    Returns
    -------
    slack_perrat : per-rat psychometrics
      columns: rat_name,freq,block,cv,k,n,p_correct
    slack_mean : mean psychometrics across rats
      columns: freq,block,cv,k,n,p_mean,p_sem,n_trials

    Parameters
    ----------
    include_templates:
      If True, keep CV==0 and CV==1 template trials in slack_perrat.
      You can later exclude them from evaluation if you want.
    keep_only_varying_type:
      Matches older convention:
        - std0 keeps Type==1 (varying = aperiodic)
        - std1 keeps Type==2 (varying = periodic)
      If False, keeps both.
    aggregate_sessions_by_rat:
      If True (default), first sum k/n across all files belonging to the same
      rat at each (freq, block, cv), then compute p_correct and rat-level means.
      If False, each file contributes separately (session-weighted behavior).
    """
    slack_dir = str(slack_dir)
    files = [os.path.join(slack_dir, f) for f in os.listdir(slack_dir) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No Slack csv files found in slack_dir: {slack_dir}")

    rows = []
    for fp in sorted(files):
        df = pd.read_csv(fp, low_memory=False)
        rat = parse_rat_name_from_filename(fp)

        # Filter to randomized, non-forced trials with valid responses
        if "Randomized" in df.columns:
            df = df[df["Randomized"] == 1]
        if "Forced" in df.columns:
            df = df[df["Forced"] == 0]
        if "Response" in df.columns:
            df = df[df["Response"].isin([1, 2])]

        block = infer_block_from_file(df)
        if block is None:
            continue

        # Standardize columns
        if "CV" in df.columns:
            df["cv"] = pd.to_numeric(df["CV"], errors="coerce")
        elif "cv" in df.columns:
            df["cv"] = pd.to_numeric(df["cv"], errors="coerce")
        else:
            continue

        if "Freq" in df.columns:
            df["freq"] = pd.to_numeric(df["Freq"], errors="coerce")
        elif "Frequency (Hz)" in df.columns:
            df["freq"] = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
        elif "freq" in df.columns:
            df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
        else:
            continue

        if "Type" in df.columns:
            df["Type"] = pd.to_numeric(df["Type"], errors="coerce")
        else:
            continue

        if "Response" not in df.columns:
            continue
        df["Response"] = pd.to_numeric(df["Response"], errors="coerce")

        df = df.dropna(subset=["cv", "freq", "Type", "Response"])
        if df.empty:
            continue

        # Optional filtering to match the displayed psychometrics
        if not include_templates:
            if block == "std0":
                df = df[df["cv"] > 0.0]
            else:
                df = df[df["cv"] < 1.0]

        if keep_only_varying_type:
            if block == "std0":
                is_template = np.isclose(df["cv"], 0.0)
                df = df[is_template | (df["Type"].astype(int) == 1)]
            else:
                is_template = np.isclose(df["cv"], 1.0)
                df = df[is_template | (df["Type"].astype(int) == 2)]

        if df.empty:
            continue

        # Slack CVs are typically increments of 0.1
        df["cv"] = np.round(df["cv"].to_numpy(float), 1)

        # correctness: Response == Type
        is_correct = (df["Response"].astype(int) == df["Type"].astype(int)).astype(int)

        for (freq, cv), g in df.groupby(["freq", "cv"]):
            n = int(len(g))
            k = int(is_correct.loc[g.index].sum())
            rows.append(
                dict(rat_name=str(rat), freq=float(freq), block=str(block), cv=float(cv), k=k, n=n)
            )

    if not rows:
        raise RuntimeError(f"No usable Slack trials in slack_dir={slack_dir}")

    slack_trials = pd.DataFrame(rows)
    slack_trials["p_correct"] = slack_trials["k"] / slack_trials["n"].clip(lower=1)

    if aggregate_sessions_by_rat:
        slack_perrat = (
            slack_trials.groupby(["rat_name", "freq", "block", "cv"], as_index=False)
            .agg(k=("k", "sum"), n=("n", "sum"))
        )
        slack_perrat["p_correct"] = slack_perrat["k"] / slack_perrat["n"].clip(lower=1)
    else:
        slack_perrat = slack_trials.copy()

    # Mean across rats
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if x.size <= 1:
            return 0.0
        return float(np.std(x, ddof=1) / math.sqrt(x.size))

    agg = (
        slack_perrat.groupby(["freq", "block", "cv"], as_index=False)
        .agg(k=("k", "sum"), n=("n", "sum"), p_mean=("p_correct", "mean"), p_sem=("p_correct", _sem), n_trials=("p_correct", "size"))
    )

    return slack_perrat, agg
