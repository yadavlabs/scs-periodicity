"""detection_CLI.py

Cluster-friendly wrapper around detection_core that supports:
- Running thresholds per-file (job arrays) -> partial threshold + true-response CSVs
- Merging partial CSVs -> full detection_thresholds.csv
- Computing cv0_threshold_max.csv from the merged thresholds
- Merging true-threshold responses -> full detection_responses.csv
- Running cv0-based responses per-file (job arrays) -> partial CSVs
- Merging partial responses -> full cv0_based_responses.csv

Thresholds computed in detection_core (per rep) now include:
  I_first, I_volley5, I_volley10, I_5, I_10, I_15, I_25, I_50, I_max

Usage patterns:
  1) Create a file list once:
        ls /path/to/data/M*_SpaceTimeV.npz > results/npz_files.txt

  2) Array job: thresholds + true-threshold responses (one file per task):
        python src/detection/detection_CLI.py run-thresholds \
            --file-list results/npz_files.txt --file-idx $SLURM_ARRAY_TASK_ID \
            --n-jobs 24 --partial-dir data/partials/detection/thresholds \
            --response-partial-dir data/partials/detection/detection_responses

  3) Merge thresholds/responses + compute cv0 table:
        python src/detection/detection_CLI.py merge-thresholds --partial-dir data/partials/detection/thresholds --out results/detection_thresholds.csv
        python src/detection/detection_CLI.py merge-detection-responses --partial-dir data/partials/detection/detection_responses --out results/detection_responses.csv
        python src/detection/detection_CLI.py compute-cv0 --det-csv results/detection_thresholds.csv --out results/cv0_threshold_max.csv

  Optional: rerun true-threshold responses from an existing detection_thresholds.csv:
        python src/detection/detection_CLI.py run-detection-responses \
            --file-list results/npz_files.txt --file-idx $SLURM_ARRAY_TASK_ID \
            --n-jobs 24 --det-csv results/detection_thresholds.csv \
            --partial-dir data/partials/detection/detection_responses

  4) Array job: CV0-based responses (one file per task):
        python src/detection/detection_CLI.py run-responses \
            --file-list results/npz_files.txt --file-idx $SLURM_ARRAY_TASK_ID \
            --n-jobs 24 --cv0-csv results/cv0_threshold_max.csv \
            --partial-dir data/partials/detection/responses

  5) Merge CV0-based responses:
        python src/detection/detection_CLI.py merge-responses --partial-dir data/partials/detection/responses --out results/cv0_based_responses.csv
"""

import argparse
import os
from pathlib import Path
import pandas as pd

try:
    from detection.detection_core import (
        run_all_reps,
        run_responses_from_cv0,
        run_responses_from_thresholds,
        compute_cv0_threshold_max,
    )
except ModuleNotFoundError:
    # Allows direct execution via `python src/detection/detection_CLI.py ...`
    from detection_core import (  # type: ignore
        run_all_reps,
        run_responses_from_cv0,
        run_responses_from_thresholds,
        compute_cv0_threshold_max,
    )

# ---------------- Helpers ----------------

def _read_file_list(file_list: Path):
    lines = [ln.strip() for ln in file_list.read_text().splitlines() if ln.strip()]
    return [Path(x).resolve() for x in lines]


def _select_npz(npz: str | None, file_list: str | None, file_idx: int | None) -> Path:
    """Select the .npz for a given array task."""
    if npz is not None:
        return Path(npz).resolve()

    if file_list is None:
        raise ValueError("Provide either --npz OR (--file-list and --file-idx).")

    fl = Path(file_list).resolve()
    files = _read_file_list(fl)

    if file_idx is None:
        env_idx = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        if env_idx is None:
            raise ValueError("No --file-idx provided and SLURM_ARRAY_TASK_ID not set.")
        file_idx = int(env_idx)

    if file_idx < 0 or file_idx >= len(files):
        raise IndexError(f"file_idx={file_idx} out of range (0..{len(files)-1})")

    return files[file_idx]


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _merge_csvs(partial_dir: Path, pattern: str, out_csv: Path):
    """Concatenate partial CSVs into a single output file."""
    partial_dir = partial_dir.resolve()
    out_csv = out_csv.resolve()
    _ensure_parent(out_csv)

    paths = sorted(partial_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No partial CSVs found: {partial_dir}/{pattern}")

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed reading {p}: {e}")

    if not dfs:
        raise RuntimeError("All partial CSVs were empty/unreadable.")

    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all.to_csv(out_csv, index=False)
    print(f"[merge] Wrote {len(df_all)} rows -> {out_csv}")
    return out_csv


# ---------------- Pipeline entrypoints ----------------

def run_thresholds_one_file(
    npz_path: Path,
    n_jobs: int,
    partial_dir: Path,
    chunk_size: int = 200,
    response_partial_dir: Path | None = None,
):
    """Array-task entrypoint: compute thresholds and true-threshold responses for one NPZ."""
    partial_dir = partial_dir.resolve()
    partial_dir.mkdir(parents=True, exist_ok=True)

    out_csv = partial_dir / f"detection_thresholds__{npz_path.stem}.csv"
    response_out_csv = None
    if response_partial_dir is not None:
        response_partial_dir = response_partial_dir.resolve()
        response_partial_dir.mkdir(parents=True, exist_ok=True)
        response_out_csv = response_partial_dir / f"detection_responses__{npz_path.stem}.csv"

    print(f"[run-thresholds] file={npz_path}")
    print(f"[run-thresholds] partial_out={out_csv}")
    if response_out_csv is not None:
        print(f"[run-thresholds] response_partial_out={response_out_csv}")
    print(f"[run-thresholds] n_jobs={n_jobs}")

    run_all_reps(
        n_jobs=n_jobs,
        output_csv=out_csv,
        response_output_csv=response_out_csv,
        data_dir=npz_path.parent,
        npz_paths=[npz_path],
        chunk_size=chunk_size,
        compute_cv0_after=False,   # IMPORTANT for array jobs
    )


def run_responses_one_file(npz_path: Path, n_jobs: int, cv0_csv: Path, partial_dir: Path):
    """Array-task entrypoint: compute cv0-based responses for one NPZ."""
    partial_dir = partial_dir.resolve()
    partial_dir.mkdir(parents=True, exist_ok=True)

    out_csv = partial_dir / f"cv0_based_responses__{npz_path.stem}.csv"
    print(f"[run-responses] file={npz_path}")
    print(f"[run-responses] cv0_csv={cv0_csv}")
    print(f"[run-responses] partial_out={out_csv}")
    print(f"[run-responses] n_jobs={n_jobs}")

    run_responses_from_cv0(
        n_jobs=n_jobs,
        data_dir=npz_path.parent,
        npz_paths=[npz_path],
        cv0_thresh_csv=cv0_csv,
        output_csv=out_csv,
    )


def run_detection_responses_one_file(npz_path: Path, n_jobs: int, det_csv: Path, partial_dir: Path):
    """Array-task entrypoint: compute true-threshold responses for one NPZ."""
    partial_dir = partial_dir.resolve()
    partial_dir.mkdir(parents=True, exist_ok=True)

    out_csv = partial_dir / f"detection_responses__{npz_path.stem}.csv"
    print(f"[run-detection-responses] file={npz_path}")
    print(f"[run-detection-responses] det_csv={det_csv}")
    print(f"[run-detection-responses] partial_out={out_csv}")
    print(f"[run-detection-responses] n_jobs={n_jobs}")

    run_responses_from_thresholds(
        n_jobs=n_jobs,
        data_dir=npz_path.parent,
        npz_paths=[npz_path],
        threshold_csv=det_csv,
        output_csv=out_csv,
    )


def run_all(dev_data_dir: Path, n_jobs: int, results_dir: Path):
    """Convenience: run everything locally (NOT array-based)."""
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    det_csv = results_dir / "detection_thresholds.csv"
    cv0_csv = results_dir / "cv0_threshold_max.csv"
    det_resp_csv = results_dir / "detection_responses.csv"
    resp_csv = results_dir / "cv0_based_responses.csv"

    run_all_reps(
        n_jobs=n_jobs,
        output_csv=det_csv,
        response_output_csv=det_resp_csv,
        data_dir=dev_data_dir,
        npz_paths=None,
        compute_cv0_after=False,
    )

    df_det = pd.read_csv(det_csv)
    compute_cv0_threshold_max(df_det.to_dict(orient="records"), output_csv=cv0_csv, safety_margin_uA=0.0)

    run_responses_from_cv0(
        n_jobs=n_jobs,
        data_dir=dev_data_dir,
        cv0_thresh_csv=cv0_csv,
        output_csv=resp_csv,
    )


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Threshold + response pipeline (cluster-friendly).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- run-thresholds ---
    p = sub.add_parser("run-thresholds", help="Run thresholds for one .npz (array task).")
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--file-list", type=str, default=None)
    p.add_argument("--file-idx", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--partial-dir", type=str, required=True)
    p.add_argument("--response-partial-dir", type=str, default=None)
    p.add_argument("--chunk-size", type=int, default=200)

    # --- merge-thresholds ---
    p = sub.add_parser("merge-thresholds", help="Merge partial threshold CSVs.")
    p.add_argument("--partial-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    # --- compute-cv0 ---
    p = sub.add_parser("compute-cv0", help="Compute cv0_threshold_max.csv from merged detection_thresholds.csv.")
    p.add_argument("--det-csv", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--safety-margin", type=float, default=0.0)

    # --- run-detection-responses ---
    p = sub.add_parser("run-detection-responses", help="Run true-threshold responses for one .npz (array task).")
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--file-list", type=str, default=None)
    p.add_argument("--file-idx", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--det-csv", type=str, required=True)
    p.add_argument("--partial-dir", type=str, required=True)

    # --- merge-detection-responses ---
    p = sub.add_parser("merge-detection-responses", help="Merge partial true-threshold response CSVs.")
    p.add_argument("--partial-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    # --- run-responses ---
    p = sub.add_parser("run-responses", help="Run cv0-based responses for one .npz (array task).")
    p.add_argument("--npz", type=str, default=None)
    p.add_argument("--file-list", type=str, default=None)
    p.add_argument("--file-idx", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--cv0-csv", type=str, required=True)
    p.add_argument("--partial-dir", type=str, required=True)

    # --- merge-responses ---
    p = sub.add_parser("merge-responses", help="Merge partial response CSVs.")
    p.add_argument("--partial-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)

    # --- run-all (dev) ---
    p = sub.add_parser("run-all", help="Run full pipeline locally (not array-based).")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--results-dir", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "run-thresholds":
        npz_path = _select_npz(args.npz, args.file_list, args.file_idx)
        run_thresholds_one_file(
            npz_path=npz_path,
            n_jobs=int(args.n_jobs),
            partial_dir=Path(args.partial_dir),
            chunk_size=int(args.chunk_size),
            response_partial_dir=(None if args.response_partial_dir is None else Path(args.response_partial_dir)),
        )
        return

    if args.cmd == "merge-thresholds":
        _merge_csvs(Path(args.partial_dir), "detection_thresholds__*.csv", Path(args.out))
        return

    if args.cmd == "compute-cv0":
        det_csv = Path(args.det_csv).resolve()
        out_csv = Path(args.out).resolve()
        df = pd.read_csv(det_csv)
        compute_cv0_threshold_max(
            df.to_dict(orient="records"),
            output_csv=out_csv,
            safety_margin_uA=float(args.safety_margin),
        )
        return

    if args.cmd == "run-detection-responses":
        npz_path = _select_npz(args.npz, args.file_list, args.file_idx)
        run_detection_responses_one_file(
            npz_path=npz_path,
            n_jobs=int(args.n_jobs),
            det_csv=Path(args.det_csv).resolve(),
            partial_dir=Path(args.partial_dir),
        )
        return

    if args.cmd == "merge-detection-responses":
        _merge_csvs(Path(args.partial_dir), "detection_responses__*.csv", Path(args.out))
        return

    if args.cmd == "run-responses":
        npz_path = _select_npz(args.npz, args.file_list, args.file_idx)
        run_responses_one_file(
            npz_path=npz_path,
            n_jobs=int(args.n_jobs),
            cv0_csv=Path(args.cv0_csv).resolve(),
            partial_dir=Path(args.partial_dir),
        )
        return

    if args.cmd == "merge-responses":
        _merge_csvs(Path(args.partial_dir), "cv0_based_responses__*.csv", Path(args.out))
        return

    if args.cmd == "run-all":
        run_all(
            dev_data_dir=Path(args.data_dir).resolve(),
            n_jobs=int(args.n_jobs),
            results_dir=Path(args.results_dir).resolve(),
        )
        return


if __name__ == "__main__":
    main()
