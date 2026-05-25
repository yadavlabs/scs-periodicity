"""Shared utilities for notebooks and manuscript figure generation.

Python package names use ``snake_case`` for importability, while data folders
keep their publication-facing names.  Use the path helpers here to avoid
sprinkling those filesystem spellings through notebooks.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_FOLDER_NAMES = {
    "comsol": "comsol",
    "time_patterns": "time-patterns",
    "space_time": "space-time-V",
    "detection": "detection",
    "discrimination": "discrimination",
}


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a cwd or notebook subdirectory."""

    here = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {here}")


def ensure_src_on_path(repo_root: str | Path | None = None) -> Path:
    """Add ``repo_root/src`` to ``sys.path`` and return that path."""

    root = find_repo_root(repo_root)
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def data_subdir(repo_root: str | Path, key: str) -> Path:
    """Return a canonical data subdirectory by Python-facing key."""

    try:
        folder = DATA_FOLDER_NAMES[key]
    except KeyError as exc:
        known = ", ".join(sorted(DATA_FOLDER_NAMES))
        raise KeyError(f"Unknown data folder key {key!r}. Known keys: {known}") from exc
    return Path(repo_root).resolve() / "data" / folder


def notebook_cache_dir(repo_root: str | Path) -> Path:
    """Return the repository-local cache folder used by review notebooks."""

    return Path(repo_root).resolve() / "notebooks" / "cache"


def require_data_files(
    repo_root: str | Path,
    relative_paths: list[str] | tuple[str, ...],
    *,
    context: str = "this notebook",
) -> None:
    """Raise a clear data-bundle error if required release files are missing."""

    root = Path(repo_root).resolve()
    missing = [rel for rel in relative_paths if not (root / rel).exists()]
    if not missing:
        return

    shown = "\n".join(f"  - {rel}" for rel in missing[:20])
    extra = ""
    if len(missing) > 20:
        extra = f"\n  ... and {len(missing) - 20} more"
    raise FileNotFoundError(
        f"Required data files are missing for {context}.\n\n"
        f"Expected the release-data archive to be extracted at:\n"
        f"  {root / 'data'}\n\n"
        f"Missing:\n{shown}{extra}\n\n"
        "Download the release-data ZIP from GitHub Releases or Zenodo and "
        "extract it into the repository root "
        "before running this notebook."
    )


def to_num(value: Any):
    """Convert scalars or pandas objects to numeric values with NaNs on errors."""

    return pd.to_numeric(value, errors="coerce")


def sem(values: Any) -> float:
    """Return standard error of the mean, using 0 for singleton vectors."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def get_spikes_col(df: pd.DataFrame) -> str | None:
    """Return the first recognized spike-time column name in a response table."""

    for col in ("spikes_ms", "spike_times_ms", "spike_times", "spikes"):
        if col in df.columns:
            return col
    return None


def parse_spikes(value: Any) -> np.ndarray:
    """Parse spike times stored as arrays, JSON strings, or Python literals."""

    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return arr[np.isfinite(arr)]
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.asarray([], dtype=float)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return np.asarray([], dtype=float)
    for parser in (json.loads, ast.literal_eval):
        try:
            arr = np.asarray(parser(text), dtype=float)
            return arr[np.isfinite(arr)]
        except Exception:
            continue
    return np.asarray([], dtype=float)


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read(path))
    return ["".join(t.text or "" for t in si.findall(".//a:t", ns)) for si in root.findall("a:si", ns)]


def _xlsx_sheet_rows(xlsx_path: Path) -> dict[int, dict[str, str]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = _xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: dict[int, dict[str, str]] = {}
    for row in root.findall(".//a:row", ns):
        row_number = int(row.attrib.get("r", 0))
        vals: dict[str, str] = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", ref)
            if not match:
                continue
            col = match.group(1)
            kind = cell.attrib.get("t")
            value_node = cell.find("a:v", ns)
            if value_node is None:
                vals[col] = ""
                continue
            raw = value_node.text or ""
            if kind == "s":
                try:
                    vals[col] = shared_strings[int(raw)]
                except Exception:
                    vals[col] = raw
            else:
                vals[col] = raw
        rows[row_number] = vals
    return rows


def parse_detection_behavioral_xlsx(xlsx_path: str | Path) -> pd.DataFrame:
    """Parse the Slack detection-threshold spreadsheet used in figure 06."""

    path = Path(xlsx_path)
    columns = ["freq", "cv", "rat", "threshold_uA", "norm_threshold"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    rows = _xlsx_sheet_rows(path)
    out: list[dict[str, float | int]] = []
    row_numbers = sorted(rows.keys())
    i = 0
    while i < len(row_numbers):
        row_number = row_numbers[i]
        vals = rows[row_number]
        tag = str(vals.get("C", "")).strip()
        match = re.match(r"^(\d+)Hz$", tag)
        if not match:
            i += 1
            continue

        freq = float(match.group(1))
        cv_row = rows.get(row_number + 1, {})
        cvs = []
        for col in ["C", "D", "E", "F", "G", "H"]:
            text = str(cv_row.get(col, "")).strip()
            if text.lower().startswith("cv"):
                cvs.append(0.0)
            else:
                try:
                    cvs.append(float(text))
                except Exception:
                    cvs.append(np.nan)

        rr = row_number + 3
        while rr in rows:
            row_values = rows[rr]
            if re.match(r"^(\d+)Hz$", str(row_values.get("C", "")).strip()):
                break
            rat = str(row_values.get("B", "")).strip()
            if rat == "" and str(row_values.get("C", "")).strip() == "":
                rr += 1
                if rr + 1 in rows and re.match(r"^(\d+)Hz$", str(rows[rr + 1].get("C", "")).strip()):
                    break
                continue
            try:
                rat_i = int(float(rat))
            except Exception:
                rr += 1
                continue

            abs_cols = ["C", "D", "E", "F", "G", "H"]
            norm_cols = ["I", "J", "K", "L", "M", "N"]
            for abs_col, norm_col, cv in zip(abs_cols, norm_cols, cvs):
                try:
                    threshold = float(row_values.get(abs_col, ""))
                except Exception:
                    threshold = np.nan
                try:
                    norm = float(row_values.get(norm_col, ""))
                except Exception:
                    norm = np.nan
                if np.isfinite(cv) and (np.isfinite(threshold) or np.isfinite(norm)):
                    out.append(
                        {
                            "freq": freq,
                            "cv": float(cv),
                            "rat": rat_i,
                            "threshold_uA": float(threshold) if np.isfinite(threshold) else np.nan,
                            "norm_threshold": float(norm) if np.isfinite(norm) else np.nan,
                        }
                    )
            rr += 1
        i += 1

    df = pd.DataFrame(out, columns=columns)
    if df.empty:
        return df
    df["freq"] = to_num(df["freq"])
    df["cv"] = to_num(df["cv"])
    df["rat"] = to_num(df["rat"])
    return df[np.isfinite(df["freq"]) & np.isfinite(df["cv"])].copy()
