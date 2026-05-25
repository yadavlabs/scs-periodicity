"""Stage 02 helpers for constructing space-time extracellular potentials."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import interpolate


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CV_VALUES = tuple(round(i / 10, 1) for i in range(11))
DEFAULT_FREQUENCIES_HZ = (10, 20, 50, 100, 200)


@dataclass(frozen=True)
class SpaceTimeConfig:
    """Configuration for stage 02 space-time voltage generation."""

    cv_vals: Sequence[float] = DEFAULT_CV_VALUES
    freq_vals: Sequence[int] = DEFAULT_FREQUENCIES_HZ
    cutline: int = 101
    fs: float = 20_000.0
    comsol_dir: Path = REPO_ROOT / "data" / "comsol"
    time_pattern_dir: Path = REPO_ROOT / "data" / "time-patterns" / "generated"
    output_dir: Path = REPO_ROOT / "data" / "space-time-V" / "generated"
    n_jobs: int = 1

    def resolved(self) -> "SpaceTimeConfig":
        return replace(
            self,
            cv_vals=tuple(float(cv) for cv in self.cv_vals),
            freq_vals=tuple(int(freq) for freq in self.freq_vals),
            comsol_dir=Path(self.comsol_dir).expanduser().resolve(),
            time_pattern_dir=Path(self.time_pattern_dir).expanduser().resolve(),
            output_dir=Path(self.output_dir).expanduser().resolve(),
        )


@dataclass(frozen=True)
class ComsolProfiles:
    """Interpolated +1 uA and -1 uA COMSOL profiles at model nodes."""

    cutline: int
    node_numbers: np.ndarray
    v_neg_mv: np.ndarray
    v_pos_mv: np.ndarray
    neg_path: Path
    pos_path: Path


@dataclass(frozen=True)
class SpaceTimeData:
    """Loaded saved stage-02 bundle."""

    time_s: np.ndarray
    node_numbers: np.ndarray
    rep_labels: np.ndarray
    current: np.ndarray
    v_ext_mv: np.ndarray


@dataclass(frozen=True)
class SpaceTimeStatus:
    """Verification status for stage 02 inputs and outputs."""

    input_files: tuple[Path, ...]
    output_files: tuple[Path, ...]
    missing_input_files: tuple[Path, ...]
    missing_output_files: tuple[Path, ...]

    @property
    def inputs_complete(self) -> bool:
        return not self.missing_input_files

    @property
    def outputs_complete(self) -> bool:
        return not self.missing_output_files

    @property
    def complete(self) -> bool:
        return self.inputs_complete and self.outputs_complete

    def summary(self) -> str:
        input_msg = "OK" if self.inputs_complete else f"missing {len(self.missing_input_files)}"
        output_msg = "OK" if self.outputs_complete else f"missing {len(self.missing_output_files)}"
        return f"space-time inputs: {input_msg}; outputs: {output_msg}"

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for role, paths in (("input", self.input_files), ("output", self.output_files)):
            for path in paths:
                rows.append(
                    {
                        "role": role,
                        "path": str(_display_path(path)),
                        "exists": path.exists(),
                        "size_bytes": path.stat().st_size if path.exists() else None,
                    }
                )
        return rows


@dataclass(frozen=True)
class SpaceTimeResult:
    """Result returned by the stage runner."""

    ran: bool
    status: SpaceTimeStatus
    output_dir: Path

    def summary(self) -> str:
        action = "generated" if self.ran else "reused existing"
        return f"Stage 02 {action} outputs. {self.status.summary()}"


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def iter_pairs(config: SpaceTimeConfig) -> list[tuple[float, int]]:
    return [(float(cv), int(freq)) for freq in config.freq_vals for cv in config.cv_vals]


def comsol_profile_paths(config: SpaceTimeConfig) -> tuple[Path, Path]:
    config = config.resolved()
    neg_path = config.comsol_dir / f"M{config.cutline}_1uA_bcs1_neg.txt"
    pos_path = config.comsol_dir / f"M{config.cutline}_1uA_bcs1_pos.txt"
    return neg_path, pos_path


def time_pattern_input_path(config: SpaceTimeConfig, cv: float, freq: float | int) -> Path:
    config = config.resolved()
    return config.time_pattern_dir / f"cv{float(cv):.1f}_f{int(freq)}_waveforms.npz"


def spacetime_output_path(config: SpaceTimeConfig, cv: float, freq: float | int) -> Path:
    config = config.resolved()
    return config.output_dir / (
        f"M{config.cutline}_1uA_cv{float(cv):.1f}_f{int(freq)}_SpaceTimeV.npz"
    )


def expected_time_pattern_paths(config: SpaceTimeConfig) -> tuple[Path, ...]:
    config = config.resolved()
    return tuple(time_pattern_input_path(config, cv, freq) for cv, freq in iter_pairs(config))


def expected_output_paths(config: SpaceTimeConfig) -> tuple[Path, ...]:
    config = config.resolved()
    return tuple(spacetime_output_path(config, cv, freq) for cv, freq in iter_pairs(config))


def verify_spacetime_outputs(config: SpaceTimeConfig | None = None) -> SpaceTimeStatus:
    config = (config or SpaceTimeConfig()).resolved()
    neg_path, pos_path = comsol_profile_paths(config)
    input_files = (neg_path, pos_path, *expected_time_pattern_paths(config))
    output_files = expected_output_paths(config)
    missing_inputs = tuple(path for path in input_files if not path.exists())
    missing_outputs = tuple(path for path in output_files if not path.exists())
    return SpaceTimeStatus(
        input_files=input_files,
        output_files=output_files,
        missing_input_files=missing_inputs,
        missing_output_files=missing_outputs,
    )


def _load_comsol_text(path: Path) -> pd.DataFrame:
    """Load a COMSOL line-graph text export into arc length and potential columns."""
    return pd.read_csv(
        path,
        comment="%",
        sep=r"\s+",
        header=None,
        names=["arc_length_mm", "potential_v"],
    )


def load_comsol_profiles(config: SpaceTimeConfig | None = None) -> ComsolProfiles:
    """Load +1 uA and -1 uA COMSOL profiles and interpolate them to nodes 0..25."""
    config = (config or SpaceTimeConfig()).resolved()
    neg_path, pos_path = comsol_profile_paths(config)
    df_neg = _load_comsol_text(neg_path)
    df_pos = _load_comsol_text(pos_path)

    interp_nodes = np.arange(0, 26, 1, dtype=float)
    f_neg = interpolate.interp1d(
        df_neg["arc_length_mm"].to_numpy(dtype=float),
        df_neg["potential_v"].to_numpy(dtype=float) * 1000.0,
        kind="cubic",
        fill_value="extrapolate",
    )
    f_pos = interpolate.interp1d(
        df_pos["arc_length_mm"].to_numpy(dtype=float),
        df_pos["potential_v"].to_numpy(dtype=float) * 1000.0,
        kind="cubic",
        fill_value="extrapolate",
    )

    ev_neg = f_neg(interp_nodes)
    ev_pos = f_pos(interp_nodes)
    node_numbers = np.arange(26, dtype=int)
    v_neg_mv = np.zeros_like(node_numbers, dtype=float)
    v_pos_mv = np.zeros_like(node_numbers, dtype=float)

    for i in np.arange(len(ev_neg) - 1, -1, -1):
        node_idx = int(25 - i)
        v_neg_mv[node_idx] = ev_neg[i]
        v_pos_mv[node_idx] = ev_pos[i]

    return ComsolProfiles(
        cutline=config.cutline,
        node_numbers=node_numbers,
        v_neg_mv=v_neg_mv,
        v_pos_mv=v_pos_mv,
        neg_path=neg_path,
        pos_path=pos_path,
    )


def load_time_pattern_currents(
    config: SpaceTimeConfig | None = None,
    cv: float = 0.7,
    freq: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all stage-01 current waveforms for one CV/frequency pair."""
    config = (config or SpaceTimeConfig()).resolved()
    npz_path = time_pattern_input_path(config, cv, freq)
    with np.load(npz_path) as bundle:
        rep_labels = np.array(sorted(bundle.files))
        current = np.stack(
            [np.array(bundle[label], dtype=np.float32) for label in rep_labels],
            axis=0,
        )

    time_s = np.arange(current.shape[1], dtype=np.float32) / np.float32(config.fs)
    return time_s, rep_labels, current


def load_time_pattern_repetition(
    config: SpaceTimeConfig | None = None,
    cv: float = 0.7,
    freq: int = 100,
    rep_label_substr: str = "rep002",
) -> tuple[np.ndarray, str, np.ndarray]:
    """Load one stage-01 current waveform by repetition-label substring."""
    config = (config or SpaceTimeConfig()).resolved()
    npz_path = time_pattern_input_path(config, cv, freq)
    with np.load(npz_path) as bundle:
        rep_labels = np.array(sorted(bundle.files))
        matches = [i for i, label in enumerate(rep_labels) if rep_label_substr in str(label)]
        if not matches:
            raise ValueError(f"No repetition label contains {rep_label_substr!r}.")
        label = str(rep_labels[matches[0]])
        current = np.array(bundle[label], dtype=np.float32)

    time_s = np.arange(current.shape[0], dtype=np.float32) / np.float32(config.fs)
    return time_s, label, current


def map_current_to_spacetime(
    current: np.ndarray,
    v_neg_mv: np.ndarray,
    v_pos_mv: np.ndarray,
) -> np.ndarray:
    """Map sampled current signs to COMSOL voltage profiles."""
    current = np.asarray(current, dtype=np.float32)
    v_neg_mv = np.asarray(v_neg_mv, dtype=np.float32)
    v_pos_mv = np.asarray(v_pos_mv, dtype=np.float32)
    n_nodes = v_neg_mv.shape[0]

    if current.ndim == 1:
        neg_mask = current[:, None] < 0
        pos_mask = current[:, None] > 0
        return (
            np.where(neg_mask, v_neg_mv.reshape(1, n_nodes), np.float32(0.0))
            + np.where(pos_mask, v_pos_mv.reshape(1, n_nodes), np.float32(0.0))
        )

    if current.ndim == 2:
        neg_mask = current[..., None] < 0
        pos_mask = current[..., None] > 0
        return (
            np.where(neg_mask, v_neg_mv.reshape(1, 1, n_nodes), np.float32(0.0))
            + np.where(pos_mask, v_pos_mv.reshape(1, 1, n_nodes), np.float32(0.0))
        )

    raise ValueError(f"current must be 1D or 2D, got shape {current.shape}.")


def generate_spacetime_for_pair(
    config: SpaceTimeConfig,
    profiles: ComsolProfiles,
    cv: float,
    freq: float | int,
) -> dict[str, object]:
    """Generate and save the full stage-02 bundle for one CV/frequency pair."""
    config = config.resolved()
    time_s, rep_labels, current = load_time_pattern_currents(config, cv=cv, freq=int(freq))
    v_ext_mv = map_current_to_spacetime(current, profiles.v_neg_mv, profiles.v_pos_mv)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = spacetime_output_path(config, cv, freq)
    np.savez_compressed(
        out_npz,
        time_s=time_s,
        node_numbers=profiles.node_numbers,
        rep_labels=rep_labels,
        current=current,
        V=v_ext_mv,
    )

    info = {
        "cv": float(cv),
        "freq": int(freq),
        "n_reps": int(current.shape[0]),
        "T": int(current.shape[1]),
        "N": int(profiles.node_numbers.shape[0]),
        "npz": out_npz,
    }
    del current, v_ext_mv
    return info


def _generate_pair_job(
    config: SpaceTimeConfig,
    profiles: ComsolProfiles,
    cv: float,
    freq: int,
) -> dict[str, object]:
    return generate_spacetime_for_pair(config, profiles, cv, freq)


def generate_all_spacetime(config: SpaceTimeConfig | None = None) -> dict[tuple[float, int], dict[str, object]]:
    """Generate and save stage-02 bundles for every configured CV/frequency pair."""
    config = (config or SpaceTimeConfig()).resolved()
    profiles = load_comsol_profiles(config)
    pairs = iter_pairs(config)

    if config.n_jobs == 1:
        results = [
            generate_spacetime_for_pair(config=config, profiles=profiles, cv=cv, freq=freq)
            for cv, freq in pairs
        ]
    else:
        results = Parallel(n_jobs=config.n_jobs)(
            delayed(_generate_pair_job)(config, profiles, cv, freq) for cv, freq in pairs
        )

    return {(float(item["cv"]), int(item["freq"])): item for item in results}


def run_stage(config: SpaceTimeConfig | None = None, force: bool = False) -> SpaceTimeResult:
    """Verify stage 02 outputs and regenerate them when missing or forced."""
    config = (config or SpaceTimeConfig()).resolved()
    status = verify_spacetime_outputs(config)
    if status.missing_input_files:
        preview = "\n".join(str(path) for path in status.missing_input_files[:5])
        raise FileNotFoundError(
            "Stage 02 input files are missing. First missing files:\n" + preview
        )

    if status.outputs_complete and not force:
        return SpaceTimeResult(ran=False, status=status, output_dir=config.output_dir)

    generate_all_spacetime(config)
    status = verify_spacetime_outputs(config)
    if not status.outputs_complete:
        preview = "\n".join(str(path) for path in status.missing_output_files[:5])
        raise RuntimeError("Stage 02 generation finished, but outputs are still missing:\n" + preview)

    return SpaceTimeResult(ran=True, status=status, output_dir=config.output_dir)


def load_spacetime_npz(path: str | Path) -> SpaceTimeData:
    """Load one saved stage-02 bundle."""
    with np.load(path) as data:
        return SpaceTimeData(
            time_s=data["time_s"].copy(),
            node_numbers=data["node_numbers"].copy(),
            rep_labels=data["rep_labels"].copy(),
            current=data["current"].copy(),
            v_ext_mv=data["V"].copy(),
        )


def describe_spacetime_npz(path: str | Path) -> list[dict[str, object]]:
    """Read saved stage-02 array names, shapes, and dtypes without loading arrays."""
    path = Path(path)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".npy"):
                continue
            array_name = Path(name).stem
            with archive.open(name) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    raise ValueError(f"Unsupported .npy header version {version} in {path}.")
            rows.append(
                {
                    "array": array_name,
                    "shape": shape,
                    "dtype": str(dtype),
                }
            )
    return rows
