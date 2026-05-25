"""Stage 01 helpers for generating stimulation time patterns.

The notebook for this stage is intentionally thin. This module contains the
reusable implementation so notebooks, scripts, and tests all call the same
pipeline entry points.
"""

from __future__ import annotations

import csv
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CV_VALUES = tuple(round(i / 10, 1) for i in range(11))
DEFAULT_FREQUENCIES_HZ = (10, 20, 50, 100, 200)


def display_path(path: str | Path) -> str:
    """Return a portable repo-relative path when possible."""

    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class StimSpec:
    """Sampling grid and biphasic pulse shape used for generated waveforms."""

    fs: float = 20_000.0
    stim_dur_s: float = 2.0
    post_silence_s: float = 1.0
    amp_uA: float = 1.0
    neg_samples: int = 4
    interphase_samples: int = 1
    pos_samples: int = 4

    @classmethod
    def from_pulse_width(
        cls,
        pulse_width_s: float,
        stim_dur_s: float,
        fs: float = 20_000.0,
        amp_uA: float = 1.0,
        post_silence_s: float = 1.0,
    ) -> "StimSpec":
        """Build the manuscript pulse shape from a total pulse width."""
        interphase_samples = 1
        interphase_s = interphase_samples / fs
        phase_samples = int(round(fs * (pulse_width_s - interphase_s) / 2.0))
        if phase_samples < 1:
            raise ValueError("pulse_width_s is too short for the sampling grid.")

        return cls(
            fs=fs,
            stim_dur_s=float(stim_dur_s),
            post_silence_s=float(post_silence_s),
            amp_uA=float(amp_uA),
            neg_samples=phase_samples,
            interphase_samples=interphase_samples,
            pos_samples=phase_samples,
        )

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def pulse_samples(self) -> int:
        return self.neg_samples + self.interphase_samples + self.pos_samples

    @property
    def min_ipi_s(self) -> float:
        return self.pulse_samples / self.fs

    @property
    def stim_samples(self) -> int:
        return int(round(self.stim_dur_s * self.fs) + 1)

    @property
    def total_samples(self) -> int:
        return int(round((self.stim_dur_s + self.post_silence_s) * self.fs) + 1)


@dataclass(frozen=True)
class TimePatternConfig:
    """Configuration for the full stage 01 time-pattern generation run."""

    cv_vals: Sequence[float] = DEFAULT_CV_VALUES
    freq_vals: Sequence[int] = DEFAULT_FREQUENCIES_HZ
    pulse_width_s: float = 450e-6
    stim_duration_s: float = 2.0
    slack_dir: Path = REPO_ROOT / "data" / "time-patterns" / "slack"
    output_dir: Path = REPO_ROOT / "data" / "time-patterns" / "generated"
    parallel: bool = True
    base_seed: int = 123

    def resolved(self) -> "TimePatternConfig":
        """Return a normalized copy with concrete tuples and absolute paths."""
        return replace(
            self,
            cv_vals=tuple(float(cv) for cv in self.cv_vals),
            freq_vals=tuple(int(freq) for freq in self.freq_vals),
            slack_dir=Path(self.slack_dir).expanduser().resolve(),
            output_dir=Path(self.output_dir).expanduser().resolve(),
        )

    @property
    def spec(self) -> StimSpec:
        return StimSpec.from_pulse_width(
            pulse_width_s=self.pulse_width_s,
            stim_dur_s=self.stim_duration_s,
        )


@dataclass(frozen=True)
class TimePatternStatus:
    """Verification status for stage 01 inputs and outputs."""

    input_files: tuple[Path, ...]
    output_files: tuple[Path, ...]
    missing_input_files: tuple[Path, ...]
    missing_output_files: tuple[Path, ...]
    manifest_csv: Path

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
        return (
            f"time-pattern inputs: {input_msg}; "
            f"outputs: {output_msg}; "
            f"manifest: {self.manifest_csv}"
        )

    def rows(self) -> list[dict[str, object]]:
        """Return a compact file table that displays well in notebooks."""
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
class TimePatternResult:
    """Result returned by the stage runner."""

    ran: bool
    status: TimePatternStatus
    manifest_csv: Path
    output_dir: Path

    def summary(self) -> str:
        action = "generated" if self.ran else "reused existing"
        return f"Stage 01 {action} outputs. {self.status.summary()}"


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def iter_pairs(config: TimePatternConfig) -> Iterable[tuple[float, int]]:
    for cv in config.cv_vals:
        for freq in config.freq_vals:
            yield float(cv), int(freq)


def seed_for_pair(config: TimePatternConfig, cv: float, freq: float | int) -> int:
    """Return the deterministic per-pair seed used by the full generator."""
    config = config.resolved()
    target = (float(cv), int(freq))
    pairs = list(iter_pairs(config))
    try:
        pair_index = pairs.index(target)
    except ValueError as exc:
        raise ValueError(f"Pair {target} is not included in this time-pattern config.") from exc

    seeds = np.random.default_rng(config.base_seed).integers(
        0,
        2**32 - 1,
        size=len(pairs),
        dtype=np.uint64,
    )
    return int(seeds[pair_index])


def pair_base_name(cv: float, freq: float | int) -> str:
    return f"cv{float(cv):.1f}_f{int(freq)}"


def slack_path_format(cv: float, freq: float | int, base_dir: str | Path) -> Path:
    fname = f"Pattern_{float(cv):.1f}cv_2s_{int(freq)}Hz_100uA.txt"
    return Path(base_dir) / fname


def expected_slack_paths(config: TimePatternConfig) -> tuple[Path, ...]:
    config = config.resolved()
    return tuple(slack_path_format(cv, freq, config.slack_dir) for cv, freq in iter_pairs(config))


def expected_output_paths(config: TimePatternConfig) -> tuple[Path, ...]:
    config = config.resolved()
    paths: list[Path] = []
    for cv, freq in iter_pairs(config):
        base = pair_base_name(cv, freq)
        paths.append(config.output_dir / f"{base}_waveforms.npz")
        paths.append(config.output_dir / f"{base}_meta.json")
    paths.append(config.output_dir / "manifest.csv")
    return tuple(paths)


def verify_time_pattern_outputs(config: TimePatternConfig | None = None) -> TimePatternStatus:
    """Check whether stage 01 source inputs and generated outputs are present."""
    config = (config or TimePatternConfig()).resolved()
    input_files = expected_slack_paths(config)
    output_files = expected_output_paths(config)
    missing_inputs = tuple(path for path in input_files if not path.exists())
    missing_outputs = tuple(path for path in output_files if not path.exists())

    return TimePatternStatus(
        input_files=input_files,
        output_files=output_files,
        missing_input_files=missing_inputs,
        missing_output_files=missing_outputs,
        manifest_csv=config.output_dir / "manifest.csv",
    )


def load_slack_waveform_1ua(txt_path: str | Path, spec: StimSpec) -> np.ndarray:
    """Load a Slack waveform and convert 100 uA pulses to 1 uA pulses."""
    arr = np.loadtxt(txt_path, delimiter=",", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Slack txt must have 2 columns. Got shape {arr.shape}.")

    waveform = arr[:, 1].astype(np.float32) * 10_000.0
    if waveform.size != spec.total_samples:
        raise ValueError(
            f"Expected {spec.total_samples} samples in {txt_path}, "
            f"but found {waveform.size}."
        )
    return waveform


def pulse_start_indices_from_times(spike_times_s: np.ndarray, spec: StimSpec) -> np.ndarray:
    """Map continuous pulse times to the exact sample starts used by waveform generation."""
    times = np.asarray(spike_times_s, dtype=float)
    idx = np.floor(times * float(spec.fs)).astype(int)
    return idx[(idx >= 0) & (idx + spec.pulse_samples <= spec.stim_samples)]


def validate_pulse_start_indices(idx: np.ndarray, expected_n: int, spec: StimSpec) -> None:
    """Fail if generated pulses are not exact-count, unique, in-window, and non-overlapping."""
    idx = np.asarray(idx, dtype=int)
    if idx.size != int(expected_n):
        raise ValueError(f"Expected {expected_n} pulses, got {idx.size} after sample-grid placement.")
    if np.unique(idx).size != idx.size:
        raise ValueError("Pulse starts collapsed onto duplicate sample indices.")
    if idx.size and (idx[0] < 0 or idx[-1] + spec.pulse_samples > spec.stim_samples):
        raise ValueError("At least one pulse does not fit fully inside the stimulation window.")
    if idx.size > 1 and np.any(np.diff(idx) < spec.pulse_samples):
        raise ValueError(f"Pulse starts violate the {spec.pulse_samples}-sample minimum IPI.")


def count_negative_phase_onsets(waveform: np.ndarray, eps: float = 0.0) -> int:
    """Count pulse onsets from 0 -> negative current transitions, including sample 0 if present."""
    active_neg = np.asarray(waveform) < -eps
    if active_neg.size == 0:
        return 0
    return int(np.count_nonzero(active_neg & np.r_[True, ~active_neg[:-1]]))


def draw_gamma_times(
    freq: float,
    cv: float,
    duration_s: float,
    rng: np.random.Generator,
    spec: StimSpec,
    first_pulse_sample: int = 1,
    max_tries: int = 2000,
) -> np.ndarray:
    """Draw exactly N = round(freq * duration_s) pulse starts on the waveform grid."""
    if freq <= 0:
        return np.empty(0, float)

    expected_n = int(round(freq * duration_s))
    if expected_n <= 0:
        return np.empty(0, float)

    first_idx = int(first_pulse_sample)
    first_time = first_idx / float(spec.fs)
    last_start_time = (spec.stim_samples - spec.pulse_samples) / float(spec.fs)
    if first_idx < 0 or first_time > last_start_time:
        raise ValueError("First pulse sample is outside the valid stimulation window.")

    if expected_n == 1:
        times = np.array([first_time], dtype=float)
        validate_pulse_start_indices(pulse_start_indices_from_times(times, spec), expected_n, spec)
        return times

    n_ipis = expected_n - 1
    min_ipi_s = spec.min_ipi_s
    available_span = last_start_time - first_time
    min_required_span = n_ipis * min_ipi_s
    if min_required_span > available_span + 1e-12:
        raise ValueError(
            f"Cannot fit {expected_n} pulses at {freq} Hz with {min_ipi_s:.6g}s minimum IPI."
        )

    target_mean_ipi = 1.0 / freq
    k = 1.0 / (cv**2) if cv > 0 else 1e12
    theta = target_mean_ipi / k

    for _ in range(max_tries):
        ipis = rng.gamma(shape=k, scale=theta, size=n_ipis)
        ipis = np.maximum(ipis, min_ipi_s)

        total_span = float(ipis.sum())
        if total_span > available_span:
            excess = ipis - min_ipi_s
            excess_sum = float(excess.sum())
            if excess_sum <= 0:
                continue
            scale = (available_span - min_required_span) / excess_sum
            if scale < 0:
                continue
            ipis = min_ipi_s + excess * scale

        times = first_time + np.r_[0.0, np.cumsum(ipis)]
        idx = pulse_start_indices_from_times(times, spec)
        try:
            validate_pulse_start_indices(idx, expected_n, spec)
        except ValueError:
            continue
        return times

    raise RuntimeError(
        f"Failed to generate valid grid-aligned pulse train for cv={cv}, freq={freq} Hz "
        f"after {max_tries} tries."
    )


def regular_train_times(
    freq: float | int,
    duration_s: float,
    spec: StimSpec | None = None,
    first_pulse_sample: int = 1,
) -> np.ndarray:
    """Generate a strictly regular pulse train for CV=0."""
    if freq <= 0:
        return np.empty(0, dtype=float)

    expected_n = int(round(freq * duration_s))
    if expected_n <= 0:
        return np.empty(0, dtype=float)

    first_time = (int(first_pulse_sample) / float(spec.fs)) if spec is not None else 0.0
    times = first_time + np.arange(expected_n, dtype=float) * (1.0 / freq)

    if spec is not None:
        validate_pulse_start_indices(pulse_start_indices_from_times(times, spec), expected_n, spec)
        return times

    return times[times < duration_s]


def generate_biphasic_waveform(spike_times_s: np.ndarray, spec: StimSpec) -> np.ndarray:
    """Convert pulse times into a sampled biphasic current waveform."""
    waveform = np.zeros(spec.total_samples, dtype=np.float32)
    pulse = np.concatenate(
        [
            np.full(spec.neg_samples, -spec.amp_uA, dtype=np.float32),
            np.zeros(spec.interphase_samples, dtype=np.float32),
            np.full(spec.pos_samples, spec.amp_uA, dtype=np.float32),
        ]
    )

    last_stim_index = spec.stim_samples
    start_indices = pulse_start_indices_from_times(spike_times_s, spec)

    for start in start_indices:
        stop = start + pulse.size
        if stop > last_stim_index:
            raise ValueError("Pulse extends beyond stimulation window; validate times before waveform generation.")
        waveform[start:stop] += pulse

    return waveform


def make_bundle_for_pair(
    cv: float,
    freq: float | int,
    spec: StimSpec,
    slack_base: str | Path,
    seed: int,
) -> tuple[tuple[float, float], dict[str, np.ndarray]]:
    """Generate the Slack repetition plus 99 fixed-count generated repetitions."""
    waveforms: dict[str, np.ndarray] = {}
    slack_file = slack_path_format(cv, freq, slack_base)
    if slack_file.exists():
        waveforms["rep001_slack"] = load_slack_waveform_1ua(slack_file, spec)
    else:
        print(f"[WARN] Slack file missing: {slack_file}")
        waveforms["rep001_slack"] = np.zeros(spec.total_samples, dtype=np.float32)

    rng = np.random.default_rng(seed)
    for rep_idx in range(2, 101):
        if cv == 0.0:
            times = regular_train_times(freq, spec.stim_dur_s, spec=spec, first_pulse_sample=1)
            tag = "regular"
        else:
            times = draw_gamma_times(freq, cv, spec.stim_dur_s, rng, spec, first_pulse_sample=1)
            tag = "gamma"

        waveform = generate_biphasic_waveform(times, spec)
        expected_n = int(round(float(freq) * spec.stim_dur_s))
        realized_n = count_negative_phase_onsets(waveform)
        if realized_n != expected_n:
            raise RuntimeError(
                f"Generated {tag} waveform {rep_idx} for cv={cv}, freq={freq} has "
                f"{realized_n} detected pulses; expected {expected_n}."
            )
        waveforms[f"rep{rep_idx:03d}_{tag}"] = waveform

    gc.collect()
    return (float(cv), float(freq)), waveforms


def save_pair(
    out_dir: str | Path,
    key: tuple[float, float],
    waveforms: dict[str, np.ndarray],
    spec: StimSpec,
) -> dict[str, Path]:
    """Save one pair's waveform bundle and metadata."""
    cv, freq = key
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pair_base_name(cv, freq)
    npz_path = out_dir / f"{base}_waveforms.npz"
    np.savez_compressed(npz_path, **waveforms)

    meta = {
        "cv": float(cv),
        "freq_hz": float(freq),
        "fs": spec.fs,
        "stim_dur_s": spec.stim_dur_s,
        "post_silence_s": spec.post_silence_s,
        "amp_uA": spec.amp_uA,
        "neg_samples": spec.neg_samples,
        "interphase_samples": spec.interphase_samples,
        "pos_samples": spec.pos_samples,
        "pulse_samples": spec.pulse_samples,
        "min_ipi_s": spec.min_ipi_s,
        "generated_first_pulse_sample": 1,
        "generated_exact_pulse_count": True,
        "n_waveforms": len(waveforms),
        "npz_path": display_path(npz_path),
    }

    meta_path = out_dir / f"{base}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"npz": npz_path, "meta": meta_path}


def generate_all_time_patterns(config: TimePatternConfig | None = None) -> dict[str, Path]:
    """Run generation for every configured CV/frequency pair."""
    config = (config or TimePatternConfig()).resolved()
    spec = config.spec
    pairs = list(iter_pairs(config))
    seeds = np.random.default_rng(config.base_seed).integers(
        0,
        2**32 - 1,
        size=len(pairs),
        dtype=np.uint64,
    )

    results: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    if config.parallel:
        max_workers = os.cpu_count() or 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    make_bundle_for_pair,
                    cv,
                    freq,
                    spec,
                    config.slack_dir,
                    int(seed),
                ): (cv, freq)
                for (cv, freq), seed in zip(pairs, seeds)
            }
            for future in as_completed(futures):
                key, bundle = future.result()
                results[key] = bundle
    else:
        for (cv, freq), seed in zip(pairs, seeds):
            key, bundle = make_bundle_for_pair(
                cv,
                freq,
                spec,
                config.slack_dir,
                int(seed),
            )
            results[key] = bundle

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for key, bundle in sorted(results.items()):
        paths = save_pair(config.output_dir, key, bundle, spec)
        cv, freq = key
        manifest_rows.append(
            {
                "cv": cv,
                "freq": freq,
                "npz": display_path(paths["npz"]),
                "meta": display_path(paths["meta"]),
            }
        )

    manifest_csv = config.output_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cv", "freq", "npz", "meta"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {"manifest_csv": manifest_csv, "out_dir": config.output_dir}


def run_stage(
    config: TimePatternConfig | None = None,
    force: bool = False,
) -> TimePatternResult:
    """Verify stage 01 outputs and regenerate them when missing or forced."""
    config = (config or TimePatternConfig()).resolved()
    status = verify_time_pattern_outputs(config)
    if status.missing_input_files:
        preview = "\n".join(str(path) for path in status.missing_input_files[:5])
        raise FileNotFoundError(
            "Stage 01 source Slack files are missing. First missing files:\n" + preview
        )

    if status.outputs_complete and not force:
        return TimePatternResult(
            ran=False,
            status=status,
            manifest_csv=status.manifest_csv,
            output_dir=config.output_dir,
        )

    info = generate_all_time_patterns(config)
    status = verify_time_pattern_outputs(config)
    if not status.outputs_complete:
        preview = "\n".join(str(path) for path in status.missing_output_files[:5])
        raise RuntimeError("Stage 01 generation finished, but outputs are still missing:\n" + preview)

    return TimePatternResult(
        ran=True,
        status=status,
        manifest_csv=info["manifest_csv"],
        output_dir=info["out_dir"],
    )


def load_time_pattern_bundle(
    config: TimePatternConfig | None = None,
    cv: float = 0.5,
    freq: int = 50,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load one generated waveform bundle and its metadata."""
    config = (config or TimePatternConfig()).resolved()
    base = pair_base_name(cv, freq)
    meta_path = config.output_dir / f"{base}_meta.json"
    npz_path = config.output_dir / f"{base}_waveforms.npz"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    with np.load(npz_path) as data:
        waveforms = {key: data[key].copy() for key in data.files}

    return meta, waveforms
