#!/usr/bin/env python3
"""Population periodicity-discrimination observer.

The physical operating-state distribution and perceptual readout are scoped
independently.  The primary manuscript specification is:

* one convex mixture over index regimes per stimulation frequency, shared by
  the std0 and std1 blocks;
* separate rate/timing gains for std0 and std1, shared across frequencies;
* rate and timing evidence at both 1 ms and 10 ms acuity;
* either an ungated hybrid or a delta-CV-gated hybrid readout; and
* a matched per-frequency best-single (one-hot) state comparator.

Mixtures default to a latent-trial interpretation: a trial occupies one
operating regime, the observer produces a regime-specific choice probability,
and those probabilities are averaged using the regime occupancy weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from population.discrimination.behavioral_data import load_slack_dir


DEFAULT_EVIDENCE_CSV = "data/features/population/population_cv0_template_evidence.csv"
DEFAULT_SLACK_DIR = "data/discrimination/slack"
DEFAULT_FREQS = (20.0, 50.0, 100.0, 200.0)
EVIDENCE_COLUMNS = (
    "rate_evidence_1ms",
    "rate_evidence_10ms",
    "timing_evidence_1ms",
    "timing_evidence_10ms",
)
REGIME_ORDER = (
    "I_first", "I_volley5", "I_volley10", "I_5", "I_10",
    "I_15", "I_25", "I_50", "I_all",
)
EPS = 1e-9


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softplus(x: np.ndarray | float) -> np.ndarray:
    return np.logaddexp(0.0, np.asarray(x, dtype=float))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), EPS)


def _sem(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return 0.0 if a.size <= 1 else float(np.std(a, ddof=1) / np.sqrt(a.size))


def _delta_cv(cvs: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    cvs = np.asarray(cvs, dtype=float)
    blocks = np.asarray(blocks, dtype=str)
    return np.where(blocks == "std0", cvs, 1.0 - cvs).clip(0.0, 1.0)


def _map_choose_to_correct(p_choose: np.ndarray, cvs: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    p = np.asarray(p_choose, dtype=float)
    cv = np.asarray(cvs, dtype=float)
    b = np.asarray(blocks, dtype=str)
    out = np.empty_like(p)
    std0 = b == "std0"
    std1 = b == "std1"
    out[std0] = np.where(np.isclose(cv[std0], 0.0), 1.0 - p[std0], p[std0])
    out[std1] = np.where(np.isclose(cv[std1], 1.0), p[std1], 1.0 - p[std1])
    if not np.all(std0 | std1):
        raise ValueError("Behavior blocks must be std0 or std1")
    return out


@dataclass(frozen=True)
class Design:
    freq: np.ndarray
    block: np.ndarray
    cv: np.ndarray
    k: np.ndarray
    n: np.ndarray
    p_observed: np.ndarray
    evidence: np.ndarray  # rows x regimes x four evidence channels


@dataclass(frozen=True)
class ModelSpec:
    frequencies: tuple[float, ...]
    regimes: tuple[str, ...]
    weight_scope: str
    readout_scope: str
    state_mode: str
    model_family: str
    rate_gate_mode: str
    mixture_semantics: str
    loss_reg_l2: float
    weight_kl: float
    rate_gate_reg: float
    rate_gate_prior_floor: float
    rate_gate_prior_center: float
    rate_gate_prior_slope: float
    fixed_selection_n_starts: int
    active_regimes_by_frequency: tuple[tuple[bool, ...], ...]
    canonical_labels_by_frequency: tuple[tuple[str, ...], ...]
    member_labels_by_frequency: tuple[tuple[str, ...], ...]
    weight_entropy_floor_k: float = 0.0
    weight_entropy_floor_lambda: float = 0.0
    weight_max_cap: float = 1.0
    weight_max_cap_lambda: float = 0.0

    @property
    def weight_keys(self) -> tuple[str, ...]:
        if self.weight_scope == "global":
            return ("global",)
        if self.weight_scope == "per_block":
            return ("std0", "std1")
        if self.weight_scope == "per_frequency":
            return tuple(f"freq={int(f)}" for f in self.frequencies)
        return tuple(
            f"freq={int(f)}|block={b}"
            for f in self.frequencies for b in ("std0", "std1")
        )

    @property
    def readout_keys(self) -> tuple[str, ...]:
        if self.readout_scope == "per_block":
            return ("std0", "std1")
        if self.readout_scope == "per_frequency":
            return tuple(f"freq={int(f)}" for f in self.frequencies)
        return tuple(f"freq={int(f)}|block={b}" for f in self.frequencies for b in ("std0", "std1"))

    @property
    def active_channels(self) -> tuple[int, ...]:
        if self.model_family == "rate":
            return (0, 1)
        if self.model_family == "timing":
            return (2, 3)
        return (0, 1, 2, 3)

    @property
    def channels_by_readout(self) -> tuple[tuple[int, ...], ...]:
        if self.model_family == "rate_timing":
            return ((0, 1), (2, 3))  # std0, std1
        if self.model_family == "timing_rate":
            return ((2, 3), (0, 1))
        return tuple(self.active_channels for _ in self.readout_keys)


def _validate_options(spec: ModelSpec) -> None:
    if spec.weight_scope not in {"global", "per_block", "per_frequency", "per_panel"}:
        raise ValueError(
            "weight_scope must be global, per_block, per_frequency, or per_panel"
        )
    if spec.readout_scope not in {"per_block", "per_frequency", "per_panel"}:
        raise ValueError("readout_scope must be per_block, per_frequency, or per_panel")
    if spec.state_mode not in {"mixture", "best_single", "fixed_single"}:
        raise ValueError("state_mode must be mixture, best_single, or fixed_single")
    if spec.weight_scope in {"per_block", "per_panel"} and spec.state_mode != "mixture":
        raise ValueError(
            "per_block and per_panel operating-state scopes are currently defined "
            "only for mixture models"
        )
    if spec.model_family not in {"rate", "timing", "hybrid", "rate_timing", "timing_rate"}:
        raise ValueError("Unknown model_family")
    if spec.model_family in {"rate_timing", "timing_rate"} and (
        spec.weight_scope != "per_frequency" or spec.readout_scope != "per_block"
        or spec.state_mode != "mixture" or spec.mixture_semantics != "latent_trial"
    ):
        raise ValueError("Block cue assignments require per-frequency mixtures, per-block readouts and latent_trial semantics")
    if spec.rate_gate_mode not in {"none", "delta_cv"}:
        raise ValueError("rate_gate_mode must be none or delta_cv")
    if spec.model_family != "hybrid" and spec.rate_gate_mode != "none":
        raise ValueError("The rate gate is defined only for the hybrid family")
    if spec.mixture_semantics not in {"latent_trial", "evidence_average"}:
        raise ValueError("mixture_semantics must be latent_trial or evidence_average")
    expected_shape = (len(spec.frequencies), len(spec.regimes))
    for name, values in (
        ("active_regimes_by_frequency", spec.active_regimes_by_frequency),
        ("canonical_labels_by_frequency", spec.canonical_labels_by_frequency),
        ("member_labels_by_frequency", spec.member_labels_by_frequency),
    ):
        shape = (len(values), len(values[0]) if values else 0)
        if shape != expected_shape or any(len(row) != expected_shape[1] for row in values):
            raise ValueError(f"{name} has shape {shape}; expected {expected_shape}")
    if any(not any(row) for row in spec.active_regimes_by_frequency):
        raise ValueError("Every frequency must retain at least one response-equivalence class")
    if spec.weight_kl < 0.0:
        raise ValueError("weight_kl must be non-negative")
    if spec.weight_entropy_floor_k < 0.0 or spec.weight_entropy_floor_lambda < 0.0:
        raise ValueError("weight entropy-floor settings must be non-negative")
    if not 0.0 < spec.weight_max_cap <= 1.0:
        raise ValueError("weight_max_cap must lie in (0, 1]")
    if spec.weight_max_cap_lambda < 0.0:
        raise ValueError("weight_max_cap_lambda must be non-negative")


def load_population_evidence(path: str, frequencies: Sequence[float]) -> tuple[pd.DataFrame, tuple[str, ...], tuple[int, ...]]:
    required = ["freq", "cv", "rep_index", "thresh_type", *EVIDENCE_COLUMNS]
    equivalence_columns = [
        "canonical_regime", "representative_regime", "regime_active", "member_regimes",
    ]
    header = pd.read_csv(path, nrows=0).columns
    available_equivalence = [column for column in equivalence_columns if column in header]
    data = pd.read_csv(path, usecols=required + available_equivalence)
    data["freq"] = pd.to_numeric(data["freq"], errors="coerce")
    data["cv"] = pd.to_numeric(data["cv"], errors="coerce")
    data["rep_index"] = pd.to_numeric(data["rep_index"], errors="coerce")
    data = data.dropna(subset=["freq", "cv", "rep_index", "thresh_type"]).copy()
    data["rep_index"] = data["rep_index"].astype(int)
    data["thresh_type"] = data["thresh_type"].astype(str)
    data = data[data["freq"].isin([float(f) for f in frequencies])].copy()
    for col in EVIDENCE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    present = set(data["thresh_type"])
    regimes = tuple(r for r in REGIME_ORDER if r in present)
    extras = sorted(present.difference(regimes))
    regimes = regimes + tuple(extras)
    if not regimes:
        raise RuntimeError("No index regimes found in population evidence")
    key = ["freq", "cv", "rep_index", "thresh_type"]
    if data.duplicated(key).any():
        raise RuntimeError("Population evidence has duplicate freq/CV/rep/regime rows")

    # Use only replicates with complete coverage over the requested design.
    required_cells = data[["freq", "cv", "thresh_type"]].drop_duplicates().shape[0]
    counts = data.groupby("rep_index", observed=True).size()
    reps = tuple(sorted(int(r) for r, n in counts.items() if int(n) == int(required_cells)))
    if len(reps) < 2:
        raise RuntimeError("Fewer than two fully covered simulation replicates")
    data = data[data["rep_index"].isin(reps)].copy()
    return data, regimes, reps


def _regime_metadata(
    evidence: pd.DataFrame,
    frequencies: Sequence[float],
    regimes: Sequence[str],
) -> tuple[tuple[tuple[bool, ...], ...], tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Return fitting masks and display labels for response-equivalent states."""
    has_mapping = {
        "canonical_regime", "representative_regime", "regime_active", "member_regimes",
    }.issubset(evidence.columns)
    active_rows: list[tuple[bool, ...]] = []
    canonical_rows: list[tuple[str, ...]] = []
    member_rows: list[tuple[str, ...]] = []
    for freq in frequencies:
        freq_table = evidence[np.isclose(evidence["freq"].astype(float), float(freq))]
        active: list[bool] = []
        canonical: list[str] = []
        members: list[str] = []
        for regime in regimes:
            block = freq_table[freq_table["thresh_type"].astype(str).eq(str(regime))]
            if block.empty:
                raise RuntimeError(f"Missing regime {regime!r} at {float(freq):g} Hz")
            if not has_mapping:
                active.append(True); canonical.append(str(regime)); members.append(str(regime))
                continue
            unique = block[
                ["canonical_regime", "representative_regime", "regime_active", "member_regimes"]
            ].drop_duplicates()
            if len(unique) != 1:
                raise RuntimeError(
                    f"Inconsistent response-equivalence metadata for {regime} at {float(freq):g} Hz"
                )
            row = unique.iloc[0]
            active_value = row["regime_active"]
            if isinstance(active_value, str):
                is_active = active_value.strip().lower() in {"1", "true", "yes"}
            else:
                is_active = bool(active_value)
            active.append(is_active)
            canonical.append(str(row["canonical_regime"]))
            members.append(str(row["member_regimes"]))
        active_rows.append(tuple(active))
        canonical_rows.append(tuple(canonical))
        member_rows.append(tuple(members))
    return tuple(active_rows), tuple(canonical_rows), tuple(member_rows)


def load_behavior(slack_dir: str, frequencies: Sequence[float], evidence_cvs: set[tuple[float, float]]) -> pd.DataFrame:
    _, mean = load_slack_dir(slack_dir, include_templates=True)
    mean = mean[mean["freq"].isin([float(f) for f in frequencies])].copy()
    keep = [(float(f), float(cv)) in evidence_cvs for f, cv in zip(mean["freq"], mean["cv"])]
    mean = mean[np.asarray(keep, dtype=bool)].copy()
    mean = mean.sort_values(["freq", "block", "cv"]).reset_index(drop=True)
    expected = {(float(f), b) for f in frequencies for b in ("std0", "std1")}
    found = set(zip(mean["freq"].astype(float), mean["block"].astype(str)))
    missing = sorted(expected.difference(found))
    if missing:
        raise RuntimeError(f"Behavior is missing requested panels: {missing}")
    return mean


def split_replicates(reps: Sequence[int], train_fraction: float, seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    values = np.asarray(sorted(set(int(r) for r in reps)), dtype=int)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(values)
    n_train = int(round(float(train_fraction) * len(values)))
    n_train = max(1, min(len(values) - 1, n_train))
    return tuple(int(x) for x in values[:n_train]), tuple(int(x) for x in values[n_train:])


def build_design(evidence: pd.DataFrame, behavior: pd.DataFrame, regimes: Sequence[str], reps: Sequence[int]) -> Design:
    subset = evidence[evidence["rep_index"].isin([int(r) for r in reps])]
    means = subset.groupby(["freq", "cv", "thresh_type"], observed=True)[list(EVIDENCE_COLUMNS)].mean()
    tensor = np.zeros((len(behavior), len(regimes), len(EVIDENCE_COLUMNS)), dtype=float)
    for i, row in behavior.iterrows():
        for j, regime in enumerate(regimes):
            try:
                tensor[i, j, :] = means.loc[(float(row["freq"]), float(row["cv"]), str(regime))].to_numpy(float)
            except KeyError as exc:
                raise RuntimeError(
                    f"Missing evidence for freq={row['freq']}, cv={row['cv']}, regime={regime}"
                ) from exc
    return Design(
        freq=behavior["freq"].to_numpy(float),
        block=behavior["block"].astype(str).to_numpy(),
        cv=behavior["cv"].to_numpy(float),
        k=behavior["k"].to_numpy(float),
        n=behavior["n"].to_numpy(float),
        p_observed=behavior["p_mean"].to_numpy(float),
        evidence=tensor,
    )


class PopulationObserverModel:
    def __init__(self, spec: ModelSpec):
        _validate_options(spec)
        self.spec = spec
        self.n_regimes = len(spec.regimes)
        self.weight_keys = spec.weight_keys
        self.readout_keys = spec.readout_keys
        self.channels = spec.active_channels
        self.channel_groups = spec.channels_by_readout
        if spec.weight_scope in {"global", "per_block"}:
            # A named state shared across frequencies can be distinguishable
            # at another frequency even if it is redundant at one frequency.
            # It therefore has to be represented in the shared nominal-state
            # vector. Per-frequency and per-panel vectors can instead use the
            # frequency-specific response-equivalence classes.
            self.active_regime_indices = tuple(
                np.arange(self.n_regimes, dtype=int) for _ in self.weight_keys
            )
        elif spec.weight_scope == "per_frequency":
            self.active_regime_indices = tuple(
                np.flatnonzero(np.asarray(mask, dtype=bool))
                for mask in spec.active_regimes_by_frequency
            )
        else:
            self.active_regime_indices = tuple(
                np.flatnonzero(
                    np.asarray(spec.active_regimes_by_frequency[freq_index], dtype=bool)
                )
                for freq_index, _ in enumerate(spec.frequencies)
                for _block in ("std0", "std1")
            )
        self.n_weight_params = (
            # A K-class softmax needs only K-1 free logits.  The previous
            # K-logit representation had an additive flat direction, which
            # made finite-difference L-BFGS-B fits unnecessarily difficult to
            # terminate formally (especially for per-panel gated hybrids).
            sum(max(len(indices) - 1, 0) for indices in self.active_regime_indices)
            if spec.state_mode == "mixture" else 0
        )
        # Both opposing assignments use two channels in each block. Existing
        # families retain exactly their previous parameter ordering and count.
        self.n_readout_per_group = 1 + len(self.channel_groups[0]) + (3 if spec.rate_gate_mode == "delta_cv" else 0)
        self.n_params = self.n_weight_params + len(self.readout_keys) * self.n_readout_per_group

    def _weight_group_indices(self, design: Design) -> np.ndarray:
        if self.spec.weight_scope == "global":
            return np.zeros(len(design.cv), dtype=int)
        if self.spec.weight_scope == "per_block":
            lookup = {"std0": 0, "std1": 1}
            return np.asarray([lookup[str(b)] for b in design.block], dtype=int)
        if self.spec.weight_scope == "per_frequency":
            lookup = {float(f): i for i, f in enumerate(self.spec.frequencies)}
            return np.asarray([lookup[float(f)] for f in design.freq], dtype=int)
        lookup = {key: i for i, key in enumerate(self.weight_keys)}
        return np.asarray(
            [lookup[f"freq={int(f)}|block={b}"] for f, b in zip(design.freq, design.block)],
            dtype=int,
        )

    def _readout_group_indices(self, design: Design) -> np.ndarray:
        if self.spec.readout_scope == "per_block":
            lookup = {"std0": 0, "std1": 1}
            return np.asarray([lookup[str(b)] for b in design.block], dtype=int)
        if self.spec.readout_scope == "per_frequency":
            lookup = {float(f): i for i, f in enumerate(self.spec.frequencies)}
            return np.asarray([lookup[float(f)] for f in design.freq], dtype=int)
        lookup = {key: i for i, key in enumerate(self.readout_keys)}
        return np.asarray(
            [lookup[f"freq={int(f)}|block={b}"] for f, b in zip(design.freq, design.block)],
            dtype=int,
        )

    def _weight_group_frequency_pairs(self) -> tuple[tuple[int, int], ...]:
        """Map each fitted vector to every frequency where it is applied."""
        if self.spec.weight_scope == "global":
            return tuple((0, freq_index) for freq_index in range(len(self.spec.frequencies)))
        if self.spec.weight_scope == "per_block":
            return tuple(
                (group, freq_index)
                for group in range(len(self.weight_keys))
                for freq_index in range(len(self.spec.frequencies))
            )
        if self.spec.weight_scope == "per_frequency":
            return tuple(
                (freq_index, freq_index)
                for freq_index in range(len(self.spec.frequencies))
            )
        return tuple(
            (group, group // 2) for group in range(len(self.weight_keys))
        )

    def collapsed_weight_views(
        self, weights: np.ndarray
    ) -> tuple[tuple[int, int, tuple[str, ...], np.ndarray], ...]:
        """Return frequency-specific masses after response-equivalence collapse.

        Shared global/per-block parameters remain fitted in nominal-state space,
        because equivalence differs by frequency. For regularization and
        reporting, their masses are summed into the distinct response classes
        at each frequency where the vector is used.
        """
        views = []
        for group, freq_index in self._weight_group_frequency_pairs():
            canonical = self.spec.canonical_labels_by_frequency[freq_index]
            labels = tuple(dict.fromkeys(str(value) for value in canonical))
            label_index = {label: index for index, label in enumerate(labels)}
            collapsed = np.zeros(len(labels), dtype=float)
            for regime_index, label in enumerate(canonical):
                collapsed[label_index[str(label)]] += float(weights[group, regime_index])
            collapsed /= max(float(np.sum(collapsed)), EPS)
            views.append((group, freq_index, labels, collapsed))
        return tuple(views)

    def unpack(self, theta: np.ndarray, choices: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
        theta = np.asarray(theta, dtype=float)
        cursor = 0
        if self.spec.state_mode == "mixture":
            weights = np.zeros((len(self.weight_keys), self.n_regimes), dtype=float)
            for group, active in enumerate(self.active_regime_indices):
                n_free = max(len(active) - 1, 0)
                stop = cursor + n_free
                logits = np.concatenate([theta[cursor:stop], np.zeros(1, dtype=float)])
                weights[group, active] = _softmax(logits)
                cursor = stop
        else:
            if choices is None:
                raise ValueError("single-state mode requires choices")
            weights = np.zeros((len(self.weight_keys), self.n_regimes), dtype=float)
            chosen = np.asarray(choices, dtype=int)
            for group, regime in enumerate(chosen):
                if int(regime) not in set(int(value) for value in self.active_regime_indices[group]):
                    raise ValueError(
                        f"Regime index {regime} is inactive for weight group {self.weight_keys[group]}"
                    )
            weights[np.arange(len(self.weight_keys)), chosen] = 1.0
        readouts = []
        for channels in self.channel_groups:
            bias = float(theta[cursor]); cursor += 1
            raw_gains = theta[cursor: cursor + len(channels)]; cursor += len(channels)
            gains = _softplus(raw_gains)
            gate = {"floor": 1.0, "center": 0.5, "slope": 8.0}
            if self.spec.rate_gate_mode == "delta_cv":
                gate = {
                    "floor": float(_sigmoid(theta[cursor])),
                    "center": float(_sigmoid(theta[cursor + 1])),
                    "slope": float(_softplus(theta[cursor + 2]) + 1e-6),
                }
                cursor += 3
            readouts.append({"bias": bias, "gains": gains, "gate": gate})
        return weights, readouts

    def predict(self, theta: np.ndarray, design: Design, choices: np.ndarray | None = None) -> dict[str, np.ndarray]:
        weights, readouts = self.unpack(theta, choices)
        wg = self._weight_group_indices(design)
        rg = self._readout_group_indices(design)
        delta = _delta_cv(design.cv, design.block)
        n_rows = len(design.cv)
        p_choose = np.zeros(n_rows, dtype=float)
        rate_drive = np.zeros(n_rows, dtype=float)
        timing_drive = np.zeros(n_rows, dtype=float)
        gate_values = np.ones(n_rows, dtype=float)
        effective_rate_share = np.zeros(n_rows, dtype=float)
        for i in range(n_rows):
            ro = readouts[int(rg[i])]
            gains_full = np.zeros(4, dtype=float)
            gains_full[list(self.channel_groups[int(rg[i])])] = ro["gains"]
            gate = 1.0
            if self.spec.rate_gate_mode == "delta_cv":
                g = ro["gate"]
                gate = g["floor"] + (1.0 - g["floor"]) * float(_sigmoid(g["slope"] * (delta[i] - g["center"])))
            gate_values[i] = gate
            regime_rate = gate * (
                gains_full[0] * design.evidence[i, :, 0]
                + gains_full[1] * design.evidence[i, :, 1]
            )
            regime_timing = (
                gains_full[2] * design.evidence[i, :, 2]
                + gains_full[3] * design.evidence[i, :, 3]
            )
            w = weights[int(wg[i])]
            rate_drive[i] = float(w @ regime_rate)
            timing_drive[i] = float(w @ regime_timing)
            denom = float(w @ (np.abs(regime_rate) + np.abs(regime_timing)))
            effective_rate_share[i] = 0.0 if denom <= EPS else float(w @ np.abs(regime_rate)) / denom
            if self.spec.mixture_semantics == "latent_trial":
                p_choose[i] = float(w @ _sigmoid(ro["bias"] + regime_rate + regime_timing))
            else:
                p_choose[i] = float(_sigmoid(ro["bias"] + rate_drive[i] + timing_drive[i]))
        p_correct = _map_choose_to_correct(p_choose, design.cv, design.block).clip(1e-6, 1.0 - 1e-6)
        return {
            "p_choose": p_choose,
            "p_correct": p_correct,
            "rate_drive": rate_drive,
            "timing_drive": timing_drive,
            "gate": gate_values,
            "effective_rate_share": effective_rate_share,
        }

    def objective_components(
        self,
        theta: np.ndarray,
        design: Design,
        choices: np.ndarray | None = None,
    ) -> dict[str, float]:
        pred = self.predict(theta, design, choices)["p_correct"]
        nll = -np.sum(design.k * np.log(pred) + (design.n - design.k) * np.log(1.0 - pred))
        nll /= max(float(np.sum(design.n)), 1.0)
        _, readouts = self.unpack(theta, choices)
        regularized = []
        for readout in readouts:
            regularized.append(float(readout["bias"]))
            regularized.extend(float(x) for x in readout["gains"])
        readout_l2 = self.spec.loss_reg_l2 * float(np.mean(np.square(regularized)))
        weight_kl = 0.0
        entropy_floor = 0.0
        max_weight = 0.0
        if self.spec.state_mode == "mixture" and self.spec.weight_kl > 0.0:
            weights, _ = self.unpack(theta)
            terms = []
            for _group, _freq_index, labels, collapsed in self.collapsed_weight_views(weights):
                w = np.maximum(collapsed, EPS)
                terms.append(float(np.sum(w * np.log(np.maximum(w * len(labels), EPS)))))
            weight_kl = self.spec.weight_kl * float(np.mean(terms))
        if self.spec.state_mode == "mixture" and (
            self.spec.weight_entropy_floor_lambda > 0.0
            or self.spec.weight_max_cap_lambda > 0.0
        ):
            weights, _ = self.unpack(theta)
            entropy_terms: list[float] = []
            cap_terms: list[float] = []
            for _group, _freq_index, labels, collapsed in self.collapsed_weight_views(weights):
                w = np.maximum(collapsed, EPS)
                entropy = -float(np.sum(w * np.log(w)))
                effective_k = float(np.exp(entropy))
                # Response-equivalence collapsing can leave fewer than three
                # independently identifiable classes at a frequency.  Match
                # the manuscript floor when it is attainable, but never
                # penalize a fit for failing an impossible diversity target.
                effective_k_target = min(
                    float(self.spec.weight_entropy_floor_k), float(len(labels))
                )
                entropy_terms.append(
                    max(0.0, effective_k_target - effective_k) ** 2
                )
                over = np.maximum(w - self.spec.weight_max_cap, 0.0)
                cap_terms.append(float(np.sum(np.square(over))))
            if entropy_terms:
                entropy_floor = (
                    self.spec.weight_entropy_floor_lambda * float(np.mean(entropy_terms))
                )
                max_weight = self.spec.weight_max_cap_lambda * float(np.mean(cap_terms))
        rate_gate = 0.0
        if self.spec.rate_gate_mode == "delta_cv" and self.spec.rate_gate_reg > 0.0:
            gate_terms = []
            for readout in readouts:
                gate = readout["gate"]
                gate_terms.extend([
                    gate["floor"] - self.spec.rate_gate_prior_floor,
                    gate["center"] - self.spec.rate_gate_prior_center,
                    (gate["slope"] - self.spec.rate_gate_prior_slope) / max(self.spec.rate_gate_prior_slope, 1.0),
                ])
            rate_gate = self.spec.rate_gate_reg * float(np.mean(np.square(gate_terms)))
        total = float(nll + readout_l2 + weight_kl + entropy_floor + max_weight + rate_gate)
        return {
            "data_nll": float(nll),
            "readout_l2": float(readout_l2),
            "weight_kl": float(weight_kl),
            "weight_entropy_floor": float(entropy_floor),
            "weight_max_cap": float(max_weight),
            "rate_gate": float(rate_gate),
            "total": total,
        }

    def objective(self, theta: np.ndarray, design: Design, choices: np.ndarray | None = None) -> float:
        return self.objective_components(theta, design, choices)["total"]

    def initial_theta(self, rng: np.random.Generator, *, zero_weights: bool = False) -> np.ndarray:
        theta = rng.normal(0.0, 0.15, size=self.n_params)
        if self.n_weight_params and zero_weights:
            theta[: self.n_weight_params] = 0.0
        # Start gates close to ungated, centered at 0.5 with a moderate slope.
        if self.spec.rate_gate_mode == "delta_cv":
            cursor = self.n_weight_params
            for _ in self.readout_keys:
                cursor += 1 + len(self.channels)
                # Start away from the nested ungated boundary so the optimizer
                # can discover a useful gate; the explicit prior can still
                # pull an unnecessary gate back toward floor=1.
                floor = float(np.clip(min(self.spec.rate_gate_prior_floor, 0.8), 1e-4, 1.0 - 1e-4))
                center = float(np.clip(self.spec.rate_gate_prior_center, 1e-4, 1.0 - 1e-4))
                slope = float(max(self.spec.rate_gate_prior_slope, 1e-4))
                theta[cursor] = math.log(floor / (1.0 - floor))
                theta[cursor + 1] = math.log(center / (1.0 - center))
                theta[cursor + 2] = math.log(math.expm1(slope))
                cursor += 3
        return theta

    def parameter_bounds(self) -> list[tuple[float, float]]:
        """Numerically safe bounds with no meaningful restriction on predictions."""
        bounds: list[tuple[float, float]] = [(-20.0, 20.0)] * self.n_weight_params
        for channels in self.channel_groups:
            bounds.append((-20.0, 20.0))  # bias
            bounds.extend([(-20.0, 20.0)] * len(channels))  # softplus gains
            if self.spec.rate_gate_mode == "delta_cv":
                bounds.extend([
                    (-12.0, 12.0),  # floor logit
                    (-12.0, 12.0),  # center logit
                    (-12.0, 20.0),  # softplus slope
                ])
        if len(bounds) != self.n_params:
            raise AssertionError(f"constructed {len(bounds)} bounds for {self.n_params} parameters")
        return bounds


def _fit_rank(fit: tuple[np.ndarray, float, bool]) -> tuple[int, float]:
    """Prefer a formally converged finite fit, then compare objective values."""
    _, loss, success = fit
    finite = bool(np.isfinite(loss))
    return (0 if success and finite else 1 if finite else 2, float(loss) if finite else np.inf)


def _best_fit(
    fits: Iterable[tuple[np.ndarray, float, bool]],
) -> tuple[np.ndarray, float, bool]:
    values = list(fits)
    if not values:
        raise ValueError("At least one optimizer result is required")
    return min(values, key=_fit_rank)


def _optimize_continuous(
    model: PopulationObserverModel,
    design: Design,
    theta0: np.ndarray,
    choices: np.ndarray | None,
) -> tuple[np.ndarray, float, bool]:
    """Fit continuous parameters, retrying only when formal convergence fails.

    Numerical gradients can exhaust SciPy's relatively small default ``maxfun``
    well before ``maxiter`` in the larger per-panel/gated models.  The first
    pass is deliberately accurate; a failed pass is resumed from its best
    finite point with a much larger evaluation budget and standard practical
    tolerances.  Bounds prevent saturated logits from wandering into flat
    numerical regions.
    """
    objective = lambda x: model.objective(x, design, choices)
    bounds = model.parameter_bounds()
    start = np.clip(
        np.asarray(theta0, dtype=float),
        np.asarray([lo for lo, _ in bounds], dtype=float),
        np.asarray([hi for _, hi in bounds], dtype=float),
    )
    attempts: list[tuple[np.ndarray, float, bool]] = []
    schedules = (
        {"maxiter": 2_000, "maxfun": 500_000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 80},
        {"maxiter": 8_000, "maxfun": 2_000_000, "ftol": 1e-8, "gtol": 1e-5, "maxls": 120},
    )
    for options in schedules:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options=options,
        )
        fit = (np.asarray(result.x, dtype=float), float(result.fun), bool(result.success))
        attempts.append(fit)
        if fit[2] and np.isfinite(fit[1]):
            break
        finite = [candidate for candidate in attempts if np.isfinite(candidate[1])]
        start = min(finite, key=lambda candidate: candidate[1])[0] if finite else start
    return _best_fit(attempts)


def fit_mixture(model: PopulationObserverModel, design: Design, seed: int, n_starts: int) -> tuple[np.ndarray, np.ndarray | None, float, bool]:
    rng = np.random.default_rng(seed)
    fits: list[tuple[np.ndarray, float, bool]] = []
    for start in range(max(1, int(n_starts))):
        theta0 = model.initial_theta(rng, zero_weights=(start == 0))
        fits.append(_optimize_continuous(model, design, theta0, None))
    best = _best_fit(fits)
    return best[0], None, best[1], best[2]


def _slice_design(design: Design, mask: np.ndarray) -> Design:
    keep = np.asarray(mask, dtype=bool)
    return Design(
        freq=design.freq[keep], block=design.block[keep], cv=design.cv[keep],
        k=design.k[keep], n=design.n[keep], p_observed=design.p_observed[keep],
        evidence=design.evidence[keep],
    )


def _single_fit_rank(
    fit: tuple[np.ndarray, np.ndarray, float, bool],
) -> tuple[int, float]:
    return _fit_rank((fit[0], fit[2], fit[3]))


def _mapped_shared_regime_start(model: PopulationObserverModel, regime_index: int) -> np.ndarray:
    """Map one original regime name to its active class at every frequency."""
    name = model.spec.regimes[int(regime_index)]
    choices: list[int] = []
    for group, active in enumerate(model.active_regime_indices):
        direct = int(regime_index)
        if direct in set(int(value) for value in active):
            choices.append(direct)
            continue
        mapped = [
            int(index) for index in active
            if name in str(model.spec.member_labels_by_frequency[group][int(index)]).split("|")
        ]
        choices.append(mapped[0] if mapped else int(active[0]))
    return np.asarray(choices, dtype=int)


def _fit_best_single_global(
    model: PopulationObserverModel,
    design: Design,
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    candidates = [int(value) for value in model.active_regime_indices[0]]
    starts_per_candidate = max(1, int(math.ceil(max(1, int(n_starts)) / len(candidates))))
    fits = [
        fit_fixed_single(
            model, design, np.asarray([candidate], dtype=int),
            int(seed) + 10_007 * candidate, starts_per_candidate,
        )
        for candidate in candidates
    ]
    return min(fits, key=_single_fit_rank)


def _fit_best_single_factorized(
    model: PopulationObserverModel,
    design: Design,
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Exactly enumerate each frequency choice when readouts also factor by frequency."""
    total_trials = max(float(np.sum(design.n)), 1.0)
    total_readouts = len(model.readout_keys)
    choices: list[int] = []
    selection_success = True
    total_candidates = sum(len(active) for active in model.active_regime_indices)
    starts_per_candidate = max(1, int(math.ceil(max(1, int(n_starts)) / total_candidates)))

    for freq_index, freq in enumerate(model.spec.frequencies):
        sub_design = _slice_design(design, np.isclose(design.freq, float(freq)))
        sub_trials = max(float(np.sum(sub_design.n)), 1.0)
        sub_readouts = 1 if model.spec.readout_scope == "per_frequency" else 2
        # Multiplying the local objective by sub_trials / total_trials then
        # reproduces this frequency's exact contribution to the full NLL and
        # regularizers.  Hence the independently selected states jointly
        # minimize the full discrete problem.
        penalty_scale = (total_trials / sub_trials) * (sub_readouts / total_readouts)
        sub_spec = replace(
            model.spec,
            frequencies=(float(freq),),
            weight_scope="per_frequency",
            loss_reg_l2=model.spec.loss_reg_l2 * penalty_scale,
            rate_gate_reg=model.spec.rate_gate_reg * penalty_scale,
            active_regimes_by_frequency=(model.spec.active_regimes_by_frequency[freq_index],),
            canonical_labels_by_frequency=(model.spec.canonical_labels_by_frequency[freq_index],),
            member_labels_by_frequency=(model.spec.member_labels_by_frequency[freq_index],),
        )
        sub_model = PopulationObserverModel(sub_spec)
        candidate_fits = [
            fit_fixed_single(
                sub_model, sub_design, np.asarray([int(candidate)], dtype=int),
                int(seed) + 100_003 * freq_index + 1_009 * int(candidate),
                starts_per_candidate,
            )
            for candidate in sub_model.active_regime_indices[0]
        ]
        selected = min(candidate_fits, key=_single_fit_rank)
        choices.append(int(selected[1][0]))
        selection_success = bool(selection_success and selected[3])

    # Refit the complete model so the returned objective is directly
    # comparable to mixture/global-single fits and carries full-model params.
    final = fit_fixed_single(
        model, design, np.asarray(choices, dtype=int), int(seed) + 9_000_001,
        max(2, starts_per_candidate),
    )
    return final[0], final[1], final[2], bool(selection_success and final[3])


def _coupled_single_starts(
    model: PopulationObserverModel,
    rng: np.random.Generator,
    n_starts: int,
) -> list[np.ndarray]:
    baseline = np.asarray([indices[0] for indices in model.active_regime_indices], dtype=int)
    candidates: list[np.ndarray] = []
    # Most important for model nesting: explicitly test every globally shared
    # named regime after mapping frequency-specific duplicate classes.
    candidates.extend(
        _mapped_shared_regime_start(model, regime_index)
        for regime_index in range(model.n_regimes)
    )
    for rank in range(max(len(active) for active in model.active_regime_indices)):
        candidates.append(np.asarray([
            active[min(rank, len(active) - 1)] for active in model.active_regime_indices
        ], dtype=int))
    for group, active in enumerate(model.active_regime_indices):
        for regime in active:
            candidate = baseline.copy()
            candidate[group] = int(regime)
            candidates.append(candidate)
    requested = max(1, int(n_starts))
    while len(candidates) < requested:
        candidates.append(np.asarray([rng.choice(active) for active in model.active_regime_indices], dtype=int))
    unique: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        key = tuple(int(value) for value in candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
        if len(unique) >= requested:
            break
    return unique


def _fit_best_single_coupled(
    model: PopulationObserverModel,
    design: Design,
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Search states coupled by readouts shared across frequencies."""
    rng = np.random.default_rng(seed)
    initial_fits: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
    for choices in _coupled_single_starts(model, rng, n_starts):
        theta, loss, success = _optimize_continuous(
            model, design, model.initial_theta(rng), choices
        )
        initial_fits.append((theta, choices.copy(), loss, success))
    best = min(initial_fits, key=_single_fit_rank)
    theta, choices, _, _ = best

    stable = False
    for _ in range(12):
        old = choices.copy()
        for group, active in enumerate(model.active_regime_indices):
            candidate_fits: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
            for regime in active:
                trial = choices.copy()
                trial[group] = int(regime)
                trial_theta, trial_loss, trial_success = _optimize_continuous(
                    model, design, theta, trial
                )
                candidate_fits.append((trial_theta, trial, trial_loss, trial_success))
            theta, choices, _, _ = min(candidate_fits, key=_single_fit_rank)
        if np.array_equal(old, choices):
            stable = True
            break
    theta, loss, success = _optimize_continuous(model, design, theta, choices)
    return theta, choices.copy(), loss, bool(success and stable)


def fit_best_single(model: PopulationObserverModel, design: Design, seed: int, n_starts: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
    if len(model.weight_keys) == 1:
        return _fit_best_single_global(model, design, seed, n_starts)
    if model.spec.readout_scope in {"per_frequency", "per_panel"}:
        return _fit_best_single_factorized(model, design, seed, n_starts)
    return _fit_best_single_coupled(model, design, seed, n_starts)


def fit_fixed_single(
    model: PopulationObserverModel,
    design: Design,
    choices: np.ndarray,
    seed: int,
    n_starts: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Refit only the readout while retaining a preselected physical state."""
    rng = np.random.default_rng(seed)
    fits: list[tuple[np.ndarray, float, bool]] = []
    for _ in range(max(1, int(n_starts))):
        theta0 = model.initial_theta(rng)
        fits.append(_optimize_continuous(model, design, theta0, choices))
    best = _best_fit(fits)
    return best[0], np.asarray(choices, dtype=int).copy(), best[1], best[2]


def evaluate(model: PopulationObserverModel, theta: np.ndarray, choices: np.ndarray | None, design: Design) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pred = model.predict(theta, design, choices)
    p = pred["p_correct"]
    residual = p - design.p_observed
    nll = -np.sum(design.k * np.log(p) + (design.n - design.k) * np.log(1.0 - p))
    metrics = {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "nll_per_trial": float(nll / max(float(np.sum(design.n)), 1.0)),
    }
    return metrics, pred


def _masked_metrics(design: Design, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    use = np.asarray(mask, dtype=bool)
    p = np.asarray(prediction, dtype=float)[use]
    observed = design.p_observed[use]
    k = design.k[use]
    n = design.n[use]
    residual = p - observed
    nll = -np.sum(k * np.log(p) + (n - k) * np.log(1.0 - p))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "nll_per_trial": float(nll / max(float(np.sum(n)), 1.0)),
    }


def _raw_rate_share(readout: dict, channels: Sequence[int]) -> float:
    gains = np.zeros(4, dtype=float)
    gains[list(channels)] = readout["gains"]
    rate = float(gains[0] + gains[1])
    timing = float(gains[2] + gains[3])
    return 0.0 if rate + timing <= EPS else rate / (rate + timing)


def _configuration_columns(spec: ModelSpec) -> dict:
    return {
        "weight_scope": spec.weight_scope,
        "readout_scope": spec.readout_scope,
        "state_mode": spec.state_mode,
        "model_family": spec.model_family,
        "rate_gate_mode": spec.rate_gate_mode,
        "mixture_semantics": spec.mixture_semantics,
        "loss_reg_l2": spec.loss_reg_l2,
        "weight_kl": spec.weight_kl,
        "rate_gate_reg": spec.rate_gate_reg,
        "rate_gate_prior_floor": spec.rate_gate_prior_floor,
        "rate_gate_prior_center": spec.rate_gate_prior_center,
        "rate_gate_prior_slope": spec.rate_gate_prior_slope,
        "fixed_selection_n_starts": spec.fixed_selection_n_starts,
        "weight_entropy_floor_k": spec.weight_entropy_floor_k,
        "weight_entropy_floor_lambda": spec.weight_entropy_floor_lambda,
        "weight_max_cap": spec.weight_max_cap,
        "weight_max_cap_lambda": spec.weight_max_cap_lambda,
    }


def run_population_observer(
    *,
    evidence_csv: str = DEFAULT_EVIDENCE_CSV,
    slack_dir: str = DEFAULT_SLACK_DIR,
    out_dir: str,
    frequencies: Sequence[float] = DEFAULT_FREQS,
    weight_scope: str = "per_frequency",
    readout_scope: str = "per_panel",
    state_mode: str = "mixture",
    model_family: str = "hybrid",
    rate_gate_mode: str = "none",
    mixture_semantics: str = "latent_trial",
    n_mc: int = 1,
    mc_start: int = 0,
    mc_count: int | None = None,
    train_fraction: float = 0.8,
    seed: int = 0,
    loss_reg_l2: float = 1e-3,
    weight_kl: float = 0.01,
    weight_entropy_floor_k: float = 3.0,
    weight_entropy_floor_lambda: float = 0.01,
    weight_max_cap: float = 0.65,
    weight_max_cap_lambda: float = 0.01,
    rate_gate_reg: float = 0.002,
    rate_gate_prior_floor: float = 1.0,
    rate_gate_prior_center: float = 0.5,
    rate_gate_prior_slope: float = 8.0,
    n_starts: int = 4,
    fixed_selection_n_starts: int = 18,
) -> dict:
    frequencies = tuple(float(f) for f in frequencies)
    evidence, regimes, reps = load_population_evidence(evidence_csv, frequencies)
    active_regimes, canonical_labels, member_labels = _regime_metadata(
        evidence, frequencies, regimes
    )
    evidence_cvs = set(zip(evidence["freq"].astype(float), evidence["cv"].astype(float)))
    behavior = load_behavior(slack_dir, frequencies, evidence_cvs)
    spec = ModelSpec(
        frequencies=frequencies,
        regimes=regimes,
        weight_scope=str(weight_scope),
        readout_scope=str(readout_scope),
        state_mode=str(state_mode),
        model_family=str(model_family),
        rate_gate_mode=str(rate_gate_mode),
        mixture_semantics=str(mixture_semantics),
        loss_reg_l2=float(loss_reg_l2),
        weight_kl=float(weight_kl),
        rate_gate_reg=float(rate_gate_reg),
        rate_gate_prior_floor=float(rate_gate_prior_floor),
        rate_gate_prior_center=float(rate_gate_prior_center),
        rate_gate_prior_slope=float(rate_gate_prior_slope),
        fixed_selection_n_starts=max(1, int(fixed_selection_n_starts)),
        active_regimes_by_frequency=active_regimes,
        canonical_labels_by_frequency=canonical_labels,
        member_labels_by_frequency=member_labels,
        weight_entropy_floor_k=float(weight_entropy_floor_k),
        weight_entropy_floor_lambda=float(weight_entropy_floor_lambda),
        weight_max_cap=float(weight_max_cap),
        weight_max_cap_lambda=float(weight_max_cap_lambda),
    )
    model = PopulationObserverModel(spec)
    count = int(n_mc) if mc_count is None else int(mc_count)
    mc_values = range(int(mc_start), int(mc_start) + count)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        **_configuration_columns(spec),
        "n_parameters": int(model.n_params),
        "n_weight_parameters": int(model.n_weight_params),
        "n_readout_parameters": int(
            len(model.readout_keys) * model.n_readout_per_group
        ),
    }
    result_rows: list[dict] = []
    curve_rows: list[dict] = []
    weight_rows: list[dict] = []
    effective_weight_rows: list[dict] = []
    param_rows: list[dict] = []
    split_rows: list[dict] = []

    fixed_choices: np.ndarray | None = None
    fixed_selection_objective: float | None = None
    fixed_selection_success: bool | None = None
    if state_mode == "fixed_single":
        full_design = build_design(evidence, behavior, regimes, reps)
        _, fixed_choices, fixed_selection_objective, fixed_selection_success = fit_best_single(
            model,
            full_design,
            int(seed) + 8_000_021,
            spec.fixed_selection_n_starts,
        )

    for mc in mc_values:
        split_seed = int(seed) + 100_003 * int(mc)
        train_reps, test_reps = split_replicates(reps, train_fraction, split_seed)
        split_rows.append({**config, "mc": int(mc), "split_seed": split_seed,
                           "train_reps": json.dumps(sorted(train_reps)),
                           "test_reps": json.dumps(sorted(test_reps))})
        train = build_design(evidence, behavior, regimes, train_reps)
        test = build_design(evidence, behavior, regimes, test_reps)
        fit_seed = int(seed) + 1_000_003 * int(mc) + 17
        if state_mode == "mixture":
            theta, choices, objective, success = fit_mixture(model, train, fit_seed, n_starts)
        elif state_mode == "best_single":
            theta, choices, objective, success = fit_best_single(model, train, fit_seed, n_starts)
        else:
            assert fixed_choices is not None
            theta, choices, objective, success = fit_fixed_single(
                model, train, fixed_choices, fit_seed, n_starts
            )
            success = bool(success and fixed_selection_success)
        objective_parts = model.objective_components(theta, train, choices)
        if not np.isclose(float(objective), objective_parts["total"], rtol=1e-9, atol=1e-10):
            raise RuntimeError(
                f"Optimizer objective {objective} disagrees with decomposed objective "
                f"{objective_parts['total']}"
            )
        train_metrics, pred_train = evaluate(model, theta, choices, train)
        test_metrics, pred_test = evaluate(model, theta, choices, test)
        weights, readouts = model.unpack(theta, choices)

        for i, key in enumerate(model.weight_keys):
            if spec.weight_scope == "global":
                freq_value, weight_block = np.nan, "ALL"
            elif spec.weight_scope == "per_block":
                freq_value, weight_block = np.nan, key
            elif spec.weight_scope == "per_frequency":
                freq_value, weight_block = float(key.split("=")[1]), "ALL"
            else:
                freq_part, block_part = key.split("|")
                freq_value = float(freq_part.split("=")[1])
                weight_block = block_part.split("=")[1]
            if spec.weight_scope in {"global", "per_block"}:
                labels = tuple(regimes)
                members = tuple(regimes)
                active = np.ones(len(regimes), dtype=bool)
            else:
                freq_index = frequencies.index(float(freq_value))
                labels = spec.canonical_labels_by_frequency[freq_index]
                members = spec.member_labels_by_frequency[freq_index]
                active = np.asarray(spec.active_regimes_by_frequency[freq_index], dtype=bool)
            active_weights = np.maximum(weights[i, active], EPS)
            active_weights = active_weights / np.sum(active_weights)
            weight_entropy = -float(np.sum(active_weights * np.log(active_weights)))
            effective_regime_count = float(np.exp(weight_entropy))
            effective_regime_target = min(
                float(spec.weight_entropy_floor_k), float(np.sum(active))
            )
            max_regime_weight = float(np.max(active_weights))
            n_weight_ge_0p05 = int(np.sum(active_weights >= 0.05))
            n_weight_ge_0p10 = int(np.sum(active_weights >= 0.10))
            for regime_index, (regime, weight) in enumerate(zip(regimes, weights[i])):
                weight_rows.append({
                    **config, "mc": int(mc), "weight_key": key,
                    "freq": freq_value, "block": weight_block,
                    "thresh_type": regime, "weight": float(weight),
                    "canonical_thresh_type": labels[regime_index],
                    "equivalent_regimes": members[regime_index],
                    "regime_active": bool(active[regime_index]),
                    "selected": bool(weight > 1.0 - 1e-8),
                    "weight_entropy": weight_entropy,
                    "effective_regime_count": effective_regime_count,
                    "effective_regime_target": effective_regime_target,
                    "max_regime_weight": max_regime_weight,
                    "n_weight_ge_0p05": n_weight_ge_0p05,
                    "n_weight_ge_0p10": n_weight_ge_0p10,
                })

        for group, freq_index, labels, collapsed in model.collapsed_weight_views(weights):
            key = model.weight_keys[group]
            freq_value = float(frequencies[freq_index])
            if spec.weight_scope in {"global", "per_frequency"}:
                weight_block = "ALL"
            elif spec.weight_scope == "per_block":
                weight_block = key
            else:
                weight_block = key.split("block=")[1]
            entropy = -float(np.sum(np.maximum(collapsed, EPS) * np.log(np.maximum(collapsed, EPS))))
            effective_k = float(np.exp(entropy))
            effective_k_target = min(float(spec.weight_entropy_floor_k), float(len(labels)))
            max_weight = float(np.max(collapsed))
            canonical_at_frequency = spec.canonical_labels_by_frequency[freq_index]
            for label, weight in zip(labels, collapsed):
                source_regimes = "|".join(
                    str(regime) for regime, canonical_label in zip(regimes, canonical_at_frequency)
                    if str(canonical_label) == str(label)
                )
                effective_weight_rows.append({
                    **config,
                    "mc": int(mc),
                    "weight_key": key,
                    "freq": freq_value,
                    "block": weight_block,
                    "canonical_thresh_type": str(label),
                    "equivalent_regimes": source_regimes,
                    "weight": float(weight),
                    "weight_entropy": entropy,
                    "effective_regime_count": effective_k,
                    "effective_regime_target": effective_k_target,
                    "max_regime_weight": max_weight,
                    "n_weight_ge_0p05": int(np.sum(collapsed >= 0.05)),
                    "n_weight_ge_0p10": int(np.sum(collapsed >= 0.10)),
                })

        for i, key in enumerate(model.readout_keys):
            ro = readouts[i]
            gains = np.zeros(4, dtype=float)
            gains[list(model.channel_groups[i])] = ro["gains"]
            if spec.readout_scope == "per_block":
                readout_freq, readout_block = np.nan, key
            elif spec.readout_scope == "per_frequency":
                readout_freq = float(key.split("=")[1])
                readout_block = "ALL"
            else:
                freq_part, block_part = key.split("|")
                readout_freq = float(freq_part.split("=")[1])
                readout_block = block_part.split("=")[1]
            param_rows.append({
                **config, "mc": int(mc), "readout_key": key,
                "freq": readout_freq, "block": readout_block,
                "bias": float(ro["bias"]),
                "gain_rate_1ms": float(gains[0]), "gain_rate_10ms": float(gains[1]),
                "gain_timing_1ms": float(gains[2]), "gain_timing_10ms": float(gains[3]),
                "raw_rate_share": _raw_rate_share(ro, model.channel_groups[i]),
                "rate_gate_floor": float(ro["gate"]["floor"]),
                "rate_gate_center": float(ro["gate"]["center"]),
                "rate_gate_slope": float(ro["gate"]["slope"]),
            })

        for split, design_split, metrics, pred in (
            ("train", train, train_metrics, pred_train),
            ("test", test, test_metrics, pred_test),
        ):
            common = {
                **config, "mc": int(mc), "split": split,
                "n_train_reps": len(train_reps), "n_test_reps": len(test_reps),
                "objective_train": float(objective), "optimization_success": bool(success),
                "data_nll_train": objective_parts["data_nll"],
                "readout_l2_penalty": objective_parts["readout_l2"],
                "weight_kl_penalty": objective_parts["weight_kl"],
                "weight_entropy_floor_penalty": objective_parts["weight_entropy_floor"],
                "weight_max_cap_penalty": objective_parts["weight_max_cap"],
                "rate_gate_penalty": objective_parts["rate_gate"],
                "fixed_selection_objective": fixed_selection_objective,
                "fixed_selection_success": fixed_selection_success,
            }
            result_rows.append({
                **common, "freq": "ALL", "block": "ALL",
                "rmse": metrics["rmse"], "mae": metrics["mae"],
                "nll_per_trial": metrics["nll_per_trial"],
            })
            for freq in frequencies:
                for block in ("std0", "std1"):
                    mask = np.isclose(design_split.freq, float(freq)) & (design_split.block == block)
                    panel = _masked_metrics(design_split, pred["p_correct"], mask)
                    result_rows.append({
                        **common, "freq": float(freq), "block": block,
                        "rmse": panel["rmse"], "mae": panel["mae"],
                        "nll_per_trial": panel["nll_per_trial"],
                    })

        for i in range(len(behavior)):
            for split, pred in (("train", pred_train), ("test", pred_test)):
                curve_rows.append({
                    **config, "mc": int(mc), "split": split,
                    "freq": float(behavior.iloc[i]["freq"]),
                    "block": str(behavior.iloc[i]["block"]),
                    "cv": float(behavior.iloc[i]["cv"]),
                    "p_observed": float(behavior.iloc[i]["p_mean"]),
                    "p_observed_sem": float(behavior.iloc[i]["p_sem"]),
                    "p_predicted": float(pred["p_correct"][i]),
                    "rate_drive": float(pred["rate_drive"][i]),
                    "timing_drive": float(pred["timing_drive"][i]),
                    "gate": float(pred["gate"][i]),
                    "effective_rate_share": float(pred["effective_rate_share"][i]),
                })
        print(
            f"[population observer] mc={mc} model={model_family}/{rate_gate_mode} "
            f"state={state_mode} train_rmse={train_metrics['rmse']:.5f} "
            f"test_rmse={test_metrics['rmse']:.5f}",
            flush=True,
        )

    pd.DataFrame(result_rows).to_csv(out / "results.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "weights.csv", index=False)
    pd.DataFrame(effective_weight_rows).to_csv(out / "effective_weights.csv", index=False)
    with (out / "params.jsonl").open("w", encoding="utf-8") as handle:
        for row in param_rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "analysis": "population_periodicity_discrimination_v4_manuscript_regularized",
        "evidence_csv": str(Path(evidence_csv).resolve()),
        "slack_dir": str(Path(slack_dir).resolve()),
        "frequencies": list(frequencies),
        "regimes": list(regimes),
        "replicates_available": len(reps),
        "mc_start": int(mc_start), "mc_count": count,
        "train_fraction": float(train_fraction), "seed": int(seed),
        **config,
        "response_equivalence_applied": "canonical_regime" in evidence.columns,
        "mixture_regularization_basis": "frequency_specific_response_equivalent_classes",
        "fixed_selection_basis": "all_simulation_replicates" if state_mode == "fixed_single" else None,
        "fixed_choices": None if fixed_choices is None else [
            str(regimes[int(index)]) for index in fixed_choices
        ],
        "fixed_choice_canonical_by_frequency": None if fixed_choices is None else {
            str(int(float(freq))): spec.canonical_labels_by_frequency[freq_index][
                int(fixed_choices[0] if spec.weight_scope == "global" else fixed_choices[freq_index])
            ]
            for freq_index, freq in enumerate(frequencies)
        },
        "fixed_selection_objective": fixed_selection_objective,
        "fixed_selection_success": fixed_selection_success,
    }
    manifest["channels_by_readout"] = dict(zip(model.readout_keys, model.channel_groups))
    pd.DataFrame(split_rows).to_csv(out / "split_membership.csv", index=False)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def preflight(evidence_csv: str, slack_dir: str, frequencies: Sequence[float]) -> dict:
    evidence, regimes, reps = load_population_evidence(evidence_csv, frequencies)
    active, canonical, members = _regime_metadata(evidence, frequencies, regimes)
    behavior = load_behavior(
        slack_dir,
        frequencies,
        set(zip(evidence["freq"].astype(float), evidence["cv"].astype(float))),
    )
    return {
        "status": "ready",
        "evidence_csv": str(Path(evidence_csv).resolve()),
        "evidence_rows": int(len(evidence)),
        "frequencies": [float(f) for f in frequencies],
        "regimes": list(regimes),
        "response_equivalence_applied": "canonical_regime" in evidence.columns,
        "active_regime_classes_by_frequency": {
            str(int(float(freq))): [
                canonical[freq_index][regime_index]
                for regime_index, is_active in enumerate(active[freq_index]) if is_active
            ]
            for freq_index, freq in enumerate(frequencies)
        },
        "simulation_replicates": len(reps),
        "behavior_rows": int(len(behavior)),
        "behavior_panels": int(behavior[["freq", "block"]].drop_duplicates().shape[0]),
    }


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-csv", default=DEFAULT_EVIDENCE_CSV)
    parser.add_argument("--slack-dir", default=DEFAULT_SLACK_DIR)
    parser.add_argument("--out-dir", default="results/population/discrimination/smoke")
    parser.add_argument("--frequencies", default="20,50,100,200")
    parser.add_argument(
        "--weight-scope",
        default="per_frequency",
        choices=["global", "per_block", "per_frequency", "per_panel"],
    )
    parser.add_argument(
        "--readout-scope",
        default="per_panel",
        choices=["per_block", "per_frequency", "per_panel"],
    )
    parser.add_argument(
        "--state-mode",
        default="mixture",
        choices=["mixture", "best_single", "fixed_single"],
    )
    parser.add_argument("--model-family", default="hybrid", choices=["rate", "timing", "hybrid", "rate_timing", "timing_rate"])
    parser.add_argument("--rate-gate-mode", default="none", choices=["none", "delta_cv"])
    parser.add_argument("--mixture-semantics", default="latent_trial", choices=["latent_trial", "evidence_average"])
    parser.add_argument("--n-mc", type=int, default=1)
    parser.add_argument("--mc-start", type=int, default=0)
    parser.add_argument("--mc-count", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-reg-l2", type=float, default=1e-3)
    parser.add_argument("--weight-kl", type=float, default=0.01)
    parser.add_argument("--weight-entropy-floor-k", type=float, default=3.0)
    parser.add_argument("--weight-entropy-floor-lambda", type=float, default=0.01)
    parser.add_argument("--weight-max-cap", type=float, default=0.65)
    parser.add_argument("--weight-max-cap-lambda", type=float, default=0.01)
    parser.add_argument("--rate-gate-reg", type=float, default=0.002)
    parser.add_argument("--rate-gate-prior-floor", type=float, default=1.0)
    parser.add_argument("--rate-gate-prior-center", type=float, default=0.5)
    parser.add_argument("--rate-gate-prior-slope", type=float, default=8.0)
    parser.add_argument("--n-starts", type=int, default=4)
    parser.add_argument("--fixed-selection-n-starts", type=int, default=18)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    freqs = _parse_csv_floats(args.frequencies)
    if args.preflight:
        print(json.dumps(preflight(args.evidence_csv, args.slack_dir, freqs), indent=2))
        return 0
    run_population_observer(
        evidence_csv=args.evidence_csv, slack_dir=args.slack_dir, out_dir=args.out_dir,
        frequencies=freqs, weight_scope=args.weight_scope, readout_scope=args.readout_scope,
        state_mode=args.state_mode, model_family=args.model_family,
        rate_gate_mode=args.rate_gate_mode, mixture_semantics=args.mixture_semantics,
        n_mc=args.n_mc, mc_start=args.mc_start, mc_count=args.mc_count,
        train_fraction=args.train_fraction, seed=args.seed, loss_reg_l2=args.loss_reg_l2,
        weight_kl=args.weight_kl,
        weight_entropy_floor_k=args.weight_entropy_floor_k,
        weight_entropy_floor_lambda=args.weight_entropy_floor_lambda,
        weight_max_cap=args.weight_max_cap,
        weight_max_cap_lambda=args.weight_max_cap_lambda,
        rate_gate_reg=args.rate_gate_reg,
        rate_gate_prior_floor=args.rate_gate_prior_floor,
        rate_gate_prior_center=args.rate_gate_prior_center,
        rate_gate_prior_slope=args.rate_gate_prior_slope,
        n_starts=args.n_starts,
        fixed_selection_n_starts=args.fixed_selection_n_starts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
