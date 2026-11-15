"""Conditional cue controls with split-specific primary-hybrid mixtures frozen.

This is an isolated runner: production fitting code and existing jobs are untouched.
Historical input hashes were not saved. Reconstructed train/test predictions and
objectives must match the saved primary fit before any conditional optimization.
No AIC/BIC is reported for the data-derived, inherited mixtures.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from discrimination import discrimination_core as core
from discrimination.block_cue_followup import digest

FAMILIES = ("rate_timing", "timing_rate", "rate", "timing", "hybrid")
REFERENCE = "primary_hybrid_reference"
FREQS = (20., 50., 100., 200.)
GAINS = ("gain_rate_1ms", "gain_rate_10ms", "gain_timing_1ms", "gain_timing_10ms")
SOURCE = "data/discrimination/scope"
EVIDENCE = "data/features/population_cv0_template_evidence.csv"
STEMS = ("diagnostics", "results", "curves", "params", "weights", "splits", "baseline_audit")


def primary(frame):
    return frame.loc[(frame.weight_scope == "per_frequency") &
                     (frame.readout_scope == "per_block") & (frame.state_mode == "mixture") &
                     (frame.model_family == "hybrid") & (frame.rate_gate_mode == "none") &
                     (frame.mixture_semantics == "latent_trial")].copy()


class LockedMixtureModel(core.PopulationObserverModel):
    """Expose only readout parameters to the unchanged core optimizer."""
    def __init__(self, spec, weights):
        if (spec.weight_scope, spec.readout_scope, spec.state_mode,
                spec.rate_gate_mode, spec.mixture_semantics) != (
                "per_frequency", "per_block", "mixture", "none", "latent_trial"):
            raise ValueError("Locked controls require the primary ungated sharing structure")
        super().__init__(spec)
        w = np.asarray(weights, float).copy()
        active = np.asarray(spec.active_regimes_by_frequency, bool)
        if (w.shape != active.shape or not np.isfinite(w).all() or (w < 0).any()
                or not np.allclose(w.sum(axis=1), 1., atol=1e-12, rtol=0)
                or np.any(w[~active] != 0)):
            raise ValueError("Invalid locked mixture or nonzero inactive regime")
        self.locked_weights = w
        self.locked_weights.setflags(write=False)
        self.inherited_weight_parameters = self.n_weight_params
        self.n_params -= self.n_weight_params
        self.n_weight_params = 0

    def unpack(self, theta, choices=None):
        if len(theta) != self.n_params:
            raise ValueError("Wrong conditional readout dimension")
        # Core unpack uses the active-class shapes to skip these dummy logits.
        _, readouts = super().unpack(np.r_[np.zeros(self.inherited_weight_parameters), theta])
        return self.locked_weights, readouts


def readout_start(model, params):
    """Project saved hybrid gains onto each permitted cue group, without refitting."""
    rows = params.set_index("block")
    theta = []
    for block, channels in zip(model.readout_keys, model.channel_groups):
        ro = rows.loc[block]
        theta.append(float(ro.bias))
        gains = ro[list(GAINS)].to_numpy(float)[list(channels)]
        if not np.isfinite(gains).all() or (gains <= 0).any():
            raise ValueError("Invalid source gains")
        theta.extend(gains + np.log(-np.expm1(-gains)))  # inverse softplus
    return np.asarray(theta)


def conditional_fit(model, train, source_params, seed, n_starts=6):
    rng = np.random.default_rng(seed)
    starts = [readout_start(model, source_params)]
    starts += [model.initial_theta(rng) for _ in range(n_starts - 1)]
    return core._best_fit(core._optimize_continuous(model, train, start, None) for start in starts)


def load_inputs(root):
    evidence, regimes, reps = core.load_population_evidence(str(root / EVIDENCE), FREQS)
    behavior = core.load_behavior(str(root / "data/discrimination/slack"), FREQS,
                                  set(zip(evidence.freq, evidence.cv)))
    if len(reps) != 100 or len(behavior) != 88:
        raise ValueError("Expected 100 simulated repetitions and 88 behavioral conditions")
    return evidence, regimes, reps, behavior


def make_spec(evidence, regimes, source_row):
    active, canonical, members = core._regime_metadata(evidence, FREQS, regimes)
    kwargs = {key: float(source_row[key]) for key in (
        "loss_reg_l2", "weight_kl", "rate_gate_reg", "rate_gate_prior_floor",
        "rate_gate_prior_center", "rate_gate_prior_slope", "weight_entropy_floor_k",
        "weight_entropy_floor_lambda", "weight_max_cap", "weight_max_cap_lambda")}
    return core.ModelSpec(FREQS, tuple(regimes), "per_frequency", "per_block", "mixture",
                          "hybrid", "none", "latent_trial", **kwargs,
                          fixed_selection_n_starts=int(source_row.fixed_selection_n_starts),
                          active_regimes_by_frequency=active,
                          canonical_labels_by_frequency=canonical,
                          member_labels_by_frequency=members)


def verify(root, bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for rel, expected in manifest["input_code_sha256"].items():
        if digest(root / rel) != expected:
            raise ValueError(f"Changed code/input: {rel}")
    for rel, expected in manifest["bundle_sha256"].items():
        if digest(bundle / rel) != expected:
            raise ValueError(f"Changed source bundle: {rel}")
    return manifest


def prepare(root, bundle):
    if (bundle / "manifest.json").exists():
        verify(root, bundle)
        print("Existing locked-mixture bundle verified; not replaced.")
        return
    if bundle.exists() and any(bundle.iterdir()):
        raise ValueError("Nonempty incomplete bundle; inspect before preparing again")
    source = root / SOURCE
    valid = json.loads((source / "validation.json").read_text())
    if valid.get("status") != "ready" or valid.get("failed_observer_fits") != 0:
        raise ValueError("Source fits are not validated")
    evidence, regimes, reps, behavior = load_inputs(root)
    frames = {stem: primary(pd.read_csv(source / f"merged_{stem}.csv", low_memory=False))
              for stem in ("weights", "params", "curves", "results")}
    keys = {"weights": ["mc", "freq", "thresh_type"], "params": ["mc", "block"],
            "curves": ["mc", "split", "freq", "block", "cv"],
            "results": ["mc", "split", "freq", "block"]}
    counts = {"weights": 36, "params": 2, "curves": 176, "results": 18}
    for stem, frame in frames.items():
        if (set(frame.mc) != set(range(200)) or frame.duplicated(keys[stem]).any()
                or not frame.groupby("mc").size().eq(counts[stem]).all()):
            raise ValueError(f"Incomplete/duplicate source {stem}")
    results = frames["results"]
    if (not results.optimization_success.astype(str).str.lower().eq("true").all()
            or not results.n_train_reps.eq(80).all() or not results.n_test_reps.eq(20).all()):
        raise ValueError("Source convergence/splits mismatch")
    spec = make_spec(evidence, regimes, results.iloc[0])
    config = core._configuration_columns(spec)
    for frame in frames.values():
        for key, value in config.items():
            if not frame[key].eq(value).all():
                raise ValueError(f"Mixed source configuration: {key}")
    if core.PopulationObserverModel(spec).n_weight_params != 23:
        raise ValueError("Changed response-equivalence classes")
    bundle.mkdir(parents=True, exist_ok=True)
    for stem, frame in frames.items():
        frame.to_csv(bundle / f"source_{stem}.csv", index=False)
    files = [root / EVIDENCE, *sorted((root / "data/discrimination/slack").glob("*.csv")),
             *[root / p for p in (
                 "src/discrimination/locked_mixture_followup.py",
                 "src/discrimination/discrimination_core.py",
                 "src/discrimination/block_cue_followup.py",
                 "src/discrimination/behavioral_data.py")]]
    manifest = dict(version=1, source=SOURCE, families=list(FAMILIES), mc_total=200,
                    seed=0, train_fraction=.8, n_starts=6, regimes=list(regimes),
                    source_configuration=config, inherited_weight_parameters=23,
                    historical_provenance="Historical hashes unavailable; every MC must reconstruct both saved split predictions and training objective",
                    validation_design="Held-out simulated repetitions, not held-out rats",
                    information_criteria="Not reported: mixture was estimated from the same behavioral data",
                    input_code_sha256={str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in files},
                    original_source_sha256={p.name: digest(p) for p in
                        [source / f"merged_{s}.csv" for s in frames] + [source / "validation.json"]},
                    bundle_sha256={p.name: digest(p) for p in sorted(bundle.glob("source_*.csv"))})
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {bundle}: 200 split-specific mixtures, five conditional configurations.")


def audit_source(model, theta, designs, curves, results):
    gaps = []
    for split, design in designs.items():
        saved = curves[curves.split == split].sort_values(["freq", "block", "cv"])
        if len(saved) != len(design.cv) or not np.allclose(saved.cv, design.cv) or not np.allclose(saved.freq, design.freq) or not np.array_equal(saved.block, design.block):
            raise ValueError("Historical curve keys do not match reconstructed design")
        pred = model.predict(theta, design)
        for field, stored in [("p_correct", "p_predicted"), ("rate_drive", "rate_drive"),
                              ("timing_drive", "timing_drive"), ("effective_rate_share", "effective_rate_share")]:
            gaps.append(float(np.max(np.abs(pred[field] - saved[stored].to_numpy()))))
        gaps.append(float(np.max(np.abs(design.p_observed - saved.p_observed.to_numpy()))))
    objective = model.objective(theta, designs["train"])
    gap = abs(objective - float(results.objective_train.iloc[0]))
    if max(gaps) > 1e-8 or gap > 1e-8:
        raise ValueError(f"Historical reconstruction failed: prediction gap={max(gaps):g}, objective gap={gap:g}")
    return dict(max_prediction_gap=max(gaps), objective_gap=gap, source_objective=objective)


def collect(model, theta, name, mc, designs, rows):
    common = dict(model=name, mc=mc)
    for split, design in designs.items():
        pred = model.predict(theta, design)
        for i in range(len(design.cv)):
            rows["curves"].append(dict(**common, split=split, freq=design.freq[i], block=design.block[i],
                cv=design.cv[i], p_observed=design.p_observed[i], p_predicted=pred["p_correct"][i],
                rate_drive=pred["rate_drive"][i], timing_drive=pred["timing_drive"][i],
                effective_rate_share=pred["effective_rate_share"][i]))
        scopes = [("ALL", "ALL", np.ones(len(design.cv), bool))]
        scopes += [("ALL", b, design.block == b) for b in ("std0", "std1")]
        scopes += [(f, b, (design.freq == f) & (design.block == b)) for f in FREQS for b in ("std0", "std1")]
        for freq, block, mask in scopes:
            rows["results"].append(dict(**common, split=split, freq=freq, block=block,
                **core._masked_metrics(design, pred["p_correct"], mask)))
    _, readouts = model.unpack(theta)
    for block, channels, ro in zip(model.readout_keys, model.channel_groups, readouts):
        gains = np.zeros(4)
        gains[list(channels)] = ro["gains"]
        rows["params"].append(dict(**common, block=block, bias=ro["bias"], **dict(zip(GAINS, gains))))


def fit(root, bundle, out, mc):
    manifest = verify(root, bundle)
    if mc not in range(manifest["mc_total"]):
        raise ValueError("MC must be 0..199")
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing partial: {out}")
    evidence, regimes, reps, behavior = load_inputs(root)
    source = {s: pd.read_csv(bundle / f"source_{s}.csv", low_memory=False) for s in ("weights", "params", "curves", "results")}
    source = {s: frame[frame.mc == mc].copy() for s, frame in source.items()}
    spec = make_spec(evidence, regimes, source["results"].iloc[0])
    weights = source["weights"].pivot(index="freq", columns="thresh_type", values="weight").loc[list(FREQS), list(regimes)].to_numpy()
    split_seed = manifest["seed"] + 100003 * mc
    train_reps, test_reps = core.split_replicates(reps, manifest["train_fraction"], split_seed)
    designs = {s: core.build_design(evidence, behavior, regimes, subset)
               for s, subset in [("train", train_reps), ("test", test_reps)]}
    reference = LockedMixtureModel(spec, weights)
    source_theta = readout_start(reference, source["params"])
    audit = audit_source(reference, source_theta, designs, source["curves"], source["results"])
    rows = {s: [] for s in STEMS}
    rows["baseline_audit"].append(dict(mc=mc, **audit))
    rows["splits"].append(dict(mc=mc, split_seed=split_seed, train_reps=json.dumps(sorted(train_reps)), test_reps=json.dumps(sorted(test_reps))))
    rows["weights"] = source["weights"][["mc", "freq", "thresh_type", "weight"]].to_dict("records")
    collect(reference, source_theta, REFERENCE, mc, designs, rows)
    for family in FAMILIES:
        model = LockedMixtureModel(replace(spec, model_family=family), weights)
        theta, loss, success = conditional_fit(model, designs["train"], source["params"],
                                              manifest["seed"] + 1000003 * mc + 17, manifest["n_starts"])
        if not success or not np.isfinite(loss):
            raise RuntimeError(f"Conditional fit failed: mc={mc} {family}")
        if family == "hybrid" and loss > audit["source_objective"] + 1e-7:
            raise RuntimeError("Hybrid refit is worse than its feasible source start")
        parts = model.objective_components(theta, designs["train"])
        rows["diagnostics"].append(dict(mc=mc, model=family, success=success, objective=loss,
            optimized_readout_parameters=model.n_params, inherited_weight_parameters=model.inherited_weight_parameters,
            optimized_weight_parameters=0, n_starts=manifest["n_starts"],
            max_locked_weight_change=float(np.max(np.abs(model.unpack(theta)[0] - weights))),
            **{f"objective_{k}": v for k, v in parts.items()}))
        collect(model, theta, family, mc, designs, rows)
        print(f"mc={mc} locked {family}: objective={loss:.8f}", flush=True)
    out.mkdir(parents=True)
    for stem, records in rows.items():
        pd.DataFrame(records).to_csv(out / f"{stem}.csv", index=False)
    (out / "complete.json").write_text(json.dumps(dict(mc=mc, families=list(FAMILIES),
        manifest_sha256=digest(bundle / "manifest.json")), indent=2), encoding="utf-8")


def merge(root, bundle, partials, out, expected_mc=200):
    manifest = verify(root, bundle)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status="failed", expected_mc=expected_mc, errors=[])
    try:
        paths = sorted(partials.glob("mc_*"))
        if {p.name for p in paths} != {f"mc_{i}" for i in range(expected_mc)}:
            raise ValueError("Missing/extra MC partial directories")
        for p in paths:
            completion = json.loads((p / "complete.json").read_text())
            if completion != dict(mc=int(p.name[3:]), families=list(FAMILIES), manifest_sha256=digest(bundle / "manifest.json")):
                raise ValueError(f"Incomplete or mismatched partial: {p}")
        frames = {stem: pd.concat([pd.read_csv(p / f"{stem}.csv") for p in paths], ignore_index=True) for stem in STEMS}
        validate_frames(frames, bundle, expected_mc)
        for stem, frame in frames.items():
            frame.to_csv(out / f"merged_{stem}.csv", index=False)
        result = frames["results"]
        keys = ["mc", "split", "freq", "block"]
        pairs = result[result.model.isin(FAMILIES)].copy()
        for baseline in [REFERENCE, "hybrid"]:
            ref = result[result.model == baseline][keys + ["rmse", "nll_per_trial"]]
            ref = ref.rename(columns={m: f"{baseline}_{m}" for m in ["rmse", "nll_per_trial"]})
            pairs = pairs.merge(ref, on=keys, validate="many_to_one")
            for metric in ["rmse", "nll_per_trial"]:
                pairs[f"delta_{metric}_vs_{baseline}"] = pairs[metric] - pairs[f"{baseline}_{metric}"]
        pairs.to_csv(out / "paired_performance.csv", index=False)
        summary = result.groupby(["model", "split", "freq", "block"], dropna=False)[["rmse", "nll_per_trial"]].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.reset_index().to_csv(out / "performance_summary.csv", index=False)
        report.update(status="ready", conditional_fits=len(frames["diagnostics"]),
                      failed_fits=0, source_reconstructions=len(frames["baseline_audit"]),
                      max_prediction_reconstruction_gap=float(frames["baseline_audit"].max_prediction_gap.max()),
                      max_locked_weight_change=float(frames["diagnostics"].max_locked_weight_change.max()))
        (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        report["errors"].append(str(exc))
        raise
    finally:
        (out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


def validate_frames(frames, bundle, expected_mc):
    models = set(FAMILIES) | {REFERENCE}
    per_mc = dict(diagnostics=5, results=132, curves=1056, params=12, weights=36, splits=1, baseline_audit=1)
    keys = dict(diagnostics=["model"], results=["model", "split", "freq", "block"],
                curves=["model", "split", "freq", "block", "cv"], params=["model", "block"],
                weights=["freq", "thresh_type"], splits=[], baseline_audit=[])
    for stem, frame in frames.items():
        if (set(frame.mc) != set(range(expected_mc)) or frame.duplicated(["mc"] + keys[stem]).any()
                or not frame.groupby("mc").size().eq(per_mc[stem]).all()):
            raise ValueError(f"Incomplete/duplicate {stem}")
        if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError(f"Nonfinite {stem}")
    d = frames["diagnostics"]
    if (set(d.model) != set(FAMILIES) or not d.success.astype(str).str.lower().eq("true").all()
            or not d.optimized_weight_parameters.eq(0).all() or not d.inherited_weight_parameters.eq(23).all()
            or not d.max_locked_weight_change.eq(0).all() or not d.n_starts.eq(6).all()
            or not np.array_equal(d.optimized_readout_parameters, np.where(d.model == "hybrid", 10, 6))):
        raise ValueError("Invalid conditional optimization/parameter counts")
    for stem in ("curves", "params", "results"):
        if set(frames[stem].model) != models:
            raise ValueError(f"Wrong configurations in {stem}")
    c = frames["curves"]
    expected = pd.MultiIndex.from_product([range(expected_mc), sorted(models), ["train", "test"], FREQS, ["std0", "std1"], np.linspace(0, 1, 11)], names=["mc", "model", "split", "freq", "block", "cv"])
    actual = pd.MultiIndex.from_frame(c.assign(cv=c.cv.round(1))[list(expected.names)])
    expected = pd.MultiIndex.from_frame(expected.to_frame(index=False).assign(cv=lambda x: x.cv.round(1)))
    if len(expected.difference(actual)) or not c.p_predicted.between(0, 1).all():
        raise ValueError("Incomplete curve grid/invalid predictions")
    for family in FAMILIES:
        for block in ("std0", "std1"):
            allowed = ("rate" if family == "rate" else "timing" if family == "timing" else
                       (family.split("_")[0 if block == "std0" else 1] if "_" in family else "hybrid"))
            if allowed == "hybrid":
                continue
            forbidden = "timing" if allowed == "rate" else "rate"
            p = frames["params"].query("model == @family and block == @block")
            if (p[[f"gain_{forbidden}_1ms", f"gain_{forbidden}_10ms"]].to_numpy() != 0).any() or not c.loc[(c.model == family) & (c.block == block), f"{forbidden}_drive"].eq(0).all():
                raise ValueError("Forbidden cue is active")
    source = pd.read_csv(bundle / "source_weights.csv")
    merged = frames["weights"].merge(source[["mc", "freq", "thresh_type", "weight"]], on=["mc", "freq", "thresh_type"], validate="one_to_one")
    if len(merged) != 36 * expected_mc or not np.allclose(merged.weight_x, merged.weight_y, rtol=0, atol=1e-14):
        raise ValueError("Locked weights do not match source")
    for row in frames["splits"].itertuples():
        train, test = core.split_replicates(range(100), .8, 100003 * row.mc)
        if row.split_seed != 100003 * row.mc or json.loads(row.train_reps) != sorted(train) or json.loads(row.test_reps) != sorted(test):
            raise ValueError("Mismatched simulation split")
    if (frames["baseline_audit"][["max_prediction_gap", "objective_gap"]] > 1e-8).any().any():
        raise ValueError("Historical source reconstruction mismatch")
    h = d[d.model == "hybrid"].merge(frames["baseline_audit"], on="mc", validate="one_to_one")
    if (h.objective > h.source_objective + 1e-7).any():
        raise ValueError("Hybrid optimization check worsened")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "fit", "merge"])
    parser.add_argument("--root", default=".")
    parser.add_argument("--bundle", default="data/discrimination/locked/inputs")
    parser.add_argument("--out")
    parser.add_argument("--partials", default="logs/partials/discrimination_locked_mixture_followup")
    parser.add_argument("--mc", type=int)
    parser.add_argument("--expected-mc", type=int, default=200)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    bundle = (root / args.bundle).resolve()
    if args.command == "prepare":
        prepare(root, bundle)
    elif args.command == "verify":
        verify(root, bundle)
        print("Locked-mixture fingerprints verified.")
    elif args.command == "fit":
        if args.out is None or args.mc is None:
            parser.error("fit requires --out and --mc")
        fit(root, bundle, root / args.out, args.mc)
    else:
        if args.out is None:
            parser.error("merge requires --out")
        merge(root, bundle, root / args.partials, root / args.out, args.expected_mc)


if __name__ == "__main__":
    main()
