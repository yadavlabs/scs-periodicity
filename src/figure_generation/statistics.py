"""Audited, simulation-conditional manuscript comparisons; no fitting or resimulation.

The release-compatible split Wilcoxon tests are NOT tests across independent rats.
The corrected resampled t analysis is an approximate sensitivity, not a validation
of its assumptions for this nonlinear, mean-evidence observer. Repetition tests
condition on the saved cross-fitted observers; shared-fit uncertainty remains.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.integrate import trapezoid
from scipy.stats import t, wilcoxon

from discrimination import discrimination_core as core
from figure_generation.figure_model import select

FREQS = (20., 50., 100., 200.)
GAINS = ('gain_rate_1ms', 'gain_rate_10ms', 'gain_timing_1ms', 'gain_timing_10ms')
FOCUS = 'results/discrimination/scope'
LOCKED = 'results/discrimination/locked/followup'
CONTROL = 'results/discrimination/control'
EVIDENCE = 'data/features/population_cv0_template_evidence.csv'
SCOPES = {'primary_hybrid': 'primary', 'crossed_hybrid': 'crossed', 'per_panel_hybrid': 'per_panel'}


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(values):
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite observations; no silent dropping')
    return x


def holm(pvalues):
    p = np.asarray(pvalues, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p-values')
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out


def corrected_resampled(differences, test_train_ratio=.25):
    """Nadeau-Bengio approximate sensitivity: SE=sqrt((1/n+n_test/n_train)*s²)."""
    d = finite(differences)
    if test_train_ratio <= 0:
        raise ValueError('Invalid test/train ratio')
    mean = float(d.mean())
    se = float(np.sqrt((1 / len(d) + test_train_ratio) * d.var(ddof=1)))
    if se == 0:
        p = 1. if mean == 0 else np.nan  # nonzero constant differences: degenerate variance
    else:
        p = float(2 * t.sf(abs(mean / se), len(d) - 1))
    margin = float(t.ppf(.975, len(d) - 1) * se)
    return dict(corrected_t_p=p, corrected_se=se,
                corrected_ci_low=mean-margin, corrected_ci_high=mean+margin)


def resampling_contrast(a, b, *, paired, rng, n_perm=10000, n_boot=5000):
    """Two-sided absolute-mean statistic; +1 Monte Carlo p; percentile bootstrap.

    Paired sign flips assume symmetric/exchangeable differences under the null;
    unpaired label permutations assume exchangeable distributions. Bootstrap CIs
    are pointwise, conditional on fitted observers, not simultaneous intervals.
    """
    a, b = finite(a), finite(b)
    if paired and a.shape != b.shape:
        raise ValueError('Paired samples must have identical sizes')
    if n_perm < 1 or n_boot < 2:
        raise ValueError('Invalid resampling counts')
    observed = float(a.mean() - b.mean())
    exceed = 0
    boot = []
    # Batching bounds memory even if resampling counts are increased.
    for start in range(0, n_perm, 256):
        size = min(256, n_perm-start)
        if paired:
            null = (rng.choice([-1., 1.], (size, len(a))) * (a-b)).mean(axis=1)
        else:
            joined = np.tile(np.r_[a, b], (size, 1))
            shuffled = rng.permuted(joined, axis=1)
            null = shuffled[:, :len(a)].mean(axis=1) - shuffled[:, len(a):].mean(axis=1)
        exceed += int(np.count_nonzero(abs(null) >= abs(observed) - 1e-14))
    for start in range(0, n_boot, 256):
        size = min(256, n_boot-start)
        ia = rng.integers(len(a), size=(size, len(a)))
        if paired:
            boot.extend((a-b)[ia].mean(axis=1))
        else:
            ib = rng.integers(len(b), size=(size, len(b)))
            boot.extend(a[ia].mean(axis=1) - b[ib].mean(axis=1))
    low, high = np.quantile(boot, [.025, .975])
    return dict(n_a=len(a), n_b=len(b), mean_a=float(a.mean()), mean_b=float(b.mean()),
                difference_a_minus_b=observed, ci_low=float(low), ci_high=float(high),
                p_raw=(exceed+1)/(n_perm+1), paired=paired)


def predict_saved(evidence, weights, gains, bias, cv, block):
    """Vectorized ungated latent-trial observer, with optional repetition dimension.

    This is checked against core.predict in tests AND every historical train/test
    curve at runtime. Nonlinear reliance is evaluated per repetition before AUC.
    """
    rate = (evidence[..., :2] * gains[:, None, :2]).sum(axis=-1)
    timing = (evidence[..., 2:] * gains[:, None, 2:]).sum(axis=-1)
    choose = (weights * expit(bias[:, None] + rate + timing)).sum(axis=-1)
    denom = (weights * (abs(rate)+abs(timing))).sum(axis=-1)
    reliance = np.divide((weights * abs(rate)).sum(axis=-1), denom,
                         out=np.zeros_like(denom), where=denom > core.EPS)
    invert = np.where(block == 'std0', np.isclose(cv, 0), ~np.isclose(cv, 1))
    return dict(p_predicted=np.clip(np.where(invert, 1-choose, choose), 1e-6, 1-1e-6),
                rate_drive=(weights*rate).sum(axis=-1), timing_drive=(weights*timing).sum(axis=-1),
                effective_rate_share=reliance, zero_drive=denom <= core.EPS)


def panel_parameters(weights, params, behavior, regimes, weight_scope, readout_scope):
    def key(scope, row):
        if scope == 'per_frequency':
            return f'freq={int(row.freq)}'
        if scope == 'per_block':
            return row.block
        if scope == 'per_panel':
            return f'freq={int(row.freq)}|block={row.block}'
        raise ValueError(f'Unsupported scope: {scope}')
    if weights.duplicated(['weight_key', 'thresh_type']).any() or params.readout_key.duplicated().any():
        raise ValueError('Duplicate saved parameters')
    w = weights.pivot(index='weight_key', columns='thresh_type', values='weight').reindex(columns=regimes)
    p = params.set_index('readout_key')
    wp = np.array([w.loc[key(weight_scope, row)].to_numpy(float) for row in behavior.itertuples()])
    gp = np.array([p.loc[key(readout_scope, row), list(GAINS)].to_numpy(float) for row in behavior.itertuples()])
    bp = np.array([p.loc[key(readout_scope, row), 'bias'] for row in behavior.itertuples()], float)
    if (not all(np.isfinite(v).all() for v in (wp, gp, bp)) or (wp < 0).any()
            or (gp < 0).any() or not np.allclose(wp.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError('Invalid weights/gains/bias or missing regimes')
    return wp, gp, bp


def check_grid(frame, mc_ids, *, curves=True):
    keys = ['mc', 'split', 'freq', 'block'] + (['cv'] if curves else [])
    if frame.duplicated(keys).any() or set(frame.mc) != set(mc_ids):
        raise ValueError('Duplicate or incomplete split-level observations')
    values = frame[keys].copy()
    if curves:
        if not np.allclose(values.cv * 10, np.round(values.cv * 10), atol=1e-10, rtol=0):
            raise ValueError('Off-grid CV values')
        values['cv'] = values.cv.round(8)
    expected = pd.MultiIndex.from_product([mc_ids, ['train', 'test'], FREQS, ['std0', 'std1']]
                                         + ([np.round(np.linspace(0, 1, 11), 8)] if curves else []), names=keys)
    got = pd.MultiIndex.from_frame(values)
    if len(got) != len(expected) or not got.sort_values().equals(expected.sort_values()):
        raise ValueError('Incomplete frequency/block/CV/split grid')


def rmse_tests(panel_errors, families, test_train_ratio):
    """Block metric matches release: average FOUR panel RMSEs, never pooled RMSE."""
    if panel_errors.duplicated(['model', 'mc', 'freq', 'block']).any():
        raise ValueError('Duplicate panel errors')
    groups = panel_errors.groupby(['model', 'mc', 'block'])
    if not groups.freq.nunique().eq(4).all():
        raise ValueError('Expected four frequencies in every block mean')
    means = groups.rmse.mean().rename('block_mean_rmse').reset_index()
    rows = []
    for family, models in families.items():
        for block in ('std0', 'std1'):
            pivot = means[means.block.eq(block) & means.model.isin(models)].pivot(
                index='mc', columns='model', values='block_mean_rmse')
            if set(pivot.columns) != set(models) or pivot.isna().any().any():
                raise ValueError('Missing paired models/splits; refusing pairwise dropping')
            for a, b in combinations(models, 2):
                d = finite(pivot[a]-pivot[b])
                # Rounding only suppresses floating-point rank artifacts (<1e-12 RMSE).
                rounded = np.round(d, 12)
                p = 1. if not rounded.any() else float(wilcoxon(rounded, zero_method='wilcox',
                    alternative='two-sided', method='auto').pvalue)
                rows.append(dict(family=family, block=block, model_a=a, model_b=b, n_paired_mc=len(d),
                    mean_a=float(pivot[a].mean()), mean_b=float(pivot[b].mean()),
                    difference_a_minus_b=float(d.mean()), median_difference=float(np.median(d)),
                    fraction_a_lower=float((d < 0).mean()), wilcoxon_p=p,
                    **corrected_resampled(d, test_train_ratio)))
    result = pd.DataFrame(rows)
    for _, ids in result.groupby(['family', 'block']).groups.items():
        result.loc[ids, 'wilcoxon_p_holm'] = holm(result.loc[ids, 'wilcoxon_p'])
        cp = result.loc[ids, 'corrected_t_p']
        # Keep degenerate sensitivity tests undefined, and count the entire family.
        adjusted = holm(cp.fillna(1))
        result.loc[ids, 'corrected_t_p_holm'] = np.where(cp.notna(), adjusted, np.nan)
        result.loc[ids, 'family_size'] = len(ids)
    return means, result


def reliance_tests(reps, *, seed, n_perm, n_boot):
    if reps.duplicated(['model', 'freq', 'block', 'rep_index']).any():
        raise ValueError('Average test appearances before testing; repeated rep IDs are not observations')
    rng = np.random.default_rng(seed)
    rows = []
    for model in ('primary_hybrid', 'per_panel_hybrid'):
        data = reps[reps.model.eq(model)]
        for freq in FREQS:
            pair = data[data.freq.eq(freq)].pivot(index='rep_index', columns='block', values='auc')
            if len(pair) != 100 or pair.isna().any().any():
                raise ValueError('Incomplete paired repetition AUCs')
            rows.append(dict(family='block_contrasts', model=model, freq=freq, block='std0 - std1',
                comparison='std0 - std1', **resampling_contrast(pair.std0, pair.std1, paired=True,
                    rng=rng, n_perm=n_perm, n_boot=n_boot)))
        for block in ('std0', 'std1'):
            for fa, fb in combinations(FREQS, 2):
                a = data[data.block.eq(block) & data.freq.eq(fa)].sort_values('rep_index').auc
                b = data[data.block.eq(block) & data.freq.eq(fb)].sort_values('rep_index').auc
                if len(a) != 100 or len(b) != 100:
                    raise ValueError('Incomplete frequency repetition AUCs')
                rows.append(dict(family='frequency_contrasts', model=model, freq=np.nan, block=block,
                    comparison=f'{fa:g} - {fb:g} Hz', **resampling_contrast(a, b, paired=False,
                        rng=rng, n_perm=n_perm, n_boot=n_boot)))
    out = pd.DataFrame(rows)
    for _, ids in out.groupby('family').groups.items():
        out.loc[ids, 'p_bonferroni'] = np.minimum(1, out.loc[ids, 'p_raw'] * len(ids))
        out.loc[ids, 'p_holm_sensitivity'] = holm(out.loc[ids, 'p_raw'])
        out.loc[ids, 'family_size'] = len(ids)
    return out


METHODS = """# Population observer statistical audit

No model or manuscript text is changed. All comparisons use saved fits and current
population evidence, validated against BOTH original train and test predictions.
The 420 axons are one integrated model population, not independent subjects.
The behavioral data are fixed; only simulated repetitions were held out.

## Prediction errors
Two-sided paired Wilcoxon tests reproduce the production release's block metric:
the arithmetic mean of four frequency-specific test RMSEs, not pooled RMSE and not
RMSE of the mean prediction. All three pairs are tested, Holm-adjusted separately
within each model family and block. Scope comparisons and conditional locked-cue
comparisons are distinct families; any independent cue tests are another family.
These are conditional split-stability tests, not 200 independent experiments.
The corrected resampled t sensitivity uses SE=sqrt((1/n+20/80)*sample_variance)
with n-1 degrees of freedom; its pointwise 95% intervals and Holm-adjusted p-values
are APPROXIMATE, not a validated population-observer sampling distribution. It
does not remedy reuse of behavioral targets or model-selection uncertainty.
Wilcoxon differences are rounded to 12 decimals to avoid numerical rank artifacts.

## Rate reliance
For primary and per-panel hybrids only, each saved fit is evaluated on each of its
held-out repetitions. Reliance is weighted absolute rate drive / weighted absolute
total cue drive (core zero-denominator convention: 0). Trapezoidal AUC spans all
11 CVs in [0,1]. Each repetition's AUC is averaged over its held-out appearances;
there are 100 observations per model/frequency/block, not 4,000 appearances.
This per-repetition estimand is NOT the reliance of mean evidence shown in Fig. 4.
Both summaries and zero-drive fractions are exported for auditing.

Block differences use paired sign flips on repetition AUC differences (two-sided
absolute-mean statistic, symmetry/exchangeability under null). Frequency contrasts
use unpaired label permutations, matching the release: repetition IDs across
frequencies do not establish matched physical stimulus realizations. Null requires
exchangeability of distributions, not just equality of means. All six frequency
pairs are included. Bonferroni families are 8 block contrasts (2 models x 4 freqs)
and 24 frequency contrasts (2 models x 2 blocks x 6 pairs); Holm is a sensitivity.
Monte Carlo p=(exceedances+1)/(permutations+1), never zero. Percentile 95% bootstrap
intervals resample paired differences or each frequency separately and are pointwise.
These are exploratory, conditional-on-saved-observer tests. Shared training fits
still induce dependence: averaging held-out appearances does not remove fit
uncertainty. No claim of animal inference, fully independent cross-fitting, or
selection-adjusted confirmatory inference is justified. A full outer refit bootstrap
would be a separate analysis. Failure to reject is not equivalence.

## References / release provenance
- Finalized scs-periodicity/src/discrimination/discrimination_core.py:
  model_performance_statistics_table (paired MC block-mean Wilcoxon + Holm).
- Finalized scs-periodicity/src/figures/figure_generation.py embedded cells 16/17:
  repetition-level reliance, 5,000 bootstraps, 10,000 permutations, Bonferroni.
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
"""


def run_statistics(root, output=None, *, seed=20260929, n_perm=10000, n_boot=5000):
    root = Path(root).resolve()
    output = Path(output) if output else root / 'results/figures/statistics'
    output.mkdir(parents=True, exist_ok=True)
    inputs = set()

    def read(relative):
        path = root / relative
        inputs.add(path)
        return pd.read_csv(path, low_memory=False)

    for folder in (FOCUS, LOCKED):
        path = root / folder / 'validation.json'
        inputs.add(path)
        valid = json.loads(path.read_text())
        if valid.get('status') != 'ready' or valid.get('errors') or valid.get('failed_observer_fits', valid.get('failed_fits', 0)):
            raise ValueError(f'Unvalidated source: {folder}')
    manifest_path = root / LOCKED / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    inputs.add(manifest_path)
    # Check actual feature/behavior fingerprints, not mutable plotting source files.
    for rel, expected in manifest['input_code_sha256'].items():
        if rel.startswith('data/'):
            inputs.add(root / rel)
            if digest(root / rel) != expected:
                raise ValueError(f'Changed historical input: {rel}')
    evidence, regimes, reps = core.load_population_evidence(str(root / EVIDENCE), FREQS)
    inputs.add(root / EVIDENCE)
    behavior = core.load_behavior(str(root / 'data/discrimination/slack'), FREQS,
                                  set(zip(evidence.freq, evidence.cv)))
    inputs.update((root / 'data/discrimination/slack').glob('*.csv'))
    if len(reps) != 100 or len(behavior) != 88:
        raise ValueError('Expected 100 repetitions and 88 conditions')
    source = evidence.set_index(['rep_index', 'freq', 'cv', 'thresh_type'])
    index = pd.MultiIndex.from_tuples([(rep, row.freq, row.cv, regime) for rep in reps
        for row in behavior.itertuples() for regime in regimes], names=source.index.names)
    tensor = source.reindex(index)[list(core.EVIDENCE_COLUMNS)].to_numpy(float).reshape(100, 88, len(regimes), 4)
    if not np.isfinite(tensor).all():
        raise ValueError('Incomplete repetition evidence tensor')
    rep_lookup = {rep: i for i, rep in enumerate(reps)}
    split_frame = read(f'{LOCKED}/merged_splits.csv')
    mc_ids = list(range(200))
    if set(split_frame.mc) != set(mc_ids) or split_frame.mc.duplicated().any():
        raise ValueError('Incomplete/duplicate split manifest')
    splits = {}
    for row in split_frame.itertuples():
        train, test = core.split_replicates(reps, manifest['train_fraction'], manifest['seed'] + 100003*row.mc)
        if (set(train) != set(json.loads(row.train_reps)) or set(test) != set(json.loads(row.test_reps))
                or row.split_seed != manifest['seed'] + 100003*row.mc):
            raise ValueError('Split reconstruction differs from locked source manifest')
        splits[row.mc] = {'train': [rep_lookup[x] for x in sorted(train)],
                          'test': [rep_lookup[x] for x in sorted(test)]}
    focus = {s: read(f'{FOCUS}/merged_{s}.csv') for s in ('curves', 'weights', 'params', 'results')}
    datasets = {}
    for model, scope in SCOPES.items():
        datasets[model] = {s: select(frame, scope) for s, frame in focus.items()}
    locked = {s: read(f'{LOCKED}/merged_{s}.csv') for s in ('curves', 'weights', 'params', 'results')}
    # Validate only Rate/Timing against the original hybrid, not reoptimized locked Hybrid.
    for family in ('rate', 'timing'):
        frames = {s: f[f.model.eq(family)].copy() for s, f in locked.items() if s != 'weights'}
        frames['weights'] = locked['weights'].copy()
        frames['weights']['weight_key'] = frames['weights'].freq.map(lambda f: f'freq={int(f)}')
        frames['params']['readout_key'] = frames['params'].block
        datasets[f'locked_{family}'] = frames
    availability = [dict(comparison='Mixture structures', status='available', detail='Primary, crossed, per-panel hybrids'),
                    dict(comparison='Retained locked-mixture diagnostics', status='available', detail='Not plotted in Fig. 4: hybrid mixtures fixed; Rate/Timing readouts refitted')]
    independent_paths = [root / CONTROL / f'merged_{s}.csv' for s in ('curves', 'weights', 'params', 'results')]
    independent_available = all(p.exists() for p in independent_paths)
    if independent_available:
        path = root / CONTROL / 'validation.json'
        inputs.add(path)
        valid = json.loads(path.read_text())
        if valid.get('status') != 'ready' or valid.get('failed_observer_fits', 0) or valid.get('errors'):
            raise ValueError('Unvalidated independent control source')
        controls = {s: read(f'{CONTROL}/merged_{s}.csv') for s in ('curves', 'weights', 'params', 'results')}
        for family in ('rate', 'timing'):
            datasets[f'independent_{family}'] = {s: select(f, 'primary', family) for s, f in controls.items()}
        # Audit original hybrid as a bridge; do not silently replace it with another fit.
        datasets['independent_hybrid'] = {s: select(f) for s, f in controls.items()}
    availability.append(dict(comparison='Fig. 4 independent Rate/Timing/Hybrid',
        status='available' if independent_available else 'pending_source_files',
        detail=('Original split-level predictions, mixtures and readouts available for reconstruction audit'
                if independent_available else 'Needs discrimination/merged_{curves,weights,params,results}.csv; summary means/SEMs cannot supply paired tests')))

    audit, panel_rows, appearances, split_auc = [], [], [], []
    for model, frames in datasets.items():
        print(f'Auditing saved predictions: {model}', flush=True)
        curves = frames['curves'].copy()
        curves['freq'] = curves.freq.astype(float)
        check_grid(curves, mc_ids)
        if not np.isfinite(curves[['p_observed', 'p_predicted', 'rate_drive', 'timing_drive', 'effective_rate_share']].to_numpy(float)).all():
            raise ValueError('Nonfinite saved predictions')
        results = frames['results'].copy()
        results['freq'] = pd.to_numeric(results.freq, errors='coerce')
        results = results[results.freq.isin(FREQS) & results.block.isin(['std0', 'std1'])]
        check_grid(results, mc_ids, curves=False)
        if not np.isfinite(results.rmse.to_numpy(float)).all():
            raise ValueError('Nonfinite saved errors')
        if set(frames['weights'].mc) != set(mc_ids) or set(frames['params'].mc) != set(mc_ids):
            raise ValueError('Incomplete saved fit parameters')
        ws, rs = (('per_frequency', 'per_block') if model.startswith('locked') else
                  (str(curves.weight_scope.iloc[0]), str(curves.readout_scope.iloc[0])))
        if not model.startswith('locked'):
            if not curves.rate_gate_mode.eq('none').all() or not curves.mixture_semantics.eq('latent_trial').all():
                raise ValueError('Only ungated latent-trial models are implemented')
            if not frames['results'].optimization_success.astype(str).str.lower().eq('true').all():
                raise ValueError('Failed observer fits present')
            if not frames['results'].n_train_reps.eq(80).all() or not frames['results'].n_test_reps.eq(20).all():
                raise ValueError('Unexpected fit split sizes')
        for mc in mc_ids:
            w, g, b = panel_parameters(frames['weights'][frames['weights'].mc.eq(mc)],
                frames['params'][frames['params'].mc.eq(mc)], behavior, regimes, ws, rs)
            for split, ids in splits[mc].items():
                pred = predict_saved(tensor[ids].mean(axis=0), w, g, b,
                                     behavior.cv.to_numpy(), behavior.block.to_numpy())
                saved = curves[curves.mc.eq(mc) & curves.split.eq(split)].sort_values(['freq', 'block', 'cv'])
                if not np.allclose(saved.p_observed, behavior.p_mean, rtol=0, atol=1e-12):
                    raise ValueError('Saved behavioral targets differ')
                gaps = {key: float(np.max(abs(pred[key] - saved[key].to_numpy()))) for key in
                        ('p_predicted', 'rate_drive', 'timing_drive', 'effective_rate_share')}
                if max(gaps.values()) > 1e-8:
                    raise ValueError(f'Historical prediction mismatch: {model}/{mc}/{split}: {gaps}')
                audit.append(dict(model=model, mc=mc, split=split, **gaps))
                for (freq, block), ids_panel in behavior.groupby(['freq', 'block']).groups.items():
                    loc = np.asarray(ids_panel)
                    error = float(np.sqrt(np.mean((pred['p_predicted'][loc]-behavior.p_mean.to_numpy()[loc])**2)))
                    saved_error = results[(results.mc == mc) & (results.split == split)
                                          & (results.freq == freq) & (results.block == block)].rmse.item()
                    if abs(error-saved_error) > 1e-9:
                        raise ValueError('Saved panel RMSE does not match reconstructed predictions')
                    if split == 'test':
                        panel_rows.append(dict(model=model, mc=mc, freq=freq, block=block, rmse=error))
                        if model in ('primary_hybrid', 'per_panel_hybrid'):
                            split_auc.append(dict(model=model, mc=mc, freq=freq, block=block,
                                auc=float(trapezoid(pred['effective_rate_share'][loc], behavior.cv.iloc[loc]))))
            if model in ('primary_hybrid', 'per_panel_hybrid'):
                ids = splits[mc]['test']
                rp = predict_saved(tensor[ids], w, g, b, behavior.cv.to_numpy(), behavior.block.to_numpy())
                for (freq, block), ids_panel in behavior.groupby(['freq', 'block']).groups.items():
                    loc = np.asarray(ids_panel)
                    auc = trapezoid(rp['effective_rate_share'][:, loc], behavior.cv.iloc[loc], axis=1)
                    zero = rp['zero_drive'][:, loc].mean(axis=1)
                    appearances.extend(dict(model=model, mc=mc, freq=freq, block=block,
                        rep_index=int(reps[rep]), auc=float(value), zero_drive_fraction=float(z))
                        for rep, value, z in zip(ids, auc, zero))
    panel = pd.DataFrame(panel_rows)
    families = {'mixture_structure': list(SCOPES),
                'conditional_locked_cues': ['primary_hybrid', 'locked_rate', 'locked_timing']}
    if independent_available:
        families['independent_cues'] = ['independent_hybrid', 'independent_rate', 'independent_timing']
    means, errors = rmse_tests(panel, families, .25)
    appearance = pd.DataFrame(appearances)
    repetition = appearance.groupby(['model', 'freq', 'block', 'rep_index'], as_index=False).agg(
        auc=('auc', 'mean'), n_test_appearances=('mc', 'nunique'), zero_drive_fraction=('zero_drive_fraction', 'mean'))
    if len(repetition) != 1600 or repetition.n_test_appearances.min() < 1:
        raise ValueError('Incomplete repetition coverage')
    print('Running repetition-level permutations and bootstraps', flush=True)
    reliance = reliance_tests(repetition, seed=seed, n_perm=n_perm, n_boot=n_boot)
    comparison = repetition.groupby(['model', 'freq', 'block']).auc.mean().rename('mean_repetition_auc').to_frame().join(
        pd.DataFrame(split_auc).groupby(['model', 'freq', 'block']).auc.mean().rename('mean_split_evidence_auc')).reset_index()
    tables = dict(availability=pd.DataFrame(availability), reconstruction_audit=pd.DataFrame(audit),
        rmse_panel_observations=panel, rmse_block_observations=means, rmse_tests=errors,
        reliance_test_appearances=appearance, reliance_repetition_auc=repetition,
        reliance_estimand_comparison=comparison, reliance_tests=reliance)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
        if name not in ('reliance_test_appearances', 'rmse_panel_observations', 'reconstruction_audit'):
            table.to_html(output / f'{name}.html', index=False, float_format=lambda v: f'{v:.6g}')
    (output / 'METHODS.md').write_text(METHODS, encoding='utf-8')
    provenance = dict(status='complete_available_comparisons' if not independent_available else 'complete',
        seed=seed, n_permutations=n_perm, n_bootstraps=n_boot, n_mc=200, n_repetitions=100,
        conditional_inference=True, animal_inference=False, outer_refit_bootstrap=False,
        max_reconstruction_gap=float(pd.DataFrame(audit).iloc[:, 3:].to_numpy().max()),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        inputs={str(p.relative_to(root)).replace('\\', '/'): digest(p) for p in sorted(inputs)},
        code_sha256=digest(__file__),
        output_sha256={p.name: digest(p) for p in sorted(output.glob('*.csv'))})
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run_statistics(args.root, args.output)


if __name__ == '__main__':
    main()
