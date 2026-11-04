"""Insert/refresh ONLY statistical cells, preserving plots, edits, and kernel metadata."""
import argparse
from pathlib import Path
import pandas as pd

import nbformat

from population.figure_generation.notebook_execution import execute_cells

PREFIX = 'population-statistics-'


def statistics_cells():
    sources = [
        ('markdown', """## Statistical comparisons — population observer

These tables extend the finalized single-axon release's statistical workflow without changing
the population fits, figures, or Table 1. **All inference is conditional on simulations and saved
observers, not independent animals.** The same behavioral targets were used during fitting.

RMSE: two-sided paired Wilcoxon tests on each split's mean of four frequency-specific RMSEs,
Holm-adjusted over all three model pairs within each family/block. Repeated-split p-values describe
partition stability. The additional corrected-resampled t p-values/95% intervals are approximate
dependence-aware sensitivities, **not** validated confirmatory inference for this observer.
Retained mixture-locked diagnostics are kept separate from independently optimized cue models (Fig. 4).
Missing source files are reported explicitly; summary means/SEMs cannot reconstruct paired tests.

Rate reliance: each held-out repetition is evaluated with its saved observer; its CV-integrated
reliance is averaged over test appearances before testing (100 reps, not 200 splits or 420 axons).
This is a different estimand from Fig. 4's reliance of split-averaged evidence, shown below for audit.
Block contrasts use paired sign flips; frequency contrasts use unpaired label permutations.
Both use two-sided absolute-mean statistics, 10,000 permutations and 5,000 bootstrap draws.
Bonferroni families contain 8 block and 24 frequency contrasts across primary/per-panel hybrids;
Holm is also exported as a sensitivity. Bootstrap intervals are pointwise, not simultaneous.

These repetition tests are **exploratory and conditional on the saved fits**: shared-training
dependence/fit uncertainty is not removed by averaging test appearances. Sign flips require
symmetric/exchangeable null differences; frequency permutations require exchangeability across
frequency samples. Nonsignificance does not establish equivalence. No outer refit bootstrap is run.
Full methods, software versions, hashes, splits audit, and numerical observations are exported
under `figures_v3/statistics/`. Existing Fig. 4 reliance values remain unchanged.

References: [Wilcoxon assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html),
[permutation assumptions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html),
[corrected resampling sensitivity](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).
"""),
        ('code', """# Standalone/re-runnable; no figure regeneration or model fitting.
from pathlib import Path
import sys
from IPython.display import display, Markdown
_stats_root = next(p for p in [Path.cwd(), *Path.cwd().parents]
                   if (p / 'src/population').is_dir())
if str(_stats_root / 'src') not in sys.path:
    sys.path.insert(0, str(_stats_root / 'src'))
from population.figure_generation.statistics import run_statistics
statistics_output = _stats_root / 'results/figures/statistics'
statistics_tables = run_statistics(_stats_root, statistics_output)
display(statistics_tables['availability'])
display(Markdown('Reconstruction checks passed for every included train/test fit. '
                 'Missing independent controls, if listed above, have not been replaced by locked controls.'))
if 'v2' in globals():
    for name in statistics_tables:
        record = dict(kind='table', stem=f'statistics/{name}')
        if record not in v2.records:
            v2.records.append(record)
"""),
        ('code', """display(Markdown('### Prediction error: paired split comparisons'))
_rmse = statistics_tables['rmse_tests']
for _family, _table in _rmse.groupby('family', sort=False):
    display(Markdown(f'**{_family.replace("_", " ")}** — negative difference favors model A. '
                     'Wilcoxon: conditional split stability; corrected t: approximate sensitivity.'))
    display(_table[['block', 'model_a', 'model_b', 'n_paired_mc', 'mean_a', 'mean_b',
                    'difference_a_minus_b', 'wilcoxon_p_holm', 'corrected_t_p_holm',
                    'corrected_ci_low', 'corrected_ci_high']].style.format(precision=5).format(
                        {'wilcoxon_p_holm': '{:.3g}', 'corrected_t_p_holm': '{:.3g}'}))
"""),
        ('code', """display(Markdown('### Rate reliance: repetition-level comparisons'))
_reliance = statistics_tables['reliance_tests']
for _family, _table in _reliance.groupby('family', sort=False):
    display(Markdown(f'**{_family.replace("_", " ")}** — positive difference favors the first block/frequency. '
                     'Conditional exploratory tests; pointwise bootstrap intervals.'))
    display(_table[['model', 'freq', 'block', 'comparison', 'n_a', 'n_b', 'mean_a', 'mean_b',
                    'difference_a_minus_b', 'ci_low', 'ci_high', 'p_bonferroni']].style.format(
                        precision=4, na_rep='—').format({'p_bonferroni': '{:.3g}'}))
display(Markdown('**Estimand check:** per-repetition AUC versus the existing split-mean-evidence AUC. '
                 'Differences here do not change Fig. 4/Table 1.'))
display(statistics_tables['reliance_estimand_comparison'].round(4))
"""),
        ('markdown', """### Interpretation for manuscript review

- Keep prediction-error comparisons separate from model parsimony/interpretability. A lower
  RMSE for the per-panel model is not evidence that the primary model is the universal winner.
- Compare the magnitude of the Hybrid–Timing Std1 difference, not just its p-value. Small
  differences can be stable across simulation splits; that does not establish biological importance.
- The block-reliance tests address Std0 **versus Std1**, not whether either exceeds 0.5.
  In particular, repetition-level and mean-evidence reliance can fall on different sides of 0.5.
- Very small adjusted permutation p-values can be at the finite resampling resolution;
  they are not zero. All p-values remain conditional/exploratory as described above.
- Full independently fitted cue-model comparisons remain pending wherever the availability
  table says `pending_source_files`. Restore the original `discrimination/merged_curves.csv`,
  `merged_weights.csv`, `merged_params.csv`, and `merged_results.csv` to enable automatic,
  numerically audited comparisons. Do not use aggregate SEMs to synthesize paired observations.
"""),
    ]
    cells = []
    for i, (kind, source) in enumerate(sources):
        cell = (nbformat.v4.new_markdown_cell(source) if kind == 'markdown'
                else nbformat.v4.new_code_cell(source))
        cell.id = f'{PREFIX}{i}'
        cells.append(cell)
    return cells


def install(notebook):
    """Idempotent section update: leave every other cell and metadata untouched."""
    previous = {c.id: c for c in notebook.cells if c.id.startswith(PREFIX)}
    cells = [c for c in notebook.cells if not c.id.startswith(PREFIX)]
    boundary = next(i for i, c in enumerate(cells)
                    if c.cell_type == 'markdown' and c.source.startswith('## Supplementary Figure 1'))
    new = statistics_cells()
    new = [previous[c.id] if c.id in previous and previous[c.id].source == c.source else c for c in new]
    notebook.cells = cells[:boundary] + new + cells[boundary:]
    return [i for i, c in enumerate(notebook.cells) if c.id.startswith(PREFIX) and c.cell_type == 'code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    path = root / 'notebooks/06_FigureGeneration_v2.ipynb'
    notebook = nbformat.read(path, as_version=4)
    indices = install(notebook)
    if args.execute:
        execute_cells(notebook, root, indices)
        # A selective refresh does not rerun the full notebook's manifest cell.
        # Update its existing CSV without dropping earlier figure records.
        manifest = root / 'results/figures/figure_manifest_v3.csv'
        if manifest.exists():
            records = pd.read_csv(manifest)
            additions = pd.DataFrame([dict(kind='table', stem=f'statistics/{p.stem}')
                for p in sorted((manifest.parent / 'statistics').glob('*.csv'))])
            pd.concat([records, additions], ignore_index=True).drop_duplicates(['kind', 'stem']).to_csv(manifest, index=False)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print('Statistical cells updated; other cells and kernel metadata preserved.')


if __name__ == '__main__':
    main()
