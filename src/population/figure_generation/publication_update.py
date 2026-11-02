"""Current manuscript presentation without changing fitted models or raw statistics."""
from pathlib import Path
import nbformat
import pandas as pd
from population.figure_generation.notebook_execution import execute_cells

MODEL_NAMES = {'primary_hybrid': 'Per-frequency hybrid', 'crossed_hybrid': 'Per-block hybrid',
               'per_panel_hybrid': 'Per-panel hybrid', 'primary_rate': 'Per-frequency rate-only',
               'primary_timing': 'Per-frequency timing-only'}
COLUMNS = ['model', 'parameters', 'simulation_test_RMSE', 'std0_frequency_mean_RMSE',
           'std1_frequency_mean_RMSE', 'delta_AIC', 'delta_BIC', 'std0_rate_share', 'std1_rate_share']


def consolidated_table(v2):
    metrics = v2.comparisons()
    frame = metrics[COLUMNS].copy()
    frame['model'] = frame.model.map(MODEL_NAMES)
    if frame.model.isna().any() or len(frame) != 5:
        raise ValueError('Expected the five independent/scope models')
    v2.table(frame, 'table1_consolidated')
    frame.columns = pd.MultiIndex.from_tuples([
        ('Model', ''), ('Complexity', 'k'), ('RMSE', 'Overall'), ('RMSE', 'Std0'),
        ('RMSE', 'Std1'), ('Criterion scores', 'ΔAIC*'), ('Criterion scores', 'ΔBIC*'),
        ('Rate reliance', 'Std0'), ('Rate reliance', 'Std1')])
    return frame.style.hide(axis='index').format({
        ('Complexity', 'k'): '{:.0f}', **{('RMSE', x): '{:.4f}' for x in ['Overall', 'Std0', 'Std1']},
        **{('Criterion scores', x): '{:.1f}' for x in ['ΔAIC*', 'ΔBIC*']},
        **{('Rate reliance', x): '{:.3f}' for x in ['Std0', 'Std1']}})


def install(notebook):
    """Update presentation cells only; preserve kernel and all diagnostic analyses."""
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'markdown':
            continue
        if cell.source.startswith('## Figure 3'):
            cell.source = cell.source.replace(
                '(with frequency-specific biases), and the per-CV capacity fit to the six measured points.',
                '(green), per-frequency mixtures held constant across CV (purple dashed), and the\n'
                'per-CV capacity fit (orange squares) to the six measured points. All three use\n'
                'the same mean behavioral targets and frequency-specific biases.')
        elif cell.source.startswith('## Table 1'):
            cell.source = '''## Table 1 — model performance and rate reliance

One table combines the five models, nominal parameter counts, overall/block RMSE,
heuristic ΔAIC*/ΔBIC*, and block-mean rate reliance. Model names refer to mixture sharing:
per-frequency mixtures use block-specific gains/intercepts; per-block mixtures use
frequency-specific gains/intercepts; per-panel mixtures use panel-specific parameters.
Overall RMSE pools all frequency–block–CV points; block RMSE averages four frequency RMSEs.
Rate/timing-only reliance is fixed at 1/0 by construction. Behavioral animals are not held out.
Full numerical diagnostic exports remain available; they are not numbered supplementary tables.
'''
            notebook.cells[i+1].source = ('from population.figure_generation.publication_update import consolidated_table\n'
                                          'display(consolidated_table(v2))')
        elif cell.source.startswith('## Supplementary tables and remaining manual decisions'):
            cell.source = '''## Supplementary Table 1 and analysis diagnostics

Only the conductivity/material table remains in the manuscript. Verify active COMSOL
physics before removing platinum conductivity and add material sources. Supplementary
Tables 2–4 are omitted: detection comparisons appear in Fig. 3, reliance in Fig. 4/S9,
and selected RMSE comparisons are summarized in the text. Statistical and nodal-phase
outputs below/above remain reproducibility diagnostics, not manuscript table assignments.
'''
        elif cell.source.startswith('## Supplementary Figure 5'):
            if 'True-CV means' not in cell.source:
                cell.source += ('\n\nTrue-CV means amplitudes determined separately at each CV; CV0-based means\n'
                                'amplitudes determined at CV = 0 and held fixed across CV. Both use\n'
                                'index-axon regime amplitudes applied to the entire population.\n')
        elif cell.source.startswith('## Figure 4') or cell.source.startswith('## Supplementary Figure 9'):
            cell.source = cell.source.replace('per-block\nreadouts', 'block-specific\ndecision parameters')
            cell.source = cell.source.replace('Primary-hybrid', 'Per-frequency hybrid')
            cell.source = cell.source.replace('per-frequency mixtures/per-block readouts (primary)',
                                               'per-frequency mixtures/block-specific gains and intercepts')
            cell.source = cell.source.replace('per-block mixtures/per-frequency readouts (crossed)',
                                               'per-block mixtures/frequency-specific gains and intercepts')
            cell.source = cell.source.replace('Existing statistical tables retain', 'Archived statistical exports retain')
    nbformat.validate(notebook)


def main():
    root = Path(__file__).resolve().parents[3]
    path = root / 'notebooks/population/06_FigureGeneration_v2.ipynb'
    notebook = nbformat.read(path, as_version=4)
    install(notebook)
    indices = [1]
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'markdown' and cell.source.startswith(
                ('## Figure 3', '## Table 1', '## Supplementary Figure 9')):
            indices.append(i+1)
    execute_cells(notebook, root, indices)
    nbformat.write(notebook, path)
    # A selective run must not replace the full export manifest with a partial one.
    output = root / 'results/population/figures_v3'
    manifest_path = output / 'figure_manifest_v3.csv'
    manifest = pd.read_csv(manifest_path)
    if 'table1_consolidated' not in set(manifest.stem):
        row = {col: '' for col in manifest.columns}
        row.update(kind='table', stem='table1_consolidated')
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
    print('Updated Fig. 3, unified Table 1, and S9 names; unrelated outputs/kernel preserved.')


if __name__ == '__main__':
    main()
