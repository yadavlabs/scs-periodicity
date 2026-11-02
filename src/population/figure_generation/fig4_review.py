"""Promote independent cue models; compare three Fig. 4c displays without refitting."""
from copy import copy
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nbformat
import pandas as pd

from population.manuscript_v2 import ManuscriptFigures, FREQS, REGIMES, CUES, COLORS, label, cue_line_style
from population.figure_generation.notebook_execution import execute_cells

ACTIVE_MIXTURE_STYLE = 'grouped'
ACTIVE_FIG4_STEM = f'fig4_combined_behavior_mixtures_reliance_{ACTIVE_MIXTURE_STYLE}'

FIG4_MARKDOWN = """## Figure 4 — independently fitted evidence families

a: Independent Rate, Timing, and Hybrid predictions under per-frequency mixtures and per-block
readouts, with behavioral mean/SEM and individual rats. Each family optimizes its own mixtures,
gains and biases. Hybrid is bold as the focal model, not a claim of superiority to every model.
b: Primary-hybrid CV-averaged rate reliance; dots are means and tails central 95% ranges across
simulation splits, not animal confidence intervals or repetition-level bootstrap intervals.
c: Each family's own frequency-specific mixture, shared across Std0/Std1 within that family.
These are NOT the former hybrid-mixture-locked controls.

Version B (grouped bars) is the selected panel-c display. Each retained regime has Rate (red),
Hybrid (purple), and Timing (blue) bars; joined x-axis labels identify frequency-specific
response-equivalent regimes. All frequency columns share the same mixture-weight y range.
Alternative stacked and categorical-line displays remain in `figures_v3/exploratory_fig4c/`
for reference and are not additional manuscript outputs.

This is the preferred constrained structure versus the crossed alternative; the per-panel hybrid
remains a less-constrained benchmark in S9/Table 1. The primary is not the lowest-RMSE structure
overall. Independent Timing closely matches Hybrid in Std1; their small error difference is not
robust to the corrected-resampling sensitivity. Existing statistical tables retain the conditional
locked-mixture diagnostics separately, without displaying them as main-figure controls.
"""

FIG4_CODE = f"""# Selected version B: independent cue models with grouped mixture bars.
fig = v2.fig4_reliance_preview(mixture_style='{ACTIVE_MIXTURE_STYLE}', locked_mixtures=False)
display(fig)
plt.close(fig)
"""

S9_MARKDOWN = """## Supplementary Figure 9 — hybrid mixture/readout structures

Panel a compares only three hybrid observers: per-frequency mixtures/per-block readouts (primary),
per-block mixtures/per-frequency readouts (crossed), and per-panel mixtures/per-panel readouts.
The first two are constrained sharing hypotheses; the per-panel model is a less-constrained
benchmark, not a post hoc exclusion from the model comparison. Primary is repeated as a reference.
Panel b shows their CV-averaged rate reliance and central 95% ranges across simulation splits.
These are not the repetition-level observations/intervals used in the permutation analysis.

Independent Rate/Timing curves now appear in main Fig. 4 and are omitted here. Locked-mixture
controls are retained as diagnostics in the results/statistics, not plotted in this figure.
Legend RMSE values are pooled overall simulation-test RMSEs averaged across splits, as in Table 1;
they are not block-mean RMSE or RMSE of the displayed mean curve. Table 1 retains all five models,
including the per-panel model's lower error/AIC-based score and primary's lower BIC-based score.
The primary and per-panel hybrids agree on the direction of the Std0/Std1 difference, not on a
common frequency trend or identical reliance magnitudes. No statistical stars are added here.
"""


def mixture_strip(v2, style):
    """Standalone c-row asset, with a separate right-hand legend column."""
    fig = plt.figure(figsize=(17, 4.4))
    axes = [fig.add_axes([.065 + col*.187, .43, .168, .44]) for col in range(4)]
    mixtures = {f: v2.mixture(family=f, controls=f != 'hybrid') for f in CUES}
    for ax, freq in zip(axes, FREQS):
        v2.independent_mixture_panel(ax, freq, mixtures, style)
        ax.title.set_fontweight('bold')
        ax.tick_params(labelsize=9 if style == 'bars' else 8)
    axes[0].set_ylabel('Mixture weight\n(shared across blocks)')
    legend_ax = fig.add_axes([.82, .43, .175, .44])
    legend_ax.set_axis_off()
    handles = ([Patch(facecolor=CUES[f], label=f.title(), alpha=1 if f == 'hybrid' else .75)
                for f in ['rate', 'timing', 'hybrid']] if style == 'grouped' else
               [Line2D([], [], color=CUES[f], label=f.title(), **cue_line_style(f))
                for f in ['rate', 'timing', 'hybrid']])
    cue = legend_ax.legend(handles=handles, title='Observer Model Components',
                          loc='upper center' if style == 'bars' else 'center',
                          ncol=1, frameon=False, fontsize=10, title_fontsize=10)
    cue.get_texts()[2].set_fontweight('bold')
    if style == 'bars':
        # Avoid overlapping two keys: observer names are already colored under bars.
        cue.remove()
        legend_ax.legend(handles=[Patch(color=COLORS[r], label=label(r)) for r in REGIMES],
                         ncol=3, loc='center', frameon=False, fontsize=9,
                         title='Operating Regimes', title_fontsize=10,
                         columnspacing=.7, handlelength=1.1, handletextpad=.3)
    return v2.figure(fig, f'fig4c_{style}')


def previews(v2):
    preview = copy(v2)
    preview.output = v2.output / 'exploratory_fig4c'
    preview.output.mkdir(exist_ok=True)
    preview.records = []
    for style in ('bars', 'grouped', 'lines'):
        print(f'Exporting independent Fig. 4c: {style}', flush=True)
        fig = mixture_strip(preview, style)
        plt.close(fig)
        fig = preview.fig4_reliance_preview(mixture_style=style)
        plt.close(fig)
    pd.DataFrame(preview.records).drop_duplicates().to_csv(preview.output / 'preview_manifest.csv', index=False)
    (preview.output / 'README.md').write_text(
        '# Fig. 4c alternatives\n\n'
        'All three versions use identical independent Rate/Timing/Hybrid fitted mixtures. '
        'Their full Fig. 4 counterparts have identical behavior and reliance panels.\n\n'
        '- `fig4c_bars`: stacked mixtures, viridis retained-regime colors.\n'
        '- `fig4c_grouped`: three model-colored bars per retained regime.\n'
        '- `fig4c_lines`: model-colored categorical profiles; lines are visual guides only.\n\n'
        'Each mixture is shared across blocks, not across models. Joined labels identify '
        'response-equivalent regimes. The stacked color key names representatives; the '
        'frequency-specific full labels are in `fig4_independent_mixtures_source.csv`. '
        'Grouped and line plots use the same y limits across all frequencies. '
        'Hybrid bolding denotes the focal model, not universal superiority.\n', encoding='utf-8')


def install(notebook):
    """Selective edit; never rebuild unrelated user-edited figure cells."""
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'markdown' and cell.source.startswith('## Figure 4'):
            cell.source = FIG4_MARKDOWN
            if notebook.cells[i+1].source != FIG4_CODE:
                notebook.cells[i+1].source = FIG4_CODE
                notebook.cells[i+1].outputs = []
                notebook.cells[i+1].execution_count = None
        elif cell.cell_type == 'markdown' and cell.source.startswith('## Supplementary Figure 9'):
            if cell.source != S9_MARKDOWN:
                cell.source = S9_MARKDOWN
                notebook.cells[i+1].outputs = []
                notebook.cells[i+1].execution_count = None
        elif cell.cell_type == 'markdown':
            cell.source = cell.source.replace(
                'Main Figure 4 uses mixture-matched\nrefitted cue readouts; Supplementary Figure 9 uses independently optimized cue models.',
                'Main Figure 4 uses independently fitted cue models; Supplementary Figure 9 compares hybrid sharing structures.')
            cell.source = cell.source.replace('the mixture-locked cue comparisons remain in Fig. 4.',
                                              'the independent cue-model comparisons appear in Fig. 4.')


def main():
    # CLI exports do not need a GUI/Tk installation; notebook imports are untouched.
    plt.switch_backend('Agg')
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 300,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .20})
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--previews-only', action='store_true')
    mode.add_argument('--promote-only', action='store_true',
                      help='Refresh only selected Fig. 4 and its manifest, preserving S9/statistical outputs')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = root / 'results/population/figures_v3'
    if not args.previews_only:
        path = root / 'notebooks/population/06_FigureGeneration_v2.ipynb'
        notebook = nbformat.read(path, as_version=4)
        install(notebook)
        statistical = []
        if not args.promote_only:
            from population.figure_generation.refresh_statistics import install as install_statistics
            statistical = install_statistics(notebook)
        indices = [1]
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown' and (cell.source.startswith('## Figure 4') or
                                                (not args.promote_only and cell.source.startswith('## Supplementary Figure 9'))):
                indices.append(i+1)
        # Keep statistical labels/provenance consistent with the new figure roles.
        indices = sorted(set(indices + statistical))
        execute_cells(notebook, root, indices)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        manifest = output / 'figure_manifest_v3.csv'
        if manifest.exists():
            records = pd.read_csv(manifest)
            superseded = {'fig4_combined_behavior_locked_mixtures_reliance',
                          'fig4_combined_behavior_mixtures_reliance_bars',
                          'fig4_combined_behavior_mixtures_reliance_grouped',
                          'fig4_combined_behavior_mixtures_reliance_lines'}
            records = records[~records.stem.isin(superseded)]
            additions = pd.DataFrame([
                dict(kind='figure', stem=ACTIVE_FIG4_STEM),
                dict(kind='table', stem='fig4_independent_mixtures_source')])
            records = pd.concat([records, additions], ignore_index=True).drop_duplicates(['kind', 'stem'])
            records.to_csv(manifest, index=False)
            for cell in notebook.cells:
                if cell.cell_type == 'code' and cell.source.startswith('manifest = pd.concat'):
                    # Do not retain an obsolete displayed manifest after a selective refresh.
                    cell.outputs = [nbformat.v4.new_output('display_data', data={
                        'text/html': records[['kind', 'stem']].to_html(index=False),
                        'text/plain': records[['kind', 'stem']].to_string(index=False)})]
                    cell.execution_count = None
            nbformat.write(notebook, path)
    if not args.promote_only:
        previews(ManuscriptFigures(root, output))
    print(f'Selected Figure 4: {ACTIVE_FIG4_STEM}. Unrelated notebook outputs retained.')


if __name__ == '__main__':
    main()
