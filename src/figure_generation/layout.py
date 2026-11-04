"""Shared physical spacing for main and supplementary observer comparisons."""
import numpy as np
import matplotlib.pyplot as plt


def observer_grid(mixture_row=False):
    width, left, right = 17., 1.05, .20
    gap, reliance_gap, row_gap = .33, .80, .48
    heights = [2.1, 2.1] + ([1.47] if mixture_row else [])
    bottom, top = (1.55 if mixture_row else 1.15), .48
    height = sum(heights) + row_gap * (len(heights) - 1) + bottom + top
    column = (width - left - right - 3 * gap - reliance_gap) / 4.95
    xs = [left + i * (column + gap) for i in range(4)]
    xs.append(xs[-1] + column + reliance_gap)
    fig = plt.figure(figsize=(width, height))
    axes = np.empty((len(heights), 5), dtype=object)
    y = height - top
    for row, h in enumerate(heights):
        y -= h
        for col, x in enumerate(xs):
            axes[row, col] = fig.add_axes([x / width, y / height,
                                         column * (.95 if col == 4 else 1) / width, h / height])
        y -= row_gap
    return fig, axes


def observer_headers(fig, axes, frequencies):
    """Shared column headers and block labels, outside individual panels."""
    for col, freq in enumerate(frequencies):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, box.y1 + .16 / fig.get_figheight(),
                 f'{freq} Hz', ha='center', fontsize=12, weight='bold')
    for row, block in enumerate(['STD0', 'STD1']):
        box = axes[row, 0].get_position()
        fig.text(box.x0 - .88 / fig.get_figwidth(), (box.y0 + box.y1) / 2,
                 block, rotation=90, ha='center', va='center', fontsize=11, weight='bold')
    for ax in axes.flat:
        ax.set_title('')


def reliance_background(ax):
    """Cue-dominance guide, not an uncertainty interval or fitted quantity."""
    ax.axhspan(.5, 1, color='#e30613', alpha=.045, linewidth=0, zorder=-5)
    ax.axhspan(0, .5, color='#0575ab', alpha=.045, linewidth=0, zorder=-5)


def panel_label(fig, ax, letter):
    """Outer-margin labels clear row labels; b stays within its narrower gutter."""
    box = ax.get_position()
    offset = .64 if letter == 'b' else 1.04
    fig.text(box.x0 - offset / fig.get_figwidth(), box.y1 + .12 / fig.get_figheight(),
             letter, weight='bold', fontsize=13)
