"""Render an alternative S6 regime without changing the manuscript notebook."""
import argparse
from pathlib import Path
import nbformat

try:
    from .notebook_execution import execute_cells
except ImportError:
    from notebook_execution import execute_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["I_10", "I_25", "I_50", "I_all"], default="I_50")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = nbformat.read(root / "notebooks/06_FigureGeneration_v2.ipynb", as_version=4)
    setup = next(c.source for c in source.cells if c.cell_type == "code")
    stem = f"preview_suppfig6_{args.regime}"
    feature = ("from population.figure_generation.supplementary import feature_example\n"
               f"fig = feature_example(v2, regime={args.regime!r}, stem={stem!r})\n"
               "display(fig)\nplt.close(fig)")
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup),
                                              nbformat.v4.new_code_cell(feature)])
    execute_cells(notebook, root, [0, 1])
    print(f"Preview saved as {stem}.png/.pdf; manuscript notebook unchanged.")


if __name__ == "__main__":
    main()
