"""Refresh selected supplementary figures without changing other cell outputs."""
import argparse
from pathlib import Path
import re
import nbformat

try:
    from .notebook_execution import execute_cells
except ImportError:
    from notebook_execution import execute_cells


def selected_cells(notebook, figures):
    indices = [next(i for i, c in enumerate(notebook.cells) if c.cell_type == "code")]
    selected = False
    found = set()
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "markdown" and "## " in cell.source:
            headings = re.findall(r"^## Supplementary Figures? (\d+)(?: and (\d+))?", cell.source, re.M)
            numbers = {int(n) for heading in headings for n in heading if n}
            selected = bool(numbers & set(figures))
            found |= numbers
        elif cell.cell_type == "code" and selected:
            indices.append(i)
    if set(figures) - found:
        raise ValueError(f"Unknown supplementary figures: {set(figures) - found}")
    return sorted(set(indices))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    path = root / "notebooks/06_FigureGeneration_v2.ipynb"
    notebook = nbformat.read(path, as_version=4)
    execute_cells(notebook, root, selected_cells(notebook, args.figures))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Selected supplementary outputs updated; other outputs and kernel metadata preserved.")


if __name__ == "__main__":
    main()
