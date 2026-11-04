"""Execute code cells of a data-generation notebook in one HPC Python process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--through-cell", type=int, default=None,
                        help="Stop after this zero-based notebook cell index.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = args.notebook.resolve()
    if not notebook.is_relative_to(root / "notebooks") or not notebook.is_file():
        raise ValueError("Expected a notebook in this repository's notebooks directory")
    os.chdir(root)
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    namespace: dict[str, object] = {"__name__": "__main__"}
    for number, cell in enumerate(cells):
        if args.through_cell is not None and number > args.through_cell:
            break
        if cell.get("cell_type") == "code":
            print(f"Running {notebook.name} cell {number}", flush=True)
            source = cell.get("source", "")
            exec(compile("".join(source), f"{notebook.name}:cell-{number}", "exec"), namespace)
    print(f"Completed {notebook.name}", flush=True)


if __name__ == "__main__":
    main()
