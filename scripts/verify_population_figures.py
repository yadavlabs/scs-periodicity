"""Execute the accepted population figure notebook from archived outputs.

The executed notebook is written under results/ for audit. The source notebook
remains clear of machine-specific outputs and execution metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/06_FigureGeneration.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    interpreter = args.python.resolve()
    if not interpreter.is_file():
        parser.error(f"Python interpreter not found: {interpreter}")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    ipython_dir = ROOT / "results/figure_verification/ipython"
    jupyter_runtime = ROOT / "results/figure_verification/jupyter_runtime"
    ipython_dir.mkdir(parents=True, exist_ok=True)
    jupyter_runtime.mkdir(parents=True, exist_ok=True)
    os.environ["IPYTHONDIR"] = str(ipython_dir)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(jupyter_runtime)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        notebook, timeout=args.timeout, km=manager,
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    output_dir = ROOT / "results/figure_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "06_FigureGeneration.executed.ipynb"
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    report = {
        "status": "passed",
        "interpreter": str(interpreter),
        "notebook": str(NOTEBOOK.relative_to(ROOT)),
        "code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
        "executed_code_cells": sum(
            cell.cell_type == "code" and cell.execution_count is not None
            for cell in notebook.cells
        ),
        "output_notebook": str(output.relative_to(ROOT)),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
