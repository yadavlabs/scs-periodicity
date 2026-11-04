"""Execute selected notebook cells using this repository's virtual environment."""
from copy import deepcopy
from pathlib import Path

from jupyter_client import KernelManager
from nbclient import NotebookClient


def local_kernel_manager(root):
    root = Path(root)
    candidates = [root / ".venv/Scripts/python.exe", root / ".venv/bin/python"]
    interpreter = next((p for p in candidates if p.is_file()), None)
    if interpreter is None:
        raise FileNotFoundError("Repository .venv interpreter is required; refusing a global fallback")
    manager = KernelManager(kernel_name="python3")
    # This modifies only the in-memory specification, not registered kernels.
    manager.kernel_spec.argv = [str(interpreter), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    return manager


def execute_cells(notebook, root, indices):
    metadata = deepcopy(notebook.metadata)
    client = NotebookClient(notebook, timeout=600, km=local_kernel_manager(root),
                            resources={"metadata": {"path": str(root)}})
    try:
        with client.setup_kernel():
            for i in indices:
                print(f"Executing cell {i}", flush=True)
                client.execute_cell(notebook.cells[i], i)
    finally:
        # nbclient updates kernel/language metadata during execution. Keep the
        # user's display name and any VS Code interpreter metadata intact.
        notebook.metadata = metadata
