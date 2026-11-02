import unittest
from pathlib import Path
import nbformat
from population.figure_generation.notebook_execution import local_kernel_manager, execute_cells
from population.figure_generation.refresh_manuscript_v2_supplementary import selected_cells
from population.figure_generation.audit_supplementary_examples import accepted_count


class FigureGenerationHelperTests(unittest.TestCase):
    def require_local_kernel(self):
        root = Path(__file__).resolve().parents[1]
        if not any((root / path).is_file() for path in ('.venv/Scripts/python.exe', '.venv/bin/python')):
            self.skipTest('Notebook execution requires a repository-local .venv')
        return root

    def test_kernel_is_explicit_repository_venv(self):
        root = self.require_local_kernel()
        manager = local_kernel_manager(root)
        self.assertTrue(Path(manager.kernel_spec.argv[0]).is_relative_to(root / ".venv"))
        self.assertEqual(manager.kernel_spec.argv[1:3], ["-m", "ipykernel_launcher"])

    def test_metadata_and_unselected_outputs_are_preserved(self):
        root = self.require_local_kernel()
        metadata = {"kernelspec": {"name": "python3", "display_name": ".venv (3.12.10)", "language": "python"},
                    "vscode": {"interpreter": {"hash": "preserve-this"}}}
        n = nbformat.v4.new_notebook(metadata=metadata, cells=[nbformat.v4.new_code_cell(
            "import sys; print(sys.executable)")])
        execute_cells(n, root, [0])
        self.assertEqual(dict(n.metadata), metadata)
        self.assertIn(".venv", n.cells[0].outputs[0].text)

    def test_supplementary_selection(self):
        n = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("setup"),
            nbformat.v4.new_markdown_cell("## Supplementary Figures 3 and 4"),
            nbformat.v4.new_code_cell("both"), nbformat.v4.new_markdown_cell("## Supplementary Figure 6"),
            nbformat.v4.new_code_cell("evidence")])
        self.assertEqual(selected_cells(n, [4]), [0, 2])
        self.assertEqual(selected_cells(n, [3, 6]), [0, 2, 4])
        with self.assertRaises(ValueError):
            selected_cells(n, [9])

    def test_short_gap_count_is_not_always_spike_loss_count(self):
        # Two short input gaps, but a toy dead-time counter rejects only one pulse.
        self.assertEqual(accepted_count([0, 15, 30]), 2)


if __name__ == "__main__":
    unittest.main()
