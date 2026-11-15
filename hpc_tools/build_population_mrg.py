"""Generate the population HOC from a locally installed ModelDB MRG model.

The generated HOC is a local build artifact. The release contains this
transformation, not a copy of the upstream model or its derived HOC file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one {label} in the pinned MRGaxon.hoc; found {count}")
    return result


def population_template(upstream: str) -> str:
    """Apply only the changes required for population geometry and batch use."""
    text = upstream.replace("\r\n", "\n").replace("\r", "\n")
    text = _replace_once(text, r'^\s*(?://\s*)?load_proc\("nrnmainmenu"\)\s*\n', "", "GUI loader")
    text = _replace_once(text, r'\bproc\s+model_globels\s*\(', "proc model_globals(", "globals procedure")
    text = _replace_once(text, r'^model_globels\s*\(\s*\)', "model_globals()", "globals call")
    assignments = {
        "axonnodes": "mrg_population_axonnodes",
        "paranodes1": "2*(axonnodes-1)",
        "paranodes2": "2*(axonnodes-1)",
        "axoninter": "6*(axonnodes-1)",
        "axontotal": "axonnodes+paranodes1+paranodes2+axoninter",
        "fiberD": "mrg_population_fiberD",
    }
    for name, expression in assignments.items():
        text = _replace_once(
            text,
            rf'^(\s*{name}\s*=\s*)[^\n/]+(?:\s*//[^\n]*)?$',
            lambda match, value=expression: match.group(1) + value,
            name,
        )
    # The original HOC adds a fixed node-10 IClamp and GUI after initialize().
    # Population stimulation instead plays extracellular potentials at each node.
    text = _replace_once(
        text,
        r'(?ms)^initialize\(\)\s*\n(?:.|\n)*\Z',
        "initialize()\n",
        "interactive stimulus section",
    ) if text.strip().splitlines()[-1].strip() != "initialize()" else text
    if not text.endswith("\n"):
        text += "\n"
    return text


def build(source: Path, output: Path) -> Path:
    template = population_template(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template, encoding="utf-8", newline="\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "hpc_tools/MRG_model/MRGaxon.hoc")
    parser.add_argument("--output", type=Path, default=ROOT / "hpc_tools/MRG_model/MRGaxon_population_template.hoc")
    args = parser.parse_args()
    print(build(args.source, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
