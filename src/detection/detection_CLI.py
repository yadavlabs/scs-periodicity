"""Command-line wrapper for index-anchored population detection.

Examples:
    python -m detection.detection_CLI identify-index --n-jobs 4
    python -m detection.detection_CLI index-regimes --n-jobs 4
    python -m detection.detection_CLI simulate-population --n-jobs 4
    python -m detection.detection_CLI simulate-population-cv0 --n-jobs 4

The final command is intentionally separate because it is the long-running
recruitment simulation.  Outputs are written under data/detection/.
"""

from __future__ import annotations

from detection.detection_core import main


if __name__ == "__main__":
    raise SystemExit(main())
