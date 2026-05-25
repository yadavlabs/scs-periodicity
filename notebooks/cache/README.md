# Notebook Cache Files

This folder stores derived tables that make the review notebooks faster to run.
They are reproducible from the generated stage data, but are not primary
pipeline outputs.

- `feature_metric_summaries.csv`: rate and timing feature summaries used by
  `04_FeatureEngineering.ipynb`.
- `cache_detection_isi_rep_stats_absref_1.10ms.csv`: detection
  inter-spike-interval summary cache used by `06_FigureGeneration.ipynb`.
- `static-figures/`: static schematic panels extracted from the manuscript draft
  for display in `06_FigureGeneration.ipynb`.

The static figures are tracked in Git. Other CSV/PKL cache files are excluded
from Git, included in the release-data ZIP for rapid figure redraws, and can be
regenerated from the current `data/` bundle by running
`06_FigureGeneration.ipynb` with `USE_NB_CACHE = False`.
