"""Convenience exports for the discrimination stage."""

from .discrimination_core import (
    DEFAULT_FREQS,
    DiscriminationDataConfig,
    FeatureEngineeringConfig,
    FeatureSettings,
    FeatureSimulation,
    FeatureWorkflow,
    ModelSettings,
    collect_feature_evidence_matrices,
    compute_trial_feature_details,
    build_feature_metric_summary_cache,
    feature_metric_pivot,
    feature_metric_summary,
    feature_metric_summary_cached,
    feature_frequencies,
    feature_rep_options,
    feature_threshold_options,
    fixed_settings_table,
    load_discrimination_data,
    load_feature_simulation,
    make_feature_settings,
    model_comparison_pivot,
    model_comparison_table,
    model_performance_statistics_table,
    normalize_feature_evidence_summary,
    per_block_regime_weight_summary,
    prepare_feature_workflow,
    run_table,
    spike_count_pivot,
    spike_count_summary,
    threshold_display_name,
    threshold_short_label,
    trial_feature_details,
    trial_feature_evidence,
)

_PLOT_EXPORTS = {
    "build_feature_evidence_figure": "build_feature_evidence_figure",
    "make_feature_evidence_widget": "make_feature_evidence_widget",
    "plot_psychometric_fits": "plot_psychometric_fits",
    "plot_regime_mixture_comparison": "plot_regime_mixture_comparison",
}


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        from . import discrimination_plot

        value = getattr(discrimination_plot, _PLOT_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_FREQS",
    "DiscriminationDataConfig",
    "FeatureEngineeringConfig",
    "FeatureSettings",
    "FeatureSimulation",
    "FeatureWorkflow",
    "ModelSettings",
    "build_feature_evidence_figure",
    "collect_feature_evidence_matrices",
    "compute_trial_feature_details",
    "build_feature_metric_summary_cache",
    "feature_metric_pivot",
    "feature_metric_summary",
    "feature_metric_summary_cached",
    "feature_frequencies",
    "feature_rep_options",
    "feature_threshold_options",
    "fixed_settings_table",
    "load_discrimination_data",
    "load_feature_simulation",
    "make_feature_evidence_widget",
    "make_feature_settings",
    "model_comparison_pivot",
    "model_comparison_table",
    "model_performance_statistics_table",
    "normalize_feature_evidence_summary",
    "per_block_regime_weight_summary",
    "plot_psychometric_fits",
    "plot_regime_mixture_comparison",
    "prepare_feature_workflow",
    "run_table",
    "spike_count_pivot",
    "spike_count_summary",
    "threshold_display_name",
    "threshold_short_label",
    "trial_feature_details",
    "trial_feature_evidence",
]
