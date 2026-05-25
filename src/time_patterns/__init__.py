"""Time-pattern generation helpers for stage 01."""

from .time_patterns_core import (
    DEFAULT_CV_VALUES,
    DEFAULT_FREQUENCIES_HZ,
    StimSpec,
    TimePatternConfig,
    TimePatternResult,
    TimePatternStatus,
    count_negative_phase_onsets,
    draw_gamma_times,
    expected_output_paths,
    expected_slack_paths,
    generate_all_time_patterns,
    generate_biphasic_waveform,
    load_slack_waveform_1ua,
    make_bundle_for_pair,
    pulse_start_indices_from_times,
    regular_train_times,
    run_stage,
    seed_for_pair,
    slack_path_format,
    validate_pulse_start_indices,
    verify_time_pattern_outputs,
)

_PLOT_EXPORTS = {
    "plot_time_pattern_example": "plot_time_pattern_example",
    "plot_waveform_construction": "plot_waveform_construction",
}


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        from . import time_patterns_plot

        value = getattr(time_patterns_plot, _PLOT_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_CV_VALUES",
    "DEFAULT_FREQUENCIES_HZ",
    "StimSpec",
    "TimePatternConfig",
    "TimePatternResult",
    "TimePatternStatus",
    "count_negative_phase_onsets",
    "draw_gamma_times",
    "expected_output_paths",
    "expected_slack_paths",
    "generate_all_time_patterns",
    "generate_biphasic_waveform",
    "load_slack_waveform_1ua",
    "make_bundle_for_pair",
    "pulse_start_indices_from_times",
    "plot_time_pattern_example",
    "plot_waveform_construction",
    "regular_train_times",
    "run_stage",
    "seed_for_pair",
    "slack_path_format",
    "validate_pulse_start_indices",
    "verify_time_pattern_outputs",
]
