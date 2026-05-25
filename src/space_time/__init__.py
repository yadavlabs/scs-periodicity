"""Space-time extracellular potential helpers for stage 02."""

from .space_time_core import (
    DEFAULT_CV_VALUES,
    DEFAULT_FREQUENCIES_HZ,
    ComsolProfiles,
    SpaceTimeConfig,
    SpaceTimeData,
    SpaceTimeResult,
    SpaceTimeStatus,
    describe_spacetime_npz,
    expected_output_paths,
    expected_time_pattern_paths,
    generate_all_spacetime,
    generate_spacetime_for_pair,
    load_comsol_profiles,
    load_spacetime_npz,
    load_time_pattern_currents,
    load_time_pattern_repetition,
    map_current_to_spacetime,
    run_stage,
    spacetime_output_path,
    time_pattern_input_path,
    verify_spacetime_outputs,
)

_PLOT_EXPORTS = {
    "plot_comsol_profiles": "plot_comsol_profiles",
    "plot_spacetime_construction": "plot_spacetime_construction",
    "plot_spacetime_pub": "plot_spacetime_pub",
}


def __getattr__(name):
    if name in _PLOT_EXPORTS:
        from . import space_time_plot

        value = getattr(space_time_plot, _PLOT_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_CV_VALUES",
    "DEFAULT_FREQUENCIES_HZ",
    "ComsolProfiles",
    "SpaceTimeConfig",
    "SpaceTimeData",
    "SpaceTimeResult",
    "SpaceTimeStatus",
    "describe_spacetime_npz",
    "expected_output_paths",
    "expected_time_pattern_paths",
    "generate_all_spacetime",
    "generate_spacetime_for_pair",
    "load_comsol_profiles",
    "load_spacetime_npz",
    "load_time_pattern_currents",
    "load_time_pattern_repetition",
    "map_current_to_spacetime",
    "plot_comsol_profiles",
    "plot_spacetime_construction",
    "plot_spacetime_pub",
    "run_stage",
    "spacetime_output_path",
    "time_pattern_input_path",
    "verify_spacetime_outputs",
]
