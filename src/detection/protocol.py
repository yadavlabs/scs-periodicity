"""Versioned constants shared across population pipeline stages."""

DETECTION_PROTOCOL_VERSION = "canonical-cv0_tol0p01_peakfirst-imax_v3"
CV0_PROTOCOL_VERSION = "canonical-cv0_tol0p01_v1"
POPULATION_THRESHOLD_TOL_UA = 0.01
DEFAULT_EXPECTED_INDEX_AXON_ID = "S01_140um_ph1"

# I_max is selected from the initial recruitment branch without consulting the
# arbitrary 2000-uA endpoint or a later suprastimulation branch.  A clear early
# response-count peak is preferred; otherwise the strictest stable pre-decline
# saturation plateau is used.  The 25-uA branch scan is followed by a local
# 0.01-uA threshold refinement.
I_MAX_COARSE_STEP_UA = 25.0
I_MAX_BRANCH_SEARCH_WIDTH_UA = 500.0
I_MAX_EARLY_PEAK_SEARCH_WIDTH_UA = 300.0
I_MAX_EARLY_PEAK_TAIL_SAMPLES = 3
I_MAX_EARLY_PEAK_MINIMUM_DROP_SPIKES = 1
I_MAX_EARLY_PEAK_MINIMUM_RISE_FRACTION = 0.05
I_MAX_PLATEAU_CANDIDATE_WIDTH_UA = 100.0
I_MAX_PLATEAU_CONFIRMATION_WIDTH_UA = 100.0
I_MAX_PLATEAU_TOLERANCE_FRACTIONS = (0.0, 0.01, 0.02)
I_MAX_DECLINE_WIDTH_UA = 50.0

# Retained only for the archived stable-near-maximum diagnostic helper.  The
# production peak-first definition above does not use a sampled global maximum.
I_MAX_PLATEAU_WIDTH_UA = 100.0
I_MAX_MIN_SAMPLED_MAX_FRACTION = 0.95
