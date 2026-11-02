"""Audit example recruitment and the S7 counting reference without new simulations."""
from pathlib import Path
import numpy as np
import pandas as pd


def accepted_count(starts, refractory_samples=22):
    """Toy non-paralyzable dead-time counter; not an axon biophysics model."""
    last = -refractory_samples
    count = 0
    for start in starts:
        if start - last >= refractory_samples:
            count += 1
            last = start
    return count


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / "results/population/figures_v3"
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(root / "data/detection/population/index_regime_amplitudes.csv")
    index = index[index.thresh_type.eq("I_all")]
    records = []
    for (freq, cv), panel in index.groupby(["freq", "cv"]):
        path = root / f"data/time-patterns/population/cv{cv:.1f}_f{int(freq)}_waveforms.npz"
        with np.load(path) as bundle:
            for row in panel.itertuples():
                waveform = np.asarray(bundle[row.rep_label])
                active = waveform < 0
                starts = np.flatnonzero(np.diff(np.r_[False, active].astype(int)) == 1)
                gaps = np.diff(starts)
                denominator = row.n_input_pulses  # Match the published plot's nominal count.
                actual = row.n_index_spikes / denominator
                lt = int((gaps < 22).sum())
                equal = int((gaps == 22).sum())
                records.append(dict(freq=freq, cv=cv, rep_index=row.rep_index,
                                    n_input_pulses=denominator, n_detected_onsets=len(starts),
                                    n_short_lt=lt, n_equal=equal,
                                    short_fraction=lt / denominator, actual_fraction=actual,
                                    residual_lt=actual - (1 - lt / denominator),
                                    residual_le=actual - (1 - (lt + equal) / denominator),
                                    toy_deadtime_fraction=accepted_count(starts) / denominator))
    data = pd.DataFrame(records)
    data.to_csv(out / "audit_suppfig7_cutoff_sensitivity.csv", index=False)
    print(data.groupby("freq")[["n_equal", "residual_lt", "residual_le"]].mean().to_string())
    data["short_fraction_bin"] = pd.cut(data.short_fraction, [-.001, .05, .1, .15, .2, 1])
    print(data.groupby("short_fraction_bin", observed=True)[["residual_lt", "residual_le"]].agg(["mean", "count"]).to_string())
    print("Sequences affected by equality:", int((data.n_equal > 0).sum()), "/", len(data))
    print("Sequences with nominal/detected onset mismatch:", int((data.n_input_pulses != data.n_detected_onsets).sum()))
    features = pd.read_csv(root / "data/features/population/population_cv0_based_features.csv")
    example = features[(features.freq == 200) & (features.rep_index == 0)]
    summary = example.groupby("thresh_type")[["active_axon_count", "total_population_spikes", "n_events_10ms"]].agg(["min", "median", "max"])
    summary.to_csv(out / "audit_suppfig5_candidate_regimes.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
