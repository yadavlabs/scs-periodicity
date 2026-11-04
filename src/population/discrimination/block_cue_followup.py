"""Fingerprint the fixed inputs/code for the two exploratory block-cue fits."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, root=None):
    manifest = json.loads(Path(path).read_text())
    root = Path(manifest["project"] if root is None else root)
    for relative, expected in manifest["sha256"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Follow-up input/code changed since preparation: {relative}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.out_root).resolve()
    path = out / "followup_inputs.json"
    if path.exists() or args.verify_only:
        verify(path, root=root)
        print("Follow-up fingerprints verified.")
        return
    from population.discrimination.discrimination_core import preflight
    evidence = root / "data/features/population_cv0_template_evidence.csv"
    behavior = root / "data/discrimination/slack"
    report = preflight(str(evidence), str(behavior), (20., 50., 100., 200.))
    if report["simulation_replicates"] != 100 or report["behavior_rows"] != 88:
        raise ValueError("Follow-up requires 100 repetitions and 88 behavioral conditions")
    free_weights = sum(len(classes) - 1 for classes in report["active_regime_classes_by_frequency"].values())
    if free_weights != 23:
        raise ValueError("Response equivalence changed; expected 23 shared mixture parameters")
    files = [evidence, root / "src/population/discrimination/discrimination_core.py",
             root / "src/population/discrimination/discrimination_CLI.py",
             root / "src/population/discrimination/behavioral_data.py", *sorted(behavior.glob("*.csv"))]
    report.update(project=str(root), model_families=["rate_timing", "timing_rate"],
                  parameters_per_model=29, mc_total=200, seed=0, train_fraction=.8,
                  purpose="Exploratory opposing block-cue assignments; not held-out animal validation",
                  sha256={str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in files})
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {path}")


if __name__ == "__main__":
    main()
