"""Make the verified population figure exports portable and check their manifest."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results/population/figures_v3"
OLD_ROOT = "C:\\Users\\tvrdy\\Desktop\\simulation-behavioral-prediction\\"
PATH_REMAP = {
    "results/population/detection/imax_population_refresh/population_responses.csv":
        "data/detection/population/population_responses.csv",
}


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def portable(value: object) -> object:
    if isinstance(value, dict):
        return {portable(key): portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.lower().startswith(OLD_ROOT.lower()):
        relative = value[len(OLD_ROOT):].replace("\\", "/")
        return PATH_REMAP.get(relative, relative)
    return value


def main() -> None:
    report = ROOT / "results/population/figure_verification/report.json"
    if json.loads(report.read_text(encoding="utf-8")).get("status") != "passed":
        raise ValueError("Figure notebook must pass before finalizing exports")

    for name in (
        "suppfig5_population_recruitment_data/_data/provenance.json",
        "suppfig5_population_recruitment_data/_data/response_cache_manifest.json",
    ):
        path = FIGURES / name
        data = portable(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    manifest = FIGURES / "figure_manifest_v3.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if not row["path"]:
            suffix = ".png" if row["kind"] == "figure" else ".csv"
            row["path"] = (FIGURES / f"{row['stem']}{suffix}").relative_to(ROOT).as_posix()
        else:
            row["path"] = row["path"].replace("\\", "/")
        path = ROOT / row["path"]
        if not path.is_file() or not path.resolve().is_relative_to(FIGURES.resolve()):
            raise ValueError(f"Figure manifest points outside verified exports: {row}")
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("kind", "stem", "path"))
        writer.writeheader()
        writer.writerows(rows)

    for path in FIGURES.rglob("*.json"):
        contents = path.read_text(encoding="utf-8")
        if "simulation-behavioral-prediction" in contents or "C:\\\\Users\\\\tvrdy" in contents:
            raise ValueError(f"Source-machine path remains in {path}")

    promotion = FIGURES / "promotion_manifest.json"
    files = sorted((path for path in FIGURES.rglob("*") if path.is_file() and path != promotion),
                   key=lambda path: path.relative_to(FIGURES).as_posix())
    entries = [{"path": path.relative_to(FIGURES).as_posix(), "sha256": digest(path)}
               for path in files]
    promotion.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(rows)} figure records and hashed {len(entries)} export files")


if __name__ == "__main__":
    main()
