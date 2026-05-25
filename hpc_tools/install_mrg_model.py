"""Download the upstream MRG axon model used by the optional NEURON workflow."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELDB_ACCESSION = "3810"
MODELDB_REPOSITORY = "https://github.com/ModelDBRepository/3810"
MODELDB_COMMIT = "b15daeda6ea368df0198d257f339cf5e840cf31c"
MODELDB_ARCHIVE = f"{MODELDB_REPOSITORY}/archive/{MODELDB_COMMIT}.zip"
MODEL_FILES = ("AXNODE.mod", "MRGaxon.hoc", "MRGaxon.ses", "mosinit.hoc", "README")


def install_model(target: Path, *, force: bool = False) -> None:
    existing = [target / name for name in MODEL_FILES if (target / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"MRG model files already exist in {target}. Use --force to replace them."
        )

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mrg-model-") as temp_dir:
        archive = Path(temp_dir) / "modeldb-3810.zip"
        print(f"Downloading ModelDB accession {MODELDB_ACCESSION} at commit {MODELDB_COMMIT}")
        urllib.request.urlretrieve(MODELDB_ARCHIVE, archive)
        with zipfile.ZipFile(archive) as zip_file:
            prefix = f"3810-{MODELDB_COMMIT}/"
            missing = [name for name in MODEL_FILES if prefix + name not in zip_file.namelist()]
            if missing:
                raise FileNotFoundError(
                    "ModelDB archive did not contain expected files: " + ", ".join(missing)
                )
            for name in MODEL_FILES:
                output = target / name
                with zip_file.open(prefix + name) as source, output.open("wb") as destination:
                    shutil.copyfileobj(source, destination)

    provenance = target / "MODELDB_SOURCE.txt"
    provenance.write_text(
        "Downloaded for optional HPC regeneration; not distributed as project-authored code.\n"
        f"ModelDB accession: {MODELDB_ACCESSION}\n"
        f"Repository: {MODELDB_REPOSITORY}\n"
        f"Commit: {MODELDB_COMMIT}\n"
        "Citation: McIntyre CC, Richardson AG, and Grill WM. Modeling the excitability "
        "of mammalian nerve fibers: influence of afterpotentials on the recovery cycle. "
        "Journal of Neurophysiology 87:995-1006, 2002.\n",
        encoding="utf-8",
    )
    print(f"Installed MRG model files in {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the pinned upstream MRG model from ModelDB for HPC regeneration."
    )
    parser.add_argument(
        "--target",
        default=str(REPO_ROOT / "hpc_tools" / "MRG_model"),
        help="Directory receiving the ModelDB model files.",
    )
    parser.add_argument("--force", action="store_true", help="Replace any existing model files.")
    args = parser.parse_args()
    install_model(Path(args.target).resolve(), force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
