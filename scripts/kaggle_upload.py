"""kaggle_upload.py — Upload the image dataset and ImageNet weights to Kaggle.

OPERATIONAL (not part of the training pipeline). Creates/updates two Kaggle
Datasets so the notebook can attach them (Add Input) and `config.py` auto-detects
them at `/kaggle/input/`:

  - images  : `data_local/`           → preserves the BeefCattle_Muzzle_Individualized/ folder
                                        → /kaggle/input/<slug>/BeefCattle_Muzzle_Individualized/
  - weights : `imagenet-pretrained/`  → .pth files directly under the slug
                                        → /kaggle/input/<slug>/vgg16_bn-...pth

Paths are derived from `config.py` (single source of truth), not hardcoded.

Credentials (DO NOT commit), any of:
  - `KAGGLE_API_TOKEN=KGAT_...` (new Kaggle token; requires kagglehub>=0.4.1),
  - `KAGGLE_USERNAME` + `KAGGLE_KEY` in the environment,
  - `~/.kaggle/kaggle.json`.
Generate the token at kaggle.com → Settings → API. The dataset owner (--user) must
match your Kaggle username.

Run from the local checkout that has the data (not from Kaggle or a worktree):

    pip install kagglehub
    export KAGGLE_USERNAME=your_username KAGGLE_KEY=xxxxxxxx
    python scripts/kaggle_upload.py --user your_username
    python scripts/kaggle_upload.py --user your_username --only weights --version-notes "v2"

The slug does not affect auto-detection (config.py searches by folder/file name),
but a stable slug is convenient for reusing the dataset across notebooks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def _require_auth() -> None:
    """Fail early and clearly if Kaggle credentials are missing."""
    # New API token (KGAT_...): requires kagglehub >= 0.4.1.
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return
    raise SystemExit(
        "Missing Kaggle credentials. Options: (1) KAGGLE_API_TOKEN=KGAT_... "
        "(new token; needs kagglehub>=0.4.1), (2) KAGGLE_USERNAME + KAGGLE_KEY, "
        "or (3) ~/.kaggle/kaggle.json. Generate the token at kaggle.com → Settings → API."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"),
                    help="Kaggle username (or export KAGGLE_USERNAME).")
    ap.add_argument("--images-slug", default="cattle-muzzle-db",
                    help="Slug for the image dataset.")
    ap.add_argument("--weights-slug", default="cattle-imagenet-pretrained",
                    help="Slug for the ImageNet weights dataset.")
    ap.add_argument("--only", choices=["images", "weights"],
                    help="Upload only one (default: both).")
    ap.add_argument("--version-notes", default=None,
                    help="Version notes (if dataset already exists, creates a new version).")
    args = ap.parse_args()

    if not args.user:
        raise SystemExit("Provide --user or export KAGGLE_USERNAME.")
    _require_auth()

    import kagglehub  # late import: only when actually uploading.

    # Paths derived from config (single source of truth).
    images_dir = config.DATA_DIR.parent          # data_local/ (contains the dataset folder)
    weights_dir = config.PRETRAINED_DIR           # imagenet-pretrained/ (contains .pth files)

    jobs: list[tuple[str, str, Path]] = []
    if args.only in (None, "images"):
        if not config.DATA_DIR.is_dir():
            raise SystemExit(f"Image dataset not found at {config.DATA_DIR}.")
        jobs.append(("images", f"{args.user}/{args.images_slug}", images_dir))
    if args.only in (None, "weights"):
        if weights_dir is None or not Path(weights_dir).is_dir():
            raise SystemExit(
                "Weights not found (config.PRETRAINED_DIR is None). Download the .pth files to "
                "imagenet-pretrained/ first (see README) or set CATTLE_PRETRAINED_DIR."
            )
        jobs.append(("weights", f"{args.user}/{args.weights_slug}", Path(weights_dir)))

    for label, handle, path in jobs:
        print(f"\n=== Uploading {label}: {handle}  ←  {path} ===")
        kwargs = {"version_notes": args.version_notes} if args.version_notes else {}
        kagglehub.dataset_upload(handle, str(path), **kwargs)
        print(f"OK → https://www.kaggle.com/datasets/{handle}")

    print("\nDone. In the Kaggle notebook: Add Input → search for the slugs above.")
    print("config.py will auto-detect them (by folder/file name).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
