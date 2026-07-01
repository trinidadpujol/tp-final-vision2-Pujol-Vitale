"""download_zenodo.py — Resumable download of a Zenodo record.

Utility designed for large datasets that do NOT fit on local disk: run it directly
ON THE VM. Downloads all files in the record, verifies the md5 checksum published
by Zenodo and, optionally, extracts .zip archives.

Usage:
    # Cows Frontal Face Dataset (record 10535934), extracting the zip:
    python scripts/download_zenodo.py --record 10535934 --dest ~/data/cows_face --extract

    # Retry / force re-download:
    python scripts/download_zenodo.py --record 10535934 --dest ~/data/cows_face --extract --force

Notes:
  - Uses `wget -c` (RESUMABLE): if SSH drops or the download cuts, re-run the SAME
    command and it continues from where it left off. Recommended to run inside `tmux`.
  - Verifies md5 against what Zenodo publishes; aborts on mismatch.
  - No external dependencies: urllib + hashlib (stdlib) and system `wget`.
  - DOMAIN NOTE: record 10535934 is "Cows Frontal Face Dataset" (frontal FACES),
    a DIFFERENT modality from the muzzle pipeline. Download it to a separate folder
    (e.g. ~/data/cows_face), not to DATA_DIR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://zenodo.org/api/records/{record}"


def fetch_record(record: str) -> dict:
    """Return the record JSON (metadata + file list)."""
    with urllib.request.urlopen(API.format(record=record), timeout=60) as r:
        return json.load(r)


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    """md5 of a file read in chunks (works for files of several GB)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Resumable download with wget (-c). Raises if wget fails."""
    cmd = ["wget", "-c", "--tries=10", "--timeout=60", "-O", str(dest), url]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default="10535934", help="Zenodo record ID.")
    ap.add_argument("--dest", default="~/data/zenodo", help="Destination folder.")
    ap.add_argument("--extract", action="store_true", help="Extract .zip archives.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if md5 already matches.")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    rec = fetch_record(args.record)
    title = rec.get("metadata", {}).get("title", "?")
    files = rec.get("files", [])
    print(f"Record {args.record}: {title}")
    print(f"{len(files)} file(s) -> {dest}\n")

    for f in files:
        key = f["key"]
        url = f["links"]["self"]
        want_md5 = f.get("checksum", "").split(":")[-1]
        out = dest / key

        if out.is_file() and not args.force and want_md5 and md5sum(out) == want_md5:
            print(f"[OK] {key}: already downloaded and verified.")
        else:
            print(f"[>>] {key}: downloading (resumable)...")
            download(url, out)
            if want_md5:
                got = md5sum(out)
                if got != want_md5:
                    print(f"[ERROR] {key}: expected md5 {want_md5}, got {got}. "
                          f"Re-run the command to retry.", file=sys.stderr)
                    return 1
                print(f"[OK] {key}: md5 verified.")

        if args.extract and out.suffix.lower() == ".zip":
            print(f"[++] {key}: extracting to {dest} ...")
            with zipfile.ZipFile(out) as z:
                z.extractall(dest)
            print(f"[OK] {key}: extracted.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
