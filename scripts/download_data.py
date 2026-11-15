"""Download the PAMAP2 Physical Activity Monitoring dataset from UCI.

Usage:
    python scripts/download_data.py

Result:
    data/raw/PAMAP2_Dataset/Protocol/subject101.dat ... subject109.dat
    data/raw/PAMAP2_Dataset/Optional/subject101.dat ...

Uses only the standard library. Safe to re-run: it skips work already done.
"""

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "pamap2.zip"
PROTOCOL_DIR = RAW_DIR / "PAMAP2_Dataset" / "Protocol"


def download(url: str, dest: Path) -> None:
    """Stream the file to disk with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            size = f"{done / 1e6:,.0f} MB" + (
                f" / {total / 1e6:,.0f} MB" if total else ""
            )
            print(f"\r  downloaded {size}", end="", flush=True)
    print()
    tmp.rename(dest)


def extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip; the UCI archive wraps the dataset in a second zip, so unpack those too."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
        inner = [dest / n for n in zf.namelist() if n.lower().endswith(".zip")]
    for z in inner:
        extract(z, dest)
        z.unlink()


def main() -> int:
    if PROTOCOL_DIR.exists() and len(list(PROTOCOL_DIR.glob("subject*.dat"))) == 9:
        print(f"Already present: {PROTOCOL_DIR}")
        return 0
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        download(URL, ZIP_PATH)
    print("Extracting...")
    extract(ZIP_PATH, RAW_DIR)
    files = sorted(PROTOCOL_DIR.glob("subject*.dat"))
    if len(files) != 9:
        print(
            f"Expected 9 protocol files in {PROTOCOL_DIR}, found {len(files)}",
            file=sys.stderr,
        )
        return 1
    ZIP_PATH.unlink()
    shutil.rmtree(RAW_DIR / "__MACOSX", ignore_errors=True)
    for f in files:
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size / 1e6:,.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
