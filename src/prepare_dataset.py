"""
prepare_dataset.py -- flatten a downloaded dataset into data/raw/<class>/

Most Kaggle pothole sets look like:
    archive/train/Plain/*.jpg   archive/train/Pothole/*.jpg
    archive/test/Plain/*.jpg    archive/test/Pothole/*.jpg
This script merges train+test (we make our OWN stratified split later, so the
test set is never tuned on) and renames folders to clean class names.

Usage:
    python src/prepare_dataset.py --src ~/Downloads/archive
    python src/prepare_dataset.py --src ~/Downloads/archive --max-per-class 1500
"""
import argparse
import random
import shutil
from pathlib import Path
from config import RAW_DIR, VALID_EXT, RANDOM_STATE

# folder-name aliases -> canonical class name
ALIASES = {"plain": "normal", "normal": "normal", "negative": "normal", "no_pothole": "normal",
           "pothole": "pothole", "potholes": "pothole", "positive": "pothole",
           "crack": "crack", "cracks": "crack"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="optional cap, handy for quick experiments")
    a = ap.parse_args()

    buckets = {}
    for p in Path(a.src).rglob("*"):
        if p.suffix.lower() in VALID_EXT and p.parent.name.lower() in ALIASES:
            buckets.setdefault(ALIASES[p.parent.name.lower()], []).append(p)
    if not buckets:
        raise SystemExit("No recognised class folders (plain/normal/pothole/crack...). "
                         "Edit ALIASES in this file to match your dataset.")

    random.seed(RANDOM_STATE)
    for cls, files in buckets.items():
        random.shuffle(files)
        files = files[: a.max_per_class] if a.max_per_class else files
        out = RAW_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            shutil.copy(f, out / f"{cls}_{i:05d}{f.suffix.lower()}")
        print(f"[+] {cls}: copied {len(files)} images -> {out}")


if __name__ == "__main__":
    main()
