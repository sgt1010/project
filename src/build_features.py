"""
build_features.py -- STEP 1 of the pipeline
===========================================
Walks data/raw/<class_name>/*.jpg, extracts a feature vector from every
image (see features.py) and saves everything to data/features.csv.

Folder name  == class label.   Works for 2 classes (pothole / normal) or more
(pothole / crack / normal), so it fits most public datasets unchanged.

Time complexity: O(N * S^2)  for N images of S x S pixels (one pass each).
"""
import cv2
import pandas as pd
from config import RAW_DIR, FEATURE_CSV, VALID_EXT
from features import extract_features


def main():
    rows, skipped = [], 0
    classes = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit("No class folders in data/raw/. See data/DATASET.md "
                         "or run: python src/generate_synthetic_data.py")
    print(f"[i] classes found: {classes}")

    for cls in classes:
        files = [p for p in (RAW_DIR / cls).rglob("*") if p.suffix.lower() in VALID_EXT]
        for p in files:
            img = cv2.imread(str(p))
            if img is None:               # corrupted file -> skip, don't crash
                skipped += 1
                continue
            vec = extract_features(img)
            rows.append([cls, p.name, *vec])
        print(f"    {cls:10s}: {len(files)} images")

    n_feat = len(rows[0]) - 2
    df = pd.DataFrame(rows, columns=["label", "file"] + [f"f{i}" for i in range(n_feat)])
    df.to_csv(FEATURE_CSV, index=False)
    print(f"[+] saved {df.shape[0]} rows x {n_feat} features -> {FEATURE_CSV} "
          f"(skipped {skipped} unreadable)")


if __name__ == "__main__":
    main()
