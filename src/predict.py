"""
predict.py -- CLI prototype: classify one image or a whole folder.

    python src/predict.py path/to/road.jpg
    python src/predict.py path/to/folder/ --save-annotated

Prints the predicted class + confidence. With --save-annotated it writes a
copy of each image with the label burned in (results/predictions/).
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"   # also applies to joblib workers
warnings.filterwarnings("ignore")

import argparse
import cv2
import joblib
from pathlib import Path
from config import MODEL_DIR, RESULTS_DIR, VALID_EXT
from features import extract_features


def load_model():
    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    return bundle["model"], bundle["classes"], bundle["name"]


def predict_image(img, model, classes):
    """Returns (label, confidence, {class: prob})."""
    proba = model.predict_proba(extract_features(img).reshape(1, -1))[0]
    i = int(proba.argmax())
    return classes[i], float(proba[i]), dict(zip(classes, map(float, proba)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--save-annotated", action="store_true")
    a = ap.parse_args()

    model, classes, name = load_model()
    print(f"[i] using {name}")
    p = Path(a.path)
    files = [f for f in p.rglob("*") if f.suffix.lower() in VALID_EXT] if p.is_dir() else [p]
    out = RESULTS_DIR / "predictions"
    if a.save_annotated:
        out.mkdir(parents=True, exist_ok=True)

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[!] cannot read {f}"); continue
        label, conf, _ = predict_image(img, model, classes)
        print(f"{f.name:30s} -> {label:10s} ({conf:.1%})")
        if a.save_annotated:
            color = (0, 0, 255) if label != "normal" else (0, 160, 0)
            cv2.putText(img, f"{label} {conf:.0%}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imwrite(str(out / f.name), img)


if __name__ == "__main__":
    main()
