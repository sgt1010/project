# Dataset

The pipeline reads **any** folder layout of the form

```
data/raw/
 ├── normal/    *.jpg|png     (healthy road surface)
 ├── pothole/   *.jpg|png
 └── crack/     *.jpg|png     (optional extra class)
```

Folder name = class label. Two classes (pothole / normal) or more both work.

## Option A - Real dataset (use this for your final submission)

1. Download a pothole image-classification dataset from Kaggle, e.g. the
   "Pothole Detection Dataset" (plain-road vs pothole images, split into
   train/test, roughly 6,000 JPEGs incl. rainy, waterlogged and dashcam shots).
   Search Kaggle for *"Pothole Detection Dataset"*, or see the related
   `sachinpatel21/pothole-image-dataset` on Kaggle.
2. Unzip it, then merge/rename the folders into `data/raw/`:
   ```bash
   python src/prepare_dataset.py --src path/to/unzipped_dataset
   # optional cap for faster experiments:
   python src/prepare_dataset.py --src path/to/unzipped_dataset --max-per-class 1500
   ```
3. If you want a **third class** (cracks), add images from the road-damage
   datasets such as RDD2022 (crop the annotated crack boxes into `data/raw/crack/`).

Please check each dataset's licence on its Kaggle / Roboflow page and cite it in
your report. (Annotated object-detection sets such as the Roboflow
"Pothole Dataset" by Atikur Rahman Chitholian can also be used: crop the boxes
into `pothole/` and sample background patches into `normal/`.)

## Option B - Synthetic data (offline smoke test only)

```bash
python src/generate_synthetic_data.py --n 300
```
Generates 3 classes of artificial road textures. It is **much easier** than real
photos, so scores will be near-perfect - never report these as your final results.
