"""
config.py -- single place for every tunable setting in the project.
Change values here instead of hunting through files.
"""
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"        # data/raw/<class_name>/*.jpg
FEATURE_CSV = ROOT / "data" / "features.csv"
RESULTS_DIR = ROOT / "results"
MODEL_DIR   = ROOT / "models"

IMG_SIZE     = 128      # every image is resized to IMG_SIZE x IMG_SIZE
TEST_SIZE    = 0.20     # 80/20 train/test split
CV_FOLDS     = 5        # k for stratified k-fold cross-validation
RANDOM_STATE = 42       # reproducibility

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
