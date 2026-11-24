"""
eda.py -- STEP 2: Exploratory Data Analysis
===========================================
Produces (in results/):
  eda_class_balance.png   -> bar chart of images per class
  eda_sample_images.png   -> a grid of random raw images per class
  eda_feature_boxplot.png -> distribution of key features by class
  eda_pca.png             -> 2-D PCA projection: are the classes separable?
  eda_correlation.png     -> correlation heat-map of the hand-crafted features

Why each plot matters
---------------------
 class balance  : an imbalanced set makes accuracy misleading (use F1/AUC)
 sample images  : sanity-check labels & image quality
 boxplots       : which features separate classes (e.g. dark-blob area)
 PCA            : rough idea of how hard classification will be
 correlation    : highly correlated features are redundant (matters for NB)
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import FEATURE_CSV, RAW_DIR, RESULTS_DIR, VALID_EXT, RANDOM_STATE

# names of the first 31 hand-crafted features (the rest are HOG bins)
NAMED = ["mean", "std", "median", "min", "max", "p10"] \
      + [f"hist{i}" for i in range(16)] \
      + ["edge_density", "grad_mean", "grad_std", "grad_p95"] \
      + ["blob_area", "blob_circularity", "blob_aspect", "n_blobs", "dark_ratio"]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    X = df.drop(columns=["label", "file"])
    print("[i] shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    print(df["label"].value_counts().to_string())
    print(X.iloc[:, :6].describe().round(2).to_string())

    # 1) class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="label", data=df, hue="label", legend=False)
    plt.title("Class balance"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_class_balance.png", dpi=130); plt.close()

    # 2) sample images (4 random per class)
    rng = np.random.default_rng(RANDOM_STATE)
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(classes), 4, figsize=(8, 2 * len(classes)))
    axes = np.atleast_2d(axes)
    for r, cls in enumerate(classes):
        files = [p for p in (RAW_DIR / cls).rglob("*") if p.suffix.lower() in VALID_EXT]
        for c, p in enumerate(rng.choice(files, 4, replace=False)):
            axes[r, c].imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(cls, loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "eda_sample_images.png", dpi=130); plt.close()

    # 3) box-plots of 6 interpretable features
    key = ["std", "edge_density", "blob_area", "blob_circularity", "dark_ratio", "grad_mean"]
    named = X.iloc[:, :len(NAMED)].copy(); named.columns = NAMED
    named["label"] = df["label"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, k in zip(axes.ravel(), key):
        sns.boxplot(x="label", y=k, data=named, hue="label", legend=False, ax=ax)
        ax.set_title(k)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "eda_feature_boxplot.png", dpi=130); plt.close()

    # 4) PCA 2-D
    Z = PCA(2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X))
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df["label"], s=18, alpha=.7)
    plt.title("PCA projection of features"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_pca.png", dpi=130); plt.close()

    # 5) correlation of named features
    plt.figure(figsize=(9, 7))
    sns.heatmap(named.drop(columns="label").corr(), cmap="coolwarm", center=0)
    plt.title("Feature correlation"); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "eda_correlation.png", dpi=130); plt.close()
    print(f"[+] EDA figures saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
