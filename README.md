# Road Damage & Pothole Detection using Classical Machine Learning

Classifies road-surface images as **normal / pothole / (crack)** using hand-crafted
image features and three syllabus classifiers: **KNN, SVM and Decision Tree**.

# 1. Problem statement & objectives

Potholes and cracks cause accidents, vehicle damage and costly repairs. Manual road
surveys are slow and expensive. **Goal:** build an automatic, lightweight system that
decides from a single road photo whether the surface is damaged.

Objectives
1. Collect a labelled road-image dataset.
2. Pre-process images and extract numeric features; run exploratory data analysis (EDA).
3. Train, tune (cross-validation) and compare **3 ML algorithms**.
4. Evaluate with accuracy, precision, recall, F1 and AUC.
5. Deliver a working prototype (CLI + Streamlit web app).

**Real-world use cases:** municipal road-maintenance prioritisation, dashcam / fleet
monitoring, insurance claim evidence, driver-assist warnings, smart-city dashboards.

# 2. Pipeline


 raw images (data/raw/<class>/)
        |
        v   build_features.py
  ---------------------------------------------------------
 | gray -> resize 128x128 -> CLAHE -> feature extraction   |
 | intensity stats | histogram | edges | dark-blob shape   |
 | LBP texture     | HOG                                   |
  ---------------------------------------------------------
        |  data/features.csv  (~329 features / image)
        v   eda.py            -> results/eda_*.png
        v   train_compare.py
 stratified 80/20 split
   |-- TRAIN (80%) --> StandardScaler -> model --> GridSearchCV (5-fold CV)
   '-- TEST  (20%) --> final, untouched evaluation
        v
 metrics + confusion matrices + ROC + decision surfaces
        v
 models/best_model.joblib --> predict.py / app.py (prototype)


# 3. Models & methodology

| Model | Family | Key hyper-parameters tuned | Train cost | Predict cost |
|---|---|---|---|---|
| "KNN" | distance / instance-based | k, weights, distance (L1/L2) | O(1) | O(n·d) |
| "SVM" | max-margin, kernel | C, kernel (RBF/linear), gamma | ~O(n²–n³) | O(s·d) |
| "Decision Tree" | rule-based, interpretable | depth, min leaf, gini/entropy | O(n·d·log n) | O(depth) |

*n = training samples, d = features, s = support vectors.*

* Why these three? One model from each family (distance, margin, rules) gives a
  meaningful comparison; Naive Bayes was left out because many of our features
  (HOG bins, histogram bins) are strongly correlated, which violates its independence
  assumption (see `results/eda_correlation.png`).
* Scaling: `StandardScaler` inside a `Pipeline`, fitted only on training folds, so
  there is no data leakage into cross-validation.
* Model selection: `GridSearchCV` with stratified 5-fold CV, scored by macro-F1.
* Evaluation: held-out 20 % test set; macro-averaged precision / recall / F1,
  accuracy, and (one-vs-rest) AUC. Decision surfaces are drawn on a 2-D PCA projection.

# 4. Repository layout
pothole-detection/
|-- README.md
|-- requirements.txt
|-- run_all.sh                 # whole pipeline in one command
|-- app.py                     # Streamlit prototype
|-- data/
|   |-- DATASET.md             # where to get data + folder layout
|   '-- raw/<class>/*.jpg      # (git-ignored) images
|-- src/
|   |-- config.py              # all settings
|   |-- generate_synthetic_data.py
|   |-- prepare_dataset.py     # flatten a Kaggle download into data/raw/
|   |-- features.py            # image -> feature vector
|   |-- build_features.py      # step 1
|   |-- eda.py                 # step 2
|   |-- train_compare.py       # steps 3-5
|   '-- predict.py             # CLI prototype
|-- models/best_model.joblib   # created by training
'-- results/                   # figures + model_comparison.csv

# 5. How to run

```bash
# 0. install
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. get data  (see data/DATASET.md)
python src/prepare_dataset.py --src path/to/downloaded_dataset
#    ...or for a quick offline smoke test:
python src/generate_synthetic_data.py --n 300

# 2. run the pipeline
python src/build_features.py
python src/eda.py
python src/train_compare.py

# (or just: ./run_all.sh)

# 3. prototype
python src/predict.py path/to/road.jpg                 # CLI
python src/predict.py path/to/folder --save-annotated  # batch + annotated copies
streamlit run app.py                                   # web demo
```
# 6. Dataset

This project uses the real-world **Pothole Detection Dataset** from Kaggle (`atulyakumar98/pothole-detection-dataset`), consisting of **681 labelled road surface images**:
- **Normal (Intact Road):** 352 images (`data/raw/normal/`)
- **Pothole (Damaged Road):** 329 images (`data/raw/pothole/`)

Images include varied lighting, asphalt textures, shadows, and camera angles, split using a stratified **80% training / 20% testing** split.

# 7. Results

All three classifiers were tuned via **5-Fold Stratified Cross-Validation** (`GridSearchCV`) and evaluated on the held-out 20% test set (`results/model_comparison.csv`):

| Model | Best Hyperparameters | CV F1 (Mean ± Std) | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Predict Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | `kernel='rbf', C=10, gamma='scale'` | 0.9061 ± 0.0291 | **0.9343** | **0.9376** | **0.9329** | **0.9340** | **0.9782** | 0.18 ms/img |
| **KNN** | `n_neighbors=5, p=1, weights='uniform'` | 0.8303 ± 0.0280 | 0.8321 | 0.8572 | 0.8370 | 0.8303 | 0.9581 | 1.08 ms/img |
| **Decision Tree** | `criterion='entropy', max_depth=8, min_samples_leaf=3` | 0.8005 ± 0.0476 | 0.7883 | 0.7935 | 0.7856 | 0.7861 | 0.7956 | **0.02 ms/img** |

# Key Observations
* Best Overall Generalization (SVM): With an RBF kernel (`C=10`), SVM achieved **93.43% accuracy** and **0.9782 AUC**, cleanly separating non-linear combinations of HOG, LBP texture, edge density, and dark-blob features in high-dimensional space.
* Curse of Dimensionality & Latency (KNN): Using Manhattan distance (`p=1`, `k=5`), KNN achieved solid ranking ability (`0.9581` AUC) and 83.21% accuracy, but had the highest prediction latency (`1.08 ms/img`) because it computes distances against all training samples at inference time.
* Speed vs. Overfitting Trade-off (Decision Tree): Pruned to `max_depth=8` using information gain (`entropy`), the Decision Tree was nearly 10x faster at inference (`0.019 ms/img`) than SVM and provided interpretable feature-importance rules, but achieved lower test accuracy (`78.83%`) due to axis-aligned boundary limitations on noisy real-world shadows and pavement textures.

# 8. Limitations & future work
* Image-level classification: it says *whether* damage exists, not where.
  Next step: sliding window or YOLO-style detection for bounding boxes.
* Hand-crafted features are sensitive to shadows, wet roads and camera angle -
  try data augmentation and more diverse data.
* Add LASSO-regularised logistic regression for feature selection, or Naive Bayes
  as an extra baseline.
* Combine with GPS metadata to build a city pothole map.

## 9. Syllabus mapping
Classification (KNN, SVM, Decision Tree) | training/testing classifier models |
cross-validation | precision, recall, F1, accuracy, AUC | decision surfaces |
regularisation (SVM `C`, tree depth / leaf size).
