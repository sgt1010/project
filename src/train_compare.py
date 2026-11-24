"""
train_compare.py -- STEPS 3-5: train, tune, evaluate and compare 3 models
=========================================================================
Models (all from the syllabus -- Classification unit)
-----------------------------------------------------
  1. K-Nearest Neighbours (KNN)   -- instance-based, non-parametric
  2. Support Vector Machine (SVM) -- max-margin classifier, RBF kernel
  3. Decision Tree                -- rule-based, interpretable

Why these three?  They represent 3 different families of classifier:
  distance-based  (KNN)  vs  margin-based (SVM)  vs  rule-based (Tree).
They also behave very differently on scaled features, so the comparison
is genuinely informative.

Complexity cheat-sheet  (n = train samples, d = features, k = neighbours,
                          s = support vectors, depth = tree depth)
 +---------------+--------------------+--------------------+-------------+
 | Model         | Training           | Prediction         | Memory      |
 +---------------+--------------------+--------------------+-------------+
 | KNN           | O(1) (just stores) | O(n * d) per query | O(n * d)    |
 | SVM (RBF)     | ~O(n^2 .. n^3)     | O(s * d) per query | O(s * d)    |
 | Decision Tree | O(n * d * log n)   | O(depth)           | O(#nodes)   |
 +---------------+--------------------+--------------------+-------------+

Pipeline of this script
-----------------------
  features.csv --> stratified train/test split (80/20)
               --> for each model: Pipeline(scaler -> model)
               --> GridSearchCV (stratified 5-fold CV on TRAIN only)
               --> refit best params --> evaluate on untouched TEST set
               --> metrics + confusion matrix + ROC + decision surface
               --> save best model (by test macro-F1) to models/best_model.joblib

Metrics (syllabus: "Model evaluation")
  precision = TP/(TP+FP)   recall = TP/(TP+FN)   F1 = 2PR/(P+R)
  accuracy  = (TP+TN)/N    AUC = area under ROC (one-vs-rest if >2 classes)
  Macro averaging = every class counts equally (good for imbalance).
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"   # also applies to joblib workers
warnings.filterwarnings("ignore")

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
from config import (FEATURE_CSV, RESULTS_DIR, MODEL_DIR, TEST_SIZE, CV_FOLDS,
                    RANDOM_STATE)

# ---------------------------------------------------------------- model zoo
# Each entry: (pipeline, hyper-parameter grid searched with cross-validation)
MODELS = {
    "KNN": (
        Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [3, 5, 7, 11, 15],
         "clf__weights": ["uniform", "distance"],
         "clf__p": [1, 2]},                        # Manhattan vs Euclidean
    ),
    "SVM": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        {"clf__C": [0.1, 1, 10, 100],
         "clf__kernel": ["rbf", "linear"],
         "clf__gamma": ["scale", 0.01]},
    ),
    "Decision Tree": (
        # Trees are scale-invariant, but we keep the same pipeline shape.
        Pipeline([("scale", StandardScaler()),
                  ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        {"clf__max_depth": [3, 5, 8, 12, None],
         "clf__min_samples_leaf": [1, 3, 5],
         "clf__criterion": ["gini", "entropy"]},
    ),
}


def auc_score(y_true, proba, n_classes):
    """Binary -> use P(class 1); multi-class -> one-vs-rest macro AUC."""
    if n_classes == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro")


def main():
    RESULTS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------- load + split
    df = pd.read_csv(FEATURE_CSV)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df.drop(columns=["label", "file"]).values
    classes = list(le.classes_); K = len(classes)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[i] train={len(X_tr)} test={len(X_te)} classes={classes}")

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results, fitted, roc_data = [], {}, {}

    for name, (pipe, grid) in MODELS.items():
        print(f"\n=== {name} ===")
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_tr, y_tr)                    # CV happens ONLY on training data
        train_time = time.time() - t0
        best = gs.best_estimator_
        fitted[name] = best

        t0 = time.time()
        pred = best.predict(X_te)
        proba = best.predict_proba(X_te)
        infer_ms = (time.time() - t0) / len(X_te) * 1000

        row = {
            "Model": name,
            "Best params": json.dumps({k.replace("clf__", ""): v
                                       for k, v in gs.best_params_.items()}),
            "CV F1 (mean)": gs.best_score_,
            "CV F1 (std)": gs.cv_results_["std_test_score"][gs.best_index_],
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, average="macro", zero_division=0),
            "Recall": recall_score(y_te, pred, average="macro", zero_division=0),
            "F1": f1_score(y_te, pred, average="macro"),
            "AUC": auc_score(y_te, proba, K),
            "Train time (s)": train_time,
            "Predict ms/img": infer_ms,
        }
        results.append(row)
        roc_data[name] = proba
        print("best params:", row["Best params"])
        print(classification_report(y_te, pred, target_names=classes, digits=3))

        # confusion matrix
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y_te, pred),
                               display_labels=classes).plot(ax=ax, cmap="Blues",
                                                            colorbar=False)
        ax.set_title(f"Confusion matrix - {name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"cm_{name.replace(' ', '_').lower()}.png", dpi=130)
        plt.close()

    # ------------------------------------------------------ comparison table
    res = pd.DataFrame(results).round(4)
    res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n=========== MODEL COMPARISON (test set) ===========")
    print(res.drop(columns=["Best params"]).to_string(index=False))

    # bar chart of the 5 headline metrics
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    ax = res.set_index("Model")[metrics].plot.bar(figsize=(8, 4.5), rot=0)
    ax.set_ylim(max(0, res[metrics].values.min() - 0.1), 1.02)
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="lower right", ncol=5, fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=130); plt.close()

    # ----------------------------------------------------------- ROC curves
    plt.figure(figsize=(6, 5))
    Yb = label_binarize(y_te, classes=range(K)) if K > 2 else None
    for name, proba in roc_data.items():
        if K == 2:
            fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
        else:   # micro-average ROC over all classes
            fpr, tpr, _ = roc_curve(Yb.ravel(), proba.ravel())
        plt.plot(fpr, tpr, label=f"{name} (AUC={res.set_index('Model').loc[name, 'AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves" + (" (micro-avg)" if K > 2 else "")); plt.legend()
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=130); plt.close()

    # ----------------------------------- decision surfaces (syllabus topic)
    # Project to 2-D with PCA, re-fit each classifier's best params there and
    # colour the plane by predicted class. Shows HOW each model carves up space:
    #   KNN  -> jagged, local     SVM -> smooth curved     Tree -> axis-aligned boxes
    scaler2 = StandardScaler().fit(X_tr)
    pca = PCA(2, random_state=RANDOM_STATE).fit(scaler2.transform(X_tr))
    P_tr = pca.transform(scaler2.transform(X_tr))
    xx, yy = np.meshgrid(
        np.linspace(P_tr[:, 0].min() - 1, P_tr[:, 0].max() + 1, 250),
        np.linspace(P_tr[:, 1].min() - 1, P_tr[:, 1].max() + 1, 250))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, est) in zip(axes, fitted.items()):
        clf2 = type(est.named_steps["clf"])(**est.named_steps["clf"].get_params())
        clf2.fit(P_tr, y_tr)
        Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.25, cmap="viridis")
        ax.scatter(P_tr[:, 0], P_tr[:, 1], c=y_tr, s=10, cmap="viridis", edgecolor="none")
        ax.set_title(f"Decision surface - {name}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "decision_surfaces.png", dpi=130); plt.close()

    # ------------------------------------- Decision-tree feature importance
    tree = fitted["Decision Tree"].named_steps["clf"]
    imp = pd.Series(tree.feature_importances_).nlargest(10)
    plt.figure(figsize=(6, 4))
    imp[::-1].plot.barh()
    plt.xlabel("Importance"); plt.ylabel("feature index (f#)")
    plt.title("Top-10 features (Decision Tree)")
    plt.yticks(range(len(imp)), [f"f{i}" for i in imp.index[::-1]])
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "tree_feature_importance.png", dpi=130); plt.close()

    # ------------------------------------------------------------- save best
    best_name = res.sort_values(["F1", "AUC"], ascending=False).iloc[0]["Model"]
    joblib.dump({"model": fitted[best_name], "classes": classes, "name": best_name},
                MODEL_DIR / "best_model.joblib")
    for m_name, m_est in fitted.items():
        fname = m_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump({"model": m_est, "classes": classes, "name": m_name}, MODEL_DIR / fname)

    print(f"\n[+] Best model by test macro-F1: {best_name} -> models/best_model.joblib")


if __name__ == "__main__":
    main()
