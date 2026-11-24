#!/usr/bin/env bash
# One-command pipeline. Put your dataset in data/raw/<class>/ first
set -e
python src/build_features.py     # images  -> data/features.csv
python src/eda.py                # EDA figures -> results/
python src/train_compare.py      # train + tune + compare 3 models
echo "Done. Try:  streamlit run app.py   or   python src/predict.py <image>"
