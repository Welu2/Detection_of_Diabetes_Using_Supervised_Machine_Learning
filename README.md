# Detection of Diabetes Using Supervised Machine Learning

A small end-to-end project that explores, preprocesses, trains, and evaluates multiple supervised ML models for **binary diabetes prediction**.

This repository is organized around Jupyter notebooks (EDA → preprocessing → model training → evaluation) and includes saved preprocessing/model artifacts plus a set of generated plots and reports.

## What’s inside

- **Dataset**: `data/diabetes.csv`
- **Processed dataset artifact**: `data/processed_diabetes_data.pkl` (Joblib)
- **Preprocessing pipeline**: `src/preprocessing_pipeline.pkl` (Joblib)
- **Trained models** (Joblib):
  - `models/logistic_regression_model.pkl`
  - `models/random_forest_model.pkl`
  - `models/svm_model.pkl`
  - `models/xgboost_model.pkl`
- **Evaluation outputs**: `results/` (plots, CSV/XLSX summaries, and a text report)

## Models evaluated

The project evaluates multiple classical supervised learning models for binary classification, including:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

For a quick view of the latest metrics, see:

- `results/model_performance.csv`
- `results/comprehensive_evaluation_report.txt`

## Project structure

```text
Detection_of_Diabetes_Using_Supervised_Machine_Learning/
  check_data.py
  requirements.txt
  data/
    diabetes.csv
    processed_diabetes_data.pkl
  models/
    *.pkl
  notebook/
    eda.ipynb
    evaluation_analysis.ipynb
    debug_pkl_files.ipynb
  results/
    *.png
    model_performance.csv
    model_evaluation_summary.xlsx
    comprehensive_evaluation_report.txt
    summary_statistics.csv
  src/
    data_preprocessing.ipynb
    model_training.ipynb
    preprocessing_pipeline.pkl
```

## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```


## How to run

### Option A — Run the notebooks (recommended)

Open and run notebooks in this order:

1. `notebook/eda.ipynb` (exploration and basic analysis)
2. `src/data_preprocessing.ipynb` (cleaning/feature processing + pipeline creation)
3. `src/model_training.ipynb` (train models + persist artifacts)
4. `notebook/evaluation_analysis.ipynb` (compare models and generate figures/reports)

### Option B — Validate saved artifacts

Use `check_data.py` to sanity-check the serialized processed dataset and model files:

```bash
python check_data.py
```

## Outputs

The `results/` directory includes generated deliverables such as:

- Performance table (`model_performance.csv`)
- A consolidated report (`comprehensive_evaluation_report.txt`)
- Model comparison plots (`roc_curves.png`, `confusion_matrices.png`, etc.)
- Dataset visuals (`feature_distributions.png`, `correlation_heatmap.png`, etc.)

## Reproducibility notes

- Trained artifacts are stored as Joblib `.pkl` files in `models/` and `src/`.
- If you retrain models, outputs in `results/` may change depending on random seeds and library versions.
- If you want fully reproducible runs, consider pinning versions in `requirements.txt` and setting fixed `random_state` values in training.

## Troubleshooting

- If notebooks fail to import packages, ensure your notebook kernel is using the same environment where you ran `pip install -r requirements.txt`.
- If you see file-not-found errors, confirm you are running from the project root so relative paths like `data/processed_diabetes_data.pkl` resolve correctly.