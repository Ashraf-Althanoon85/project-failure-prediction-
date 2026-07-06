# Software Engineering Project Failure Prediction

A Decision-Oriented Machine Learning Framework for Predicting Software Project Failure Using Historical Project Data.

---

## Overview

Software project failure remains one of the major challenges in software engineering. This project proposes an intelligent decision-support framework that predicts software project risks using historical issue tracking data extracted from Agile software repositories.

Unlike traditional binary prediction systems, this framework combines multiple machine learning models with decision-oriented evaluation metrics to help project managers identify high-risk software projects at an early stage.

---

## Features

- Historical project data preprocessing
- Jira issue data cleaning
- Change-log integration
- Risk feature engineering
- Software project failure prediction
- Ensemble learning
- Probability calibration
- Risk ranking
- Explainable performance evaluation
- Decision-oriented reporting

---

## Project Pipeline

```
Raw TAWOS Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Project Filtering
        │
        ▼
Change Log Integration
        │
        ▼
Feature Engineering
        │
        ▼
Training Dataset
        │
 ┌──────┼──────────────┐
 │      │              │
 ▼      ▼              ▼
CatBoost XGBoost    TabNet
 │      │              │
 └──────┼──────────────┘
        ▼
 Weighted Ensemble
        ▼
Probability Calibration
        ▼
Risk Ranking
        ▼
Performance Evaluation
        ▼
Decision Support Report
```

---

## Dataset

The project uses the **TAWOS (A Versatile Dataset of Agile Open Source Software Projects)** dataset.

The raw dataset is processed to generate a clean tabular dataset suitable for machine learning.

Main preprocessing steps include:

- Missing value handling
- CSV repair
- Change-log parsing
- Sprint aggregation
- Feature engineering
- Date normalization

---

## Risk Features

The framework extracts several software engineering risk indicators, including:

- Issue Reopen Count
- Priority Change Count
- Resolution Time
- Sprint Issue Count
- Status Changes
- Assignee Changes
- Historical Issue Activity

---

## Machine Learning Models

The framework evaluates multiple predictive models:

- CatBoost
- XGBoost
- TabNet
- Weighted Soft Voting Ensemble

---

## Evaluation Metrics

The following evaluation metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- Recall@20%
- Confusion Matrix
- Threshold Calibration

---

## Folder Structure

```
Software-Project-Failure-Prediction/

│
├── data/
│   ├── raw/
│   ├── processed/
│   └── mesos_tabular_dataset.csv
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_catboost.py
│   ├── train_xgboost.py
│   ├── train_tabnet.py
│   ├── ensemble.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
│
├── reports/
│
├── figures/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/software-project-failure-prediction.git

cd software-project-failure-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python src/preprocessing.py

python src/feature_engineering.py

python src/train_catboost.py

python src/train_xgboost.py

python src/train_tabnet.py

python src/ensemble.py

python src/evaluation.py
```

---

## Output

The framework generates:

- Trained ML models
- Risk probability scores
- Ranked high-risk projects
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Performance comparison
- Risk prediction report

---

## Research Contribution

This framework introduces:

- Decision-oriented software risk prediction
- Historical Agile project mining
- Risk-aware feature engineering
- Ensemble learning for software engineering
- Ranking-based project risk prioritization
- Probability calibration for practical decision making

---

## Citation

If you use this project, please cite:

Ashraf Abdulmunim Abdulmajeed,
"Software Engineering Project Failure Prediction Model Based on Historical Project Data."

---

## Author

**Dr. Ashraf Abdulmunim Abdulmajeed**

Department of Software Engineering

University of Mosul

Mosul, Iraq
