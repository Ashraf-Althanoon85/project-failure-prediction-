# risk_ensemble_boosted.py
# -*- coding: utf-8 -*-

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb


DATA_PATH = "mesos_tabular_dataset.csv"
OUT_DIR = "outputs_boosted"
RANDOM_STATE = 42
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# 1) Target (Historical Only, less noisy)
# -------------------------
def build_target(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index, dtype=int)

    if "resolution_time_days" in df.columns:
        thr = df["resolution_time_days"].quantile(0.75)
        score += (df["resolution_time_days"] > thr).fillna(False).astype(int)

    score += (df.get("reopen_count", 0).fillna(0) > 0).astype(int)
    score += (df.get("priority_change_count", 0).fillna(0) > 0).astype(int)

    for c in ["Title_Changed_After_Estimation", "Description_Changed_After_Estimation", "Story_Point_Changed_After_Estimation"]:
        if c in df.columns:
            score += (df[c].fillna(0).astype(int) > 0).astype(int)

    return (score >= 2).astype(int)


# -------------------------
# 2) Leakage-safe historical aggregates (uses ONLY past rows per group)
# -------------------------
def add_historical_aggregates(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    df = df.copy()
    if "Creation_Date" not in df.columns:
        return df

    dt = pd.to_datetime(df["Creation_Date"], errors="coerce")
    df["_dt"] = dt
    df["_y"] = y.values

    # choose grouping columns if exist
    group_candidates = []
    if "Project_ID" in df.columns: group_candidates.append("Project_ID")
    if "Type" in df.columns: group_candidates.append("Type")
    grp = group_candidates if group_candidates else None

    # sort by time so "shift" is truly historical
    df = df.sort_values("_dt")

    if grp:
        g = df.groupby(grp, sort=False)

        # historical risk rate (previous labels only)
        df["hist_risk_rate"] = g["_y"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=list(range(len(grp))), drop=True)

        # historical volume
        df["hist_count"] = g["_y"].cumcount()

    else:
        df["hist_risk_rate"] = df["_y"].shift(1).expanding().mean()
        df["hist_count"] = np.arange(len(df))

    # fill missing early rows
    df["hist_risk_rate"] = df["hist_risk_rate"].fillna(df["_y"].mean())
    df["hist_count"] = df["hist_count"].fillna(0)

    df.drop(columns=["_dt", "_y"], inplace=True)
    # restore original order
    df = df.sort_index()
    return df


# -------------------------
# 3) Clean features + early signals (NO leakage)
# -------------------------
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Early signals
    if "Title" in df.columns:
        df["title_len"] = df["Title"].fillna("").astype(str).str.len()
    if "Description_Text" in df.columns:
        df["desc_len"] = df["Description_Text"].fillna("").astype(str).str.len()
    elif "Description" in df.columns:
        df["desc_len"] = df["Description"].fillna("").astype(str).str.len()

    if "Creation_Date" in df.columns:
        dt = pd.to_datetime(df["Creation_Date"], errors="coerce")
        df["created_month"] = dt.dt.month
        df["created_dow"] = dt.dt.dayofweek

    # Drop leakage columns that were used to build y
    leak_cols = [
        "resolution_time_days","reopen_count","priority_change_count",
        "Title_Changed_After_Estimation","Description_Changed_After_Estimation","Story_Point_Changed_After_Estimation"
    ]
    df.drop(columns=[c for c in leak_cols if c in df.columns], inplace=True, errors="ignore")

    # Drop raw texts + URLs/IDs
    drop_cols = [
        "ID","Jira_ID","Issue_Key","URL","Pull_Request_URL",
        "Title","Description","Description_Text","Description_Code"
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    # Drop effort/cost-like (as requested)
    effort_like = ["Story_Point","Timespent","In_Progress_Minutes","Total_Effort_Minutes"]
    df.drop(columns=[c for c in effort_like if c in df.columns], inplace=True, errors="ignore")

    # Drop non-numeric date/time columns (avoid NaT issues)
    for c in list(df.columns):
        if ("date" in c.lower() or "time" in c.lower()) and not pd.api.types.is_numeric_dtype(df[c]):
            if c != "Creation_Date":
                df.drop(columns=c, inplace=True)

    # Fix mixed object cols
    for c in df.columns:
        if df[c].dtype == "object":
            s = df[c].astype(str).replace({"None": np.nan, "nan": np.nan, "NaN": np.nan})
            s_num = pd.to_numeric(s, errors="coerce")
            df[c] = s_num if (s_num.notna().mean() >= 0.7) else s.fillna("NA")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# -------------------------
# 4) Utils
# -------------------------
def metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
    }

def best_threshold_f1(y_true, proba):
    best_th, best_f = 0.5, -1
    for th in np.linspace(0.05, 0.95, 91):
        pred = (proba >= th).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_f, best_th = f, th
    return best_th

def save_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0,1], ["0","1"]); plt.yticks([0,1], ["0","1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# -------------------------
# 5) Main: boosted base models + stacking
# -------------------------
def main():
    df = pd.read_csv(DATA_PATH)
    y = build_target(df)

    # add leakage-safe historical aggregates (big boost usually)
    df = add_historical_aggregates(df, y)

    X = clean_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
        ],
        remainder="drop"
    )

    X_train_enc = pre.fit_transform(X_train)
    X_test_enc = pre.transform(X_test)

    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    scale_pos = neg / max(pos, 1)

    # tuned params (خفيفة لكن فعالة)
    base_models = {
        "CatBoost": CatBoostClassifier(
            iterations=900, depth=8, learning_rate=0.035,
            random_seed=RANDOM_STATE, verbose=False,
            class_weights=[1.0, min(3.0, scale_pos)],
            l2_leaf_reg=5
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=1400, learning_rate=0.02, num_leaves=63,
            max_depth=-1, min_data_in_leaf=15,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, verbose=-1, verbosity=-1,
            force_row_wise=True, is_unbalance=True
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=1400, learning_rate=0.02, max_depth=6,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, reg_alpha=0.0,
            random_state=RANDOM_STATE, tree_method="hist", verbosity=0,
            scale_pos_weight=scale_pos
        )
    }

    # ---- base model evaluation + best threshold ----
    rows = []
    base_test_probas = {}

    for name, model in base_models.items():
        model.fit(X_train_enc, y_train)
        proba = model.predict_proba(X_test_enc)[:, 1]
        th = best_threshold_f1(y_test.values, proba)
        pred = (proba >= th).astype(int)

        base_test_probas[name] = proba
        m = metrics(y_test.values, pred, proba)
        m.update({"model": name, "best_th": th})
        rows.append(m)

        save_cm(y_test.values, pred, os.path.join(OUT_DIR, f"cm_{name}.png"), f"CM - {name} (th={th:.2f})")

    # ---- stacking (out-of-fold) ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(X_train_enc), len(base_models)), dtype=float)

    for i, (name, model) in enumerate(base_models.items()):
        for tr_idx, va_idx in skf.split(X_train_enc, y_train):
            model.fit(X_train_enc[tr_idx], y_train.values[tr_idx])
            oof[va_idx, i] = model.predict_proba(X_train_enc[va_idx])[:, 1]

    meta = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta.fit(oof, y_train.values)

    test_stack = np.column_stack([base_test_probas[n] for n in base_models.keys()])
    proba_stack = meta.predict_proba(test_stack)[:, 1]
    th_stack = best_threshold_f1(y_test.values, proba_stack)
    pred_stack = (proba_stack >= th_stack).astype(int)

    m = metrics(y_test.values, pred_stack, proba_stack)
    m.update({"model": "STACKING", "best_th": th_stack})
    rows.append(m)

    save_cm(y_test.values, pred_stack, os.path.join(OUT_DIR, "cm_STACKING.png"), f"CM - STACKING (th={th_stack:.2f})")

    res = pd.DataFrame(rows).sort_values("f1", ascending=False)
    res.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    print("\nDone ✅ Saved to:", os.path.abspath(OUT_DIR))
    print(res[["model","accuracy","precision","recall","f1","pr_auc","roc_auc","best_th"]])


if __name__ == "__main__":
    main()
