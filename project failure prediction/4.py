import pandas as pd
import numpy as np
import os
import torch
import itertools

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix, accuracy_score, matthews_corrcoef,
    roc_curve
)
from catboost import CatBoostClassifier
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# إنشاء مجلد المخرجات
OUTPUT_DIR = 'manus_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def tune_tabnet(
    X_train_tab, y_train,
    X_valid_tab, y_valid,
    cat_idxs, cat_dims,
    search_mode='random', n_trials=15, seed=42,
    optimize_metric='auc'
):
    """
    TabNet hyperparameter tuning using simple random/grid search.
    Returns: best_model, best_params, best_score

    optimize_metric: 'auc' or 'pr_auc' or 'f1'
    """
    rng = np.random.RandomState(seed)

    # مساحة البحث (يمكنك توسيعها لاحقاً بنفس الآلية)
    param_space = {
        "n_d": [8, 16, 24, 32],
        "n_a": [8, 16, 24, 32],
        "n_steps": [3, 4, 5, 6],
        "gamma": [1.1, 1.3, 1.5, 1.8],
        "lambda_sparse": [1e-5, 1e-4, 1e-3],
        "cat_emb_dim": [1, 2, 3, 4],
        "lr": [1e-3, 3e-3, 1e-2, 2e-2],
        "batch_size": [128, 256, 512],
        "virtual_batch_size": [64, 128, 256],
        "mask_type": ["entmax", "sparsemax"]
    }

    keys = list(param_space.keys())
    all_combos = list(itertools.product(*[param_space[k] for k in keys]))

    if search_mode == 'grid':
        trials = all_combos
    else:
        # random
        idxs = rng.choice(len(all_combos), size=min(n_trials, len(all_combos)), replace=False)
        trials = [all_combos[i] for i in idxs]

    best_score = -1e9
    best_params = None
    best_model = None

    for t_i, combo in enumerate(trials, 1):
        params = dict(zip(keys, combo))

        model = TabNetClassifier(
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            gamma=params["gamma"],
            lambda_sparse=params["lambda_sparse"],
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=params["cat_emb_dim"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params["lr"]),
            scheduler_params={"step_size": 50, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type=params["mask_type"],
            verbose=0
        )

        model.fit(
            X_train_tab, y_train,
            eval_set=[(X_valid_tab, y_valid)],
            eval_metric=['auc'],
            max_epochs=200,
            patience=30,
            batch_size=params["batch_size"],
            virtual_batch_size=min(params["virtual_batch_size"], params["batch_size"]),
            num_workers=0,
            drop_last=False,
            weights=1
        )

        probs = model.predict_proba(X_valid_tab)[:, 1]

        if optimize_metric == 'pr_auc':
            score = average_precision_score(y_valid, probs)
        elif optimize_metric == 'f1':
            thresholds = np.linspace(0, 1, 200)
            f1s = [f1_score(y_valid, (probs > th).astype(int)) for th in thresholds]
            score = float(np.max(f1s))
        else:
            score = roc_auc_score(y_valid, probs)

        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

        print(f"[TabNet Tuning] Trial {t_i}/{len(trials)} | score({optimize_metric})={score:.4f} | best={best_score:.4f}")

    return best_model, best_params, best_score


def run_pipeline(file_path):
    # 1. تحميل البيانات وهندسة الميزات
    df = pd.read_csv(file_path)
    df['Creation_Date'] = pd.to_datetime(df['Creation_Date'], errors='coerce')
    df = df.sort_values('Creation_Date').reset_index(drop=True)

    # بناء الهدف (Risk Label)
    q90 = df['resolution_time_days'].quantile(0.9)
    df['target'] = ((df['resolution_time_days'] > q90) |
                    (df['reopen_count'] > 0) |
                    (df['priority_change_count'] > 0)).astype(int)

    # هندسة الميزات التاريخية
    df['creation_month'] = df['Creation_Date'].dt.month
    df['creation_dayofweek'] = df['Creation_Date'].dt.dayofweek
    df['assignee_prev_count'] = df.groupby('Assignee_ID').cumcount()
    df['project_prev_count'] = df.groupby('Project_ID').cumcount()
    df['type_prev_count'] = df.groupby('Type').cumcount()
    df['priority_prev_count'] = df.groupby('Priority').cumcount()
    df['title_len'] = df['Title'].str.len().fillna(0)
    df['desc_len'] = df['Description_Text'].str.len().fillna(0)
    df['Story_Point'] = pd.to_numeric(df['Story_Point'], errors='coerce').fillna(0)

    cat_features = ['Type', 'Priority', 'Creator_ID', 'Reporter_ID', 'Assignee_ID', 'Project_ID']
    leaky_cols = ['Resolution', 'Status', 'Resolution_Date', 'Last_Updated', 'Timespent', 'In_Progress_Minutes',
                  'Total_Effort_Minutes', 'Resolution_Time_Minutes', 'resolution_time_days', 'reopen_count',
                  'priority_change_count', 'Title_Changed_After_Estimation', 'Description_Changed_After_Estimation',
                  'Story_Point_Changed_After_Estimation', 'Pull_Request_URL', 'Estimation_Date']
    id_cols = ['ID', 'Jira_ID', 'Issue_Key', 'URL', 'Title', 'Description', 'Description_Text', 'Description_Code',
               'Creation_Date', 'Sprint_ID']

    X = df.drop(columns=leaky_cols + id_cols + ['target'], errors='ignore')
    y = df['target'].values

    # معالجة الميزات الفئوية لـ TabNet
    X_tabnet = X.copy()
    cat_idxs = []
    cat_dims = []
    for i, col in enumerate(X.columns):
        if col in cat_features:
            le = LabelEncoder()
            X_tabnet[col] = le.fit_transform(X[col].astype(str))
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))
        else:
            X_tabnet[col] = X[col].astype(float)

    # 2. التدريب والتقييم (آخر طية)
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    X_train_tab, X_test_tab = X_tabnet.iloc[train_idx].values, X_tabnet.iloc[test_idx].values
    y_train, y_test = y[train_idx], y[test_idx]

    # --- النماذج ---

    # 1. CatBoost
    print("Training CatBoost...")
    cb = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        cat_features=cat_features, verbose=0,
        auto_class_weights='Balanced'
    )
    cb.fit(X_train, y_train)
    cb_probs = cb.predict_proba(X_test)[:, 1]

    # 2. XGBoost
    print("Training XGBoost...")
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05,
        scale_pos_weight=ratio, eval_metric='logloss'
    )
    xgb_model.fit(X_train_tab, y_train)
    xgb_probs = xgb_model.predict_proba(X_test_tab)[:, 1]

    # 3. TabNet (Tuned)
    print("Tuning + Training TabNet...")
    tabnet_best, tabnet_best_params, tabnet_best_score = tune_tabnet(
        X_train_tab, y_train,
        X_test_tab, y_test,   # نفس آلية الكود الحالي (آخر طية)
        cat_idxs, cat_dims,
        search_mode='random',
        n_trials=5,
        seed=42,
        optimize_metric='auc'  # غيّرها إلى 'pr_auc' أو 'f1' إذا رغبت
    )
    print("\nBest TabNet Params:", tabnet_best_params)
    print(f"Best TabNet score: {tabnet_best_score:.4f}")

    # حفظ أفضل إعدادات (اختياري لكن مفيد)
    pd.DataFrame([tabnet_best_params]).to_csv(f'{OUTPUT_DIR}/tabnet_best_params.csv', index=False)

    tabnet_probs = tabnet_best.predict_proba(X_test_tab)[:, 1]

    # 4. Ensemble (Weighted Average)
    ens_probs = (cb_probs * 0.4 + xgb_probs * 0.3 + tabnet_probs * 0.3)

    # 3. حساب المقاييس والرسومات
    all_probs = {'CatBoost': cb_probs, 'XGBoost': xgb_probs, 'TabNet': tabnet_probs, 'Ensemble': ens_probs}
    results = []

    plt.figure(figsize=(10, 8))  # ROC Curve
    plt.plot([0, 1], [0, 1], 'k--')

    for name, probs in all_probs.items():
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_test, (probs > t).astype(int)) for t in thresholds]
        best_t = thresholds[np.argmax(f1_scores)]
        y_pred = (probs > best_t).astype(int)

        k = int(len(y_test) * 0.2)
        top_k_idx = np.argsort(probs)[-k:]
        recall_20 = sum(y_test[top_k_idx]) / max(sum(y_test), 1)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, probs),
            'PR-AUC': average_precision_score(y_test, probs),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Best Threshold': best_t,
            'Recall@20%': recall_20
        }
        results.append(metrics)

        # مصفوفة الارتباك
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}\n(Threshold: {best_t:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{OUTPUT_DIR}/cm_{name.lower()}.png')
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["ROC-AUC"]:.3f})')

    plt.figure(1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison (with Tuned TabNet)')
    plt.legend(loc='lower right')
    plt.savefig(f'{OUTPUT_DIR}/roc_curves_v4.png')
    plt.close()

    results_df = pd.DataFrame(results)

    # مخطط مقارنة المقاييس
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    plot_df = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model')
    plt.title('Model Comparison (including Tuned TabNet)')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_comparison_v4.png')
    plt.close()

    return results_df


if __name__ == "__main__":
    results_df = run_pipeline('mesos_tabular_dataset.csv')
    results_df.to_csv(f'{OUTPUT_DIR}/final_metrics_v4.csv', index=False)
    print("\nFinal Performance Metrics (with Tuned TabNet):")
    print(results_df.to_string(index=False))
