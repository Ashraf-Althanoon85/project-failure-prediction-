import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, confusion_matrix, accuracy_score, matthews_corrcoef,
                             roc_curve)
from catboost import CatBoostClassifier
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# إنشاء مجلد المخرجات
OUTPUT_DIR = 'manus_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


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
    cb = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, cat_features=cat_features, verbose=0,
                            auto_class_weights='Balanced')
    cb.fit(X_train, y_train)
    cb_probs = cb.predict_proba(X_test)[:, 1]

    # 2. XGBoost
    print("Training XGBoost...")
    # XGBoost يحتاج لترميز رقمي أيضاً
    ratio = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, scale_pos_weight=ratio, eval_metric='logloss')
    xgb_model.fit(X_train_tab, y_train)
    xgb_probs = xgb_model.predict_proba(X_test_tab)[:, 1]

    # 3. TabNet
    print("Training TabNet...")
    tabnet = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=0
    )
    tabnet.fit(
        X_train_tab, y_train,
        eval_set=[(X_test_tab, y_test)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=20,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
        weights=1
    )
    tabnet_probs = tabnet.predict_proba(X_test_tab)[:, 1]

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
        recall_20 = sum(y_test[top_k_idx]) / sum(y_test)

        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
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
    plt.title('ROC Curves Comparison (with TabNet)')
    plt.legend(loc='lower right')
    plt.savefig(f'{OUTPUT_DIR}/roc_curves_v4.png')
    plt.close()

    results_df = pd.DataFrame(results)

    # مخطط مقارنة المقاييس
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    plot_df = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model')
    plt.title('Model Comparison (including TabNet)')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_comparison_v4.png')
    plt.close()

    return results_df


if __name__ == "__main__":
    results_df = run_pipeline('mesos_tabular_dataset.csv')
    results_df.to_csv(f'{OUTPUT_DIR}/final_metrics_v4.csv', index=False)
    print("\nFinal Performance Metrics (with TabNet):")
    print(results_df.to_string(index=False))
