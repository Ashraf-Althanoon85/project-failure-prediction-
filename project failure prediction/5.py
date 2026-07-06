import pandas as pd
import numpy as np
import os
import torch

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score,
    precision_score, confusion_matrix, accuracy_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
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

# ألوان ثابتة لكل خوارزمية (Color-blind friendly)
MODEL_COLORS = {
    "CatBoost": "#1f77b4",   # أزرق
    "XGBoost": "#ff7f0e",    # برتقالي
    "TabNet": "#2ca02c",     # أخضر
    "Ensemble": "#d62728"    # أحمر
}


def _annotate_barplot(ax, fmt="{:.3f}"):
    """ضع قيمة كل عمود فوقه."""
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='bottom',
            xytext=(0, 3),
            textcoords='offset points',
            fontsize=9
        )


def _plot_feature_importance(imp_df, title, out_path, top_n=20):
    """يرسم Top-N Feature Importance مع أرقام."""
    imp_df = imp_df.sort_values("Importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, max(6, int(top_n * 0.35))))
    ax = sns.barplot(data=imp_df, x="Importance", y="Feature")
    plt.title(title)
    plt.tight_layout()
    # وضع القيم (أفقي)
    for i, v in enumerate(imp_df["Importance"].values):
        ax.text(v, i, f" {v:.4f}", va='center', fontsize=9)
    plt.savefig(out_path, dpi=300)
    plt.close()


def _normalize_importance(values):
    values = np.array(values, dtype=float)
    s = values.sum()
    if s <= 0:
        return np.zeros_like(values)
    return values / s


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
    feature_names = list(X.columns)
    y = df['target'].values

    # معالجة الميزات الفئوية لـ TabNet / XGBoost
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

    # 3. TabNet (Fixed Best Params)
    print("Training TabNet (Fixed Best Params)...")
    tabnet = TabNetClassifier(
        n_d=32,
        n_a=8,
        n_steps=4,
        gamma=1.1,
        lambda_sparse=1e-5,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=4,
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
        max_epochs=200,
        patience=30,
        batch_size=128,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=1
    )
    tabnet_probs = tabnet.predict_proba(X_test_tab)[:, 1]

    # 4. Ensemble (Weighted Average)
    ens_probs = (cb_probs * 0.4 + xgb_probs * 0.3 + tabnet_probs * 0.3)

    # 5. حساب المقاييس
    all_probs = {'CatBoost': cb_probs, 'XGBoost': xgb_probs, 'TabNet': tabnet_probs, 'Ensemble': ens_probs}
    results = []

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
        plt.title(f'Confusion Matrix: {name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/cm_{name.lower()}.png', dpi=300)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{OUTPUT_DIR}/final_metrics_v4.csv', index=False)

    # ====== ROC CURVE (محسّن بصريًا) ======
    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1],
             linestyle='--',
             color='gray',
             linewidth=1,
             alpha=0.7,
             label='Random')

    for name, probs in all_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        plt.plot(
            fpr, tpr,
            label=f'{name} (AUC = {auc_val:.3f})',
            color=MODEL_COLORS[name],
            linewidth=2.5
        )

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/roc_curve_beautified.png', dpi=300)
    plt.close()

    # ====== PR CURVE (محسّن بصريًا) ======
    plt.figure(figsize=(9, 7))

    for name, probs in all_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        plt.plot(
            recall, precision,
            label=f'{name} (PR-AUC = {pr_auc:.3f})',
            color=MODEL_COLORS[name],
            linewidth=2.5
        )

    baseline = y_test.mean()
    plt.hlines(
        baseline, xmin=0, xmax=1,
        colors='gray',
        linestyles='--',
        linewidth=1,
        alpha=0.7,
        label=f'Baseline (Pos Rate = {baseline:.2f})'
    )

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision–Recall Curve Comparison', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(loc='lower left', fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pr_curve_beautified.png', dpi=300)
    plt.close()

    # ====== مخطط مقارنة مجمّع (مع أرقام على الأعمدة) ======
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    plot_df = results_df.melt(id_vars='Model', value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x='Metric', y='Score', hue='Model')
    plt.title('Model Comparison (including TabNet Fixed Params)')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    _annotate_barplot(ax, fmt="{:.3f}")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/performance_comparison_v4.png', dpi=300)
    plt.close()

    # ====== مقارنة لكل مقياس على حدة (مع قيم على الأعمدة) ======
    compare_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'MCC', 'Recall@20%']

    for m in compare_metrics:
        plt.figure(figsize=(8, 5))
        axm = sns.barplot(data=results_df, x='Model', y=m, palette=[MODEL_COLORS[x] for x in results_df['Model']])
        plt.title(f'Comparison by Metric: {m}')
        if m == 'MCC':
            plt.ylim(-1.1, 1.1)
        else:
            plt.ylim(0, 1.1)
        _annotate_barplot(axm, fmt="{:.4f}")
        plt.tight_layout()
        fname = m.lower().replace('@', '_at_').replace('-', '_').replace('%', 'pct')
        plt.savefig(f'{OUTPUT_DIR}/metric_comparison_{fname}.png', dpi=300)
        plt.close()

    # ====== مخططات المقاييس التي يتفوّق فيها TabNet (مع القيم) ======
    tabnet_row = results_df[results_df['Model'] == 'TabNet'].iloc[0]
    tabnet_top_metrics = []
    for m in compare_metrics:
        max_val = results_df[m].max()
        if np.isclose(tabnet_row[m], max_val):
            tabnet_top_metrics.append(m)

            plt.figure(figsize=(8, 5))
            axm = sns.barplot(data=results_df, x='Model', y=m, palette=[MODEL_COLORS[x] for x in results_df['Model']])
            plt.title(f'TabNet is Best on: {m}')
            if m == 'MCC':
                plt.ylim(-1.1, 1.1)
            else:
                plt.ylim(0, 1.1)
            _annotate_barplot(axm, fmt="{:.4f}")
            plt.tight_layout()
            fname = m.lower().replace('@', '_at_').replace('-', '_').replace('%', 'pct')
            plt.savefig(f'{OUTPUT_DIR}/tabnet_best_{fname}.png', dpi=300)
            plt.close()

    # ====== Feature Importance لكل نموذج + تفسير النتائج ======

    # CatBoost importance
    try:
        cb_imp = cb.get_feature_importance()
        cb_imp = _normalize_importance(cb_imp)
        cb_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": cb_imp})
        _plot_feature_importance(
            cb_imp_df, "Feature Importance - CatBoost",
            f"{OUTPUT_DIR}/fi_catboost.png", top_n=20
        )
    except Exception as e:
        cb_imp_df = None
        print("CatBoost feature importance failed:", e)

    # XGBoost importance
    try:
        xgb_imp = xgb_model.feature_importances_
        xgb_imp = _normalize_importance(xgb_imp)
        xgb_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": xgb_imp})
        _plot_feature_importance(
            xgb_imp_df, "Feature Importance - XGBoost",
            f"{OUTPUT_DIR}/fi_xgboost.png", top_n=20
        )
    except Exception as e:
        xgb_imp_df = None
        print("XGBoost feature importance failed:", e)

    # TabNet importance
    try:
        tab_imp = tabnet.feature_importances_
        tab_imp = _normalize_importance(tab_imp)
        tab_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": tab_imp})
        _plot_feature_importance(
            tab_imp_df, "Feature Importance - TabNet",
            f"{OUTPUT_DIR}/fi_tabnet.png", top_n=20
        )
    except Exception as e:
        tab_imp_df = None
        print("TabNet feature importance failed:", e)

    # Ensemble importance = weighted normalized importances
    ens_imp_df = None
    try:
        if (cb_imp_df is not None) and (xgb_imp_df is not None) and (tab_imp_df is not None):
            cb_vec = cb_imp_df.set_index("Feature").loc[feature_names]["Importance"].values
            xgb_vec = xgb_imp_df.set_index("Feature").loc[feature_names]["Importance"].values
            tab_vec = tab_imp_df.set_index("Feature").loc[feature_names]["Importance"].values

            ens_vec = 0.4 * cb_vec + 0.3 * xgb_vec + 0.3 * tab_vec
            ens_vec = _normalize_importance(ens_vec)
            ens_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": ens_vec})

            _plot_feature_importance(
                ens_imp_df, "Feature Importance - Ensemble (0.4 CB + 0.3 XGB + 0.3 TabNet)",
                f"{OUTPUT_DIR}/fi_ensemble.png", top_n=20
            )
    except Exception as e:
        print("Ensemble feature importance failed:", e)

    # ====== تقرير تفسيري نصي ======
    report_path = f'{OUTPUT_DIR}/interpretability_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Model Performance Summary ===\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n=== Metrics where TabNet is best (including Ensemble in comparison) ===\n")
        if len(tabnet_top_metrics) == 0:
            f.write("- None\n")
        else:
            for m in tabnet_top_metrics:
                f.write(f"- {m}: {tabnet_row[m]:.6f}\n")

        f.write("\n\n=== Explainability via Feature Importance (Top 10) ===\n")

        def write_top(df_imp, title):
            if df_imp is None:
                f.write(f"\n{title}: Not available\n")
                return
            top = df_imp.sort_values("Importance", ascending=False).head(10)
            f.write(f"\n{title}:\n")
            for _, r in top.iterrows():
                f.write(f"- {r['Feature']}: {r['Importance']:.6f}\n")

        write_top(cb_imp_df, "CatBoost Feature Importance")
        write_top(xgb_imp_df, "XGBoost Feature Importance")
        write_top(tab_imp_df, "TabNet Feature Importance")
        write_top(ens_imp_df, "Ensemble Feature Importance")

        f.write("\n\n=== Interpretation Notes (Data-driven) ===\n")
        f.write("1) الميزات الأعلى أهمية غالباً تمثل إشارات مبكرة للمخاطر (مثل حجم المهمة، خبرة/سجل المكلّف، خصائص المشروع، وخصائص النص).\n")
        f.write("2) إذا ظهرت ميزات مثل assignee_prev_count / project_prev_count ضمن الأعلى، فهذا يعني أن التاريخ السلوكي للمكلّف/المشروع مؤثر في التنبؤ بالمخاطر.\n")
        f.write("3) إذا كانت Story_Point أو أطوال النص (title_len/desc_len) ضمن الأعلى، فهذا يشير إلى أن تعقيد المهمة ووصفها يرتبط باحتمالية التأخر/إعادة الفتح/تغيير الأولوية.\n")
        f.write("4) اختلاف Top features بين النماذج يوضح لماذا Ensemble قد يتفوّق: لأنه يجمع إشارات متعددة من مدارس مختلفة (Boosting + Deep Tabular).\n")

    print("\nSaved interpretability report:", report_path)
    return results_df


if __name__ == "__main__":
    results_df = run_pipeline('mesos_tabular_dataset.csv')
    print("\nFinal Performance Metrics (TabNet Fixed Params):")
    print(results_df.to_string(index=False))
