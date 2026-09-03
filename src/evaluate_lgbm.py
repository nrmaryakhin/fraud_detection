from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
import joblib
import preprocessing as pp
import numpy as np
import pandas as pd
import os




model = joblib.load("../models/lgbm_model.pkl")
y_pred_proba = model.predict_proba(pp.X_val)[:, 1]
print(f"PR-AUC: {average_precision_score(pp.y_val, y_pred_proba):.4f}")
print(f"ROC-AUC: {roc_auc_score(pp.y_val, y_pred_proba):.4f}")

precision, recall, thresholds = precision_recall_curve(
    pp.y_val,
    y_pred_proba
)

target_recall = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

results = []

for rec in target_recall:
    valid = recall[:-1] >= rec
    indices = np.where(valid)[0]
    best_idx = indices[precision[:-1][valid].argmax()]
    best_threshold = thresholds[best_idx]

    y_pred_threshold = (y_pred_proba >= best_threshold).astype(int)
    results.append({
        'target_recall': rec,
        'threshold': best_threshold,
        'precision': precision_score(pp.y_val, y_pred_threshold),
        'recall': recall_score(pp.y_val, y_pred_threshold),
        'f1': f1_score(pp.y_val, y_pred_threshold)
    })


results_df = pd.DataFrame(results)
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(os.path.join(results_dir, 'threshold_results.csv'), index=False)

model_results = pd.DataFrame([{
    'model': 'LightGBM balanced',
    'pr_auc': average_precision_score(pp.y_val, y_pred_proba),
    'roc_auc': roc_auc_score(pp.y_val, y_pred_proba)
}])

model_results.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)