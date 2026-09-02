from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from train import model
import preprocessing as pp
import numpy as np
import pandas as pd




y_pred_proba = model.predict_proba(pp.X_val_scaled)[:, 1]
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
print(results_df)