from train import model
import preprocessing as pp
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
import pandas as pd
import numpy as np




y_pred_proba = model.predict_proba(pp.X_val_scaled)[:, 1]
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