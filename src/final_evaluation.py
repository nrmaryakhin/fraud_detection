import preprocessing as pp
import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import pandas as pd



model = joblib.load('../models/lgbm_model.pkl')
y_test_proba = model.predict_proba(pp.X_test)[:, 1]

threshold = 0.9281007518
y_test_threshold = (y_test_proba >= threshold).astype(int)
cm = confusion_matrix(pp.y_test, y_test_threshold)

precision = precision_score(pp.y_test, y_test_threshold)
recall = recall_score(pp.y_test, y_test_threshold)
f1 = f1_score(pp.y_test, y_test_threshold)
roc_auc = roc_auc_score(pp.y_test, y_test_proba)
pr_auc = average_precision_score(pp.y_test, y_test_proba)
cm = confusion_matrix(pp.y_test, y_test_threshold)


results = pd.DataFrame([{
    "model": "LightGBM",
    "threshold": threshold,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "tn": cm[0, 0],
    "fp": cm[0, 1],
    "fn": cm[1, 0],
    "tp": cm[1, 1]
}])

results.to_csv("../results/final_metrics.csv", index=False)