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



model = joblib.load('../models/lgbm_model.pkl')
y_test_proba = model.predict_proba(pp.X_test)[:, 1]

threshold = 0.9281007518
y_test_threshold = (y_test_proba >= threshold).astype(int)
cm = confusion_matrix(pp.y_test, y_test_threshold)

print(f"Precision: {precision_score(pp.y_test, y_test_threshold):.4f}")
print(f"Recall: {recall_score(pp.y_test, y_test_threshold):.4f}")
print(f"F1: {f1_score(pp.y_test, y_test_threshold):.4f}")
print(f"ROC-AUC: {roc_auc_score(pp.y_test, y_test_proba):.4f}")
print(f"PR-AUC: {average_precision_score(pp.y_test, y_test_proba):.4f}")

cm = confusion_matrix(pp.y_test, y_test_threshold)
print("Confusion matrix:")
print(cm)