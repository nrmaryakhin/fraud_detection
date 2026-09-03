from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import preprocessing as pp
import pandas as pd
import os



reg_lambda = [0, 0.1, 1, 5, 10]

for r in reg_lambda:
    model = LGBMClassifier(n_estimators=1100,
                            num_leaves=31,
                            learning_rate=0.02,
                            min_child_samples=20,
                            reg_lambda=r,
                            random_state=42,
                            class_weight='balanced'
                            )

    model.fit(pp.X_train, pp.y_train)
    y_pred_proba = model.predict_proba(pp.X_val)[:, 1]

    pr_auc = average_precision_score(pp.y_val, y_pred_proba)
    roc_auc = roc_auc_score(pp.y_val, y_pred_proba)

    results.append([1100, 0.02, 31, 20, r, pr_auc, roc_auc])


results_df = pd.DataFrame(
    results,
    columns=[
        'n_estimators',
        'learning_rate',
        'num_leaves',
        'min_child_samples',
        'reg_lambda',
        'pr_auc',
        'roc_auc'
    ]
)

results_df = results_df.sort_values(by='pr_auc', ascending=False)
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
results_df.to_csv(os.path.join(results_dir, 'lgbm_tuning.csv'), index=False)