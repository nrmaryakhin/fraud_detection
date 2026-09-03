from lightgbm import LGBMClassifier
import preprocessing as pp
import joblib




model = LGBMClassifier(
    n_estimators=1100,
    num_leaves=31,
    learning_rate=0.02,
    min_child_samples=20,
    reg_lambda=0,
    random_state=42,
    class_weight="balanced"
)

model.fit(pp.X_train, pp.y_train)
joblib.dump(model, '../models/lgbm_model.pkl')