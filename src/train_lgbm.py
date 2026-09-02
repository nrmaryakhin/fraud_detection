from lightgbm import LGBMClassifier
import preprocessing as pp




model = LGBMClassifier(random_state=42, class_weight='balanced')
model.fit(pp.X_train, pp.y_train)