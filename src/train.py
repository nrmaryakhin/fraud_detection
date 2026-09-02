from sklearn.linear_model import LogisticRegression
import preprocessing as pp




model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(pp.X_train_scaled, pp.y_train)

