from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"

model = joblib.load(MODEL_PATH)

THRESHOLD = 0.9281

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent bank transactions",
    version="1.0.0"
)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])
    probability = model.predict_proba(data)[0, 1]
    prediction = int(probability >= THRESHOLD)
    return {
        "fraud_probability": float(probability),
        "is_fraud": bool(prediction)
    }