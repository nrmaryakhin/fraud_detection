from fastapi.testclient import TestClient
from src.api import app




client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    transaction = {
        "Time": 50000,
        "V1": -1.5,
        "V2": 2.0,
        "V3": -1.0,
        "V4": 1.2,
        "V5": -0.5,
        "V6": 0.3,
        "V7": -0.8,
        "V8": 0.2,
        "V9": -0.4,
        "V10": -0.7,
        "V11": 1.0,
        "V12": -0.5,
        "V13": 0.2,
        "V14": -1.0,
        "V15": 0.5,
        "V16": -0.3,
        "V17": 0.4,
        "V18": -0.2,
        "V19": 0.1,
        "V20": 0.05,
        "V21": 0.1,
        "V22": -0.2,
        "V23": 0.05,
        "V24": -0.1,
        "V25": 0.1,
        "V26": -0.05,
        "V27": 0.02,
        "V28": 0.01,
        "Amount": 100,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "is_fraud" in result

    assert 0 <= result["fraud_probability"] <= 1
    assert isinstance(result["is_fraud"], bool)