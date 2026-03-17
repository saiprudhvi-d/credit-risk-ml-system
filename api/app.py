import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

app = FastAPI(title="Credit Risk ML System", version="1.0.0")
MODEL_PATH = "artifacts/model.pkl"
_cache = None

def load_model():
    global _cache
    if _cache is None and Path(MODEL_PATH).exists():
        with open(MODEL_PATH, 'rb') as f: _cache = pickle.load(f)
    return _cache

def risk_tier(p):
    return "HIGH" if p >= 0.70 else "MEDIUM" if p >= 0.40 else "LOW"

RECS = {"HIGH":"Manual review required","MEDIUM":"Standard underwriting review","LOW":"Approved for standard terms"}

class PredictRequest(BaseModel):
    credit_limit: float = Field(..., example=50000)
    age: int = Field(..., example=35)
    payment_delay_months: int = Field(0, example=2)
    debt_to_income: float = Field(..., example=0.42)
    credit_utilization: float = Field(0.5, example=0.65)
    num_late_payments: Optional[int] = 0

class PredictResponse(BaseModel):
    default_probability: float
    risk_tier: str
    recommendation: str
    model_version: str

@app.get("/health")
def health(): return {"status":"ok","model_loaded": load_model() is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    artifact = load_model()
    if not artifact: raise HTTPException(503, "Run training first: python src/models/train.py")
    df = pd.DataFrame([req.model_dump()])
    proba = float(artifact["model"].predict_proba(df)[0][1])
    tier = risk_tier(proba)
    return PredictResponse(default_probability=round(proba,4), risk_tier=tier, recommendation=RECS[tier], model_version=artifact.get("model_name","unknown"))

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
