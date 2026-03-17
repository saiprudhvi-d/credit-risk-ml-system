# Credit Risk ML System

![Tests](https://github.com/saiprudhvi-d/credit-risk-ml-system/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)

End-to-end ML system for credit risk prediction with FastAPI inference endpoint and model card.

## Stack
Python · scikit-learn · XGBoost · FastAPI · MLflow

## Results
| Model | ROC-AUC | Recall |
|-------|---------|--------|
| Logistic Regression | 0.71 | 0.64 |
| Random Forest | 0.79 | 0.71 |
| **XGBoost** | **0.83** | **0.76** |

## Setup
```bash
git clone https://github.com/saiprudhvi-d/credit-risk-ml-system
pip install -r requirements.txt
python src/models/train.py
uvicorn api.app:app --reload
```

## CI/CD
Tests run automatically on every push via GitHub Actions.