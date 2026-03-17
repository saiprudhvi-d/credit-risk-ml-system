# Credit Risk ML System

![Tests](https://github.com/saiprudhvi-d/credit-risk-ml-system/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)

## Overview
An end-to-end machine learning system for credit risk assessment — predicting loan default probability from preprocessing through model training, evaluation, and a FastAPI inference endpoint.

## Business Problem
For lending teams, missing a high-risk applicant (false negative) is far costlier than a false positive. This system optimizes for **recall on the default class** while maintaining acceptable precision.

## ML Workflow
```
Raw Loan Data → [Preprocessing] → [Feature Engineering]
→ [Model Training: LR + RF + XGBoost] → [Evaluation + Threshold Tuning]
→ [Model Artifact] → [FastAPI Inference: POST /predict]
```

## Model Results
| Model | ROC-AUC | Recall | F1 |
|-------|---------|--------|----|
| Logistic Regression | 0.71 | 0.64 | 0.61 |
| Random Forest | 0.79 | 0.71 | 0.68 |
| **XGBoost** | **0.83** | **0.76** | **0.72** |

## Tech Stack
Python · scikit-learn · XGBoost · FastAPI · pandas · pytest · GitHub Actions

## Setup
```bash
git clone https://github.com/saiprudhvi-d/credit-risk-ml-system
pip install -r requirements.txt
python src/models/train.py
uvicorn api.app:app --reload
```

## Future Improvements
- SHAP explainability for model decisions
- Fairness audit across demographic groups
- Online learning for model drift detection