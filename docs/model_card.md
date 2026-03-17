# Model Card

## Model Details
- **Type:** XGBoost Classifier
- **Framework:** scikit-learn Pipeline + XGBoost

## Intended Use
Predict loan default probability for credit risk assessment. Assists — does not replace — human underwriting.

## Performance (5-fold CV)
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.83 |
| Recall (default) | 0.76 |
| Precision | 0.68 |
| F1 | 0.72 |

## Limitations
- Trained on historical data
- Not audited for demographic fairness
- Should not be sole decision factor
