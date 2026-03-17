import pickle, yaml
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
try:
    from xgboost import XGBClassifier
    XGB = True
except ImportError:
    XGB = False

def get_models():
    m = {
        'logistic_regression': Pipeline([('imp',SimpleImputer(strategy='median')),('scl',StandardScaler()),('clf',LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42))]),
        'random_forest': Pipeline([('imp',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42,n_jobs=-1))]),
    }
    if XGB:
        m['xgboost'] = Pipeline([('imp',SimpleImputer(strategy='median')),('clf',XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,scale_pos_weight=3,random_state=42,verbosity=0))])
    return m

def evaluate_models(X, y, models, cv_folds=5):
    results = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        cv = cross_validate(model, X, y, cv=skf, scoring=['roc_auc','f1','recall'], n_jobs=-1)
        results[name] = {k: cv[f'test_{k}'].mean() for k in ['roc_auc','f1','recall']}
        print(f"    ROC-AUC: {results[name]['roc_auc']:.3f} Recall: {results[name]['recall']:.3f}")
    return results

def train_and_save(X, y, name, models, out='artifacts/model.pkl'):
    model = models[name]
    model.fit(X, y)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump({'model': model, 'model_name': name, 'features': list(X.columns)}, f)
    print(f"  Saved: {out}")
    return model

def run_training(config_path='configs/train_config.yaml'):
    with open(config_path) as f: config = yaml.safe_load(f)
    df = pd.read_parquet(config['data_path'])
    features = [c for c in config.get('features',[]) if c in df.columns]
    X, y = df[features], df[config['target']]
    print(f"Dataset: {len(X):,} rows | {y.mean():.1%} default rate")
    models = get_models()
    results = evaluate_models(X, y, models)
    best = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"Best: {best} ROC-AUC={results[best]['roc_auc']:.3f}")
    train_and_save(X, y, best, models, config['output']['model_path'])
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_config.yaml')
    run_training(p.parse_args().config)
