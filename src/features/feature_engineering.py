import pandas as pd
import numpy as np
from typing import List

def add_debt_to_income(df):
    if 'total_debt' in df.columns and 'monthly_income' in df.columns:
        df['debt_to_income'] = (df['total_debt'] / df['monthly_income'].replace(0, np.nan)).round(4)
    return df

def add_credit_utilization(df):
    if 'balance' in df.columns and 'credit_limit' in df.columns:
        df['credit_utilization'] = (df['balance'] / df['credit_limit'].replace(0, np.nan)).clip(0,1).round(4)
    return df

def add_payment_history_score(df, payment_cols: List[str]):
    available = [c for c in payment_cols if c in df.columns]
    if available:
        df['payment_history_score'] = ((df[available] == 0).sum(axis=1) / len(available)).round(4)
        df['max_payment_delay'] = df[available].max(axis=1)
    return df

def add_age_features(df):
    if 'age' in df.columns:
        df['age_bucket'] = pd.cut(df['age'], bins=[0,25,35,50,65,120], labels=['18-25','26-35','36-50','51-65','65+'])
    return df

def engineer_features(df):
    pay_cols = [c for c in df.columns if c.startswith('pay_')]
    df = add_debt_to_income(df)
    df = add_credit_utilization(df)
    df = add_payment_history_score(df, pay_cols)
    df = add_age_features(df)
    return df
