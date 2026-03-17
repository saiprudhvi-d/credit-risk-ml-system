import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import add_debt_to_income, add_credit_utilization, add_payment_history_score, engineer_features
from api.app import risk_tier

class TestFeatureEngineering:
    def test_dti(self):
        df = pd.DataFrame({"total_debt":[2000.],"monthly_income":[5000.]})
        df = add_debt_to_income(df)
        assert df["debt_to_income"].iloc[0] == pytest.approx(0.4, abs=0.001)
    def test_dti_zero_income(self):
        df = pd.DataFrame({"total_debt":[2000.],"monthly_income":[0.]})
        assert pd.isna(add_debt_to_income(df)["debt_to_income"].iloc[0])
    def test_utilization(self):
        df = pd.DataFrame({"balance":[3000.],"credit_limit":[10000.]})
        assert add_credit_utilization(df)["credit_utilization"].iloc[0] == pytest.approx(0.3, abs=0.001)
    def test_utilization_clipped(self):
        df = pd.DataFrame({"balance":[15000.],"credit_limit":[10000.]})
        assert add_credit_utilization(df)["credit_utilization"].iloc[0] == 1.0
    def test_payment_score(self):
        df = pd.DataFrame({"pay_1":[0,0,2],"pay_2":[0,1,0]})
        df = add_payment_history_score(df, ["pay_1","pay_2"])
        assert df["payment_history_score"].iloc[0] == 1.0
        assert df["payment_history_score"].iloc[1] == 0.5
    def test_engineer_no_crash(self):
        df = pd.DataFrame({"total_debt":[1000.],"monthly_income":[4000.],"balance":[2000.],"credit_limit":[8000.],"age":[30]})
        assert len(engineer_features(df)) == 1

class TestRiskTier:
    def test_high(self): assert risk_tier(0.75) == "HIGH"
    def test_medium(self): assert risk_tier(0.50) == "MEDIUM"
    def test_low(self): assert risk_tier(0.20) == "LOW"
    def test_boundary_high(self): assert risk_tier(0.70) == "HIGH"
    def test_boundary_medium(self): assert risk_tier(0.40) == "MEDIUM"
