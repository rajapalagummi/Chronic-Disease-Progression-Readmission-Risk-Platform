import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.causal_survival import run_did, run_placebo
from data.quality.dq_pipeline import run_dq_checks
import duckdb


class TestCausalAnalysis:
    def setup_method(self):
        np.random.seed(42)
        n = 2000
        self.df = pd.DataFrame({
            "exercise_any": np.random.choice([1, 2], n),
            "exercise_treatment": np.random.binomial(1, 0.5, n),
            "period": np.random.choice(["pre", "post"], n),
            "diabetes_status": np.random.binomial(1, 0.12, n),
            "heart_disease_status": np.random.binomial(1, 0.06, n),
            "copd_status": np.random.binomial(1, 0.08, n),
            "age_group": np.random.randint(1, 13, n),
            "survey_year": np.random.choice([2019, 2022], n),
        })

    def test_did_fields(self):
        result = run_did(self.df, "diabetes_status", "exercise_treatment")
        for field in ["did_estimate", "p_value", "significant", "ci_lower", "ci_upper"]:
            assert field in result

    def test_did_estimate_is_float(self):
        result = run_did(self.df, "diabetes_status", "exercise_treatment")
        assert isinstance(result["did_estimate"], float)

    def test_ci_ordering(self):
        result = run_did(self.df, "diabetes_status", "exercise_treatment")
        assert result["ci_lower"] <= result["did_estimate"] <= result["ci_upper"]

    def test_placebo_test_type(self):
        result = run_placebo(self.df, "diabetes_status", "exercise_treatment")
        assert result["test_type"] == "placebo"


class TestDataQuality:
    def setup_method(self):
        np.random.seed(42)
        n = 1000
        self.brfss = pd.DataFrame({
            "diabetes_status": np.random.binomial(1, 0.14, n),
            "heart_disease_status": np.random.binomial(1, 0.06, n),
            "copd_status": np.random.binomial(1, 0.08, n),
            "bmi_category": np.random.randint(1, 5, n),
            "physical_activity": np.random.choice([1, 2], n),
            "smoking_status": np.random.choice([1, 2], n),
            "sleep_hours": np.random.uniform(4, 10, n),
            "age_group": np.random.randint(1, 13, n),
        })
        self.rules = [
            ("diabetes_status", "diabetes_status IN (0,1)", "Diabetes binary"),
            ("bmi_category", "bmi_category IN (1,2,3,4)", "BMI category 1-4"),
            ("sleep_hours", "sleep_hours >= 3 AND sleep_hours <= 12", "Sleep hours 3-12"),
        ]

    def test_dq_checks_run(self):
        con = duckdb.connect()
        df = self.brfss
        con.execute("CREATE TABLE brfss AS SELECT * FROM df")
        results = run_dq_checks(con, "brfss", self.rules)
        assert len(results) > 0
        assert "status" in results.columns
        assert results["status"].isin(["PASS", "WARN", "FAIL"]).all()
        con.close()

    def test_pass_rate_computed(self):
        con = duckdb.connect()
        df = self.brfss
        con.execute("CREATE TABLE brfss AS SELECT * FROM df")
        results = run_dq_checks(con, "brfss", self.rules)
        assert (results["pass_rate_pct"] >= 0).all()
        assert (results["pass_rate_pct"] <= 100).all()
        con.close()


class TestReadmissionTarget:
    def test_binary_target_has_two_classes(self):
        np.random.seed(42)
        rates = np.random.uniform(0.15, 0.30, 1000)
        threshold = np.percentile(rates, 50)
        target = (rates > threshold).astype(int)
        assert len(np.unique(target)) == 2

    def test_target_balance_near_50pct(self):
        np.random.seed(42)
        rates = np.random.uniform(0.15, 0.30, 1000)
        threshold = np.percentile(rates, 50)
        target = (rates > threshold).astype(int)
        assert 0.40 <= target.mean() <= 0.60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
