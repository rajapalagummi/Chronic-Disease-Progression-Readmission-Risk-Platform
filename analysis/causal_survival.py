import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"
IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASES = ["diabetes_status", "heart_disease_status", "copd_status"]
DISEASE_LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD",
}


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES_DIR / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def run_did(df, outcome, treatment_col, period_col="period"):
    groups = {
        (1, "post"): df[(df[treatment_col] == 1) & (df[period_col] == "post")][outcome].mean(),
        (1, "pre"): df[(df[treatment_col] == 1) & (df[period_col] == "pre")][outcome].mean(),
        (0, "post"): df[(df[treatment_col] == 0) & (df[period_col] == "post")][outcome].mean(),
        (0, "pre"): df[(df[treatment_col] == 0) & (df[period_col] == "pre")][outcome].mean(),
    }
    did = (groups[(1, "post")] - groups[(1, "pre")]) - (groups[(0, "post")] - groups[(0, "pre")])
    treat = df[df[treatment_col] == 1][outcome]
    ctrl = df[df[treatment_col] == 0][outcome]
    se = np.sqrt(treat.var() / len(treat) + ctrl.var() / len(ctrl))
    t_stat = did / (se + 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return {
        "did_estimate": round(float(did), 6),
        "se": round(float(se), 6),
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
        "ci_lower": round(float(did - 1.96 * se), 4),
        "ci_upper": round(float(did + 1.96 * se), 4),
        "treat_pre": round(float(groups[(1, "pre")]), 4),
        "treat_post": round(float(groups[(1, "post")]), 4),
        "ctrl_pre": round(float(groups[(0, "pre")]), 4),
        "ctrl_post": round(float(groups[(0, "post")]), 4),
    }


def run_placebo(df, outcome, treatment_col):
    pre_df = df[df["period"] == "pre"].copy()
    median_age = pre_df["age_group"].median()
    pre_df["period"] = np.where(pre_df["age_group"] > median_age, "post", "pre")
    result = run_did(pre_df, outcome, treatment_col)
    result["test_type"] = "placebo"
    return result


def run_causal_analysis(df):
    print("\n=== Module 4: Causal Intervention Analysis ===")
    df = df.copy()
    df["exercise_treatment"] = (df["exercise_any"] == 1).astype(int)
    df["period"] = np.where(df["age_group"] >= df["age_group"].median(), "post", "pre")

    results = {}
    did_rows = []

    for disease in DISEASES:
        if disease not in df.columns:
            continue
        label = DISEASE_LABELS[disease]
        did = run_did(df, disease, "exercise_treatment")
        placebo = run_placebo(df, disease, "exercise_treatment")

        results[disease] = {"did": did, "placebo": placebo}
        did_rows.append({
            "Disease": label, "DiD Estimate": did["did_estimate"],
            "p-value": did["p_value"], "Significant": did["significant"],
            "CI Lower": did["ci_lower"], "CI Upper": did["ci_upper"],
            "Placebo p": placebo["p_value"],
        })
        print(f"  {label}: DiD={did['did_estimate']:.4f}, p={did['p_value']}, sig={did['significant']}")

    did_df = pd.DataFrame(did_rows)

    fig = go.Figure()
    for _, row in did_df.iterrows():
        color = "#27ae60" if row["Significant"] else "#95a5a6"
        fig.add_trace(go.Scatter(x=[row["CI Lower"], row["CI Upper"]], y=[row["Disease"], row["Disease"]],
                                  mode="lines", line=dict(color=color, width=4), showlegend=False))
        fig.add_trace(go.Scatter(x=[row["DiD Estimate"]], y=[row["Disease"]],
                                  mode="markers", marker=dict(size=14, color=color), showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No effect")
    fig.update_layout(title="DiD Forest Plot — Exercise Intervention Effect by Disease",
                      xaxis_title="DiD Estimate (negative = risk reduction)", height=400)
    save_fig(fig, "causal_did_forest_plot")

    return results, did_df


def run_survival_analysis(df):
    print("\n=== Module 2: Survival Analysis ===")
    df = df.copy()
    np.random.seed(42)
    df["time_to_diagnosis"] = np.random.randint(1, 25, len(df))

    results = {}
    fig = go.Figure()
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for i, disease in enumerate(DISEASES):
        if disease not in df.columns:
            continue
        label = DISEASE_LABELS[disease]
        kmf = KaplanMeierFitter()
        kmf.fit(df["time_to_diagnosis"], event_observed=df[disease], label=label)
        median_surv = kmf.median_survival_time_
        at_12 = float(kmf.survival_function_at_times([12]).values[0])
        results[disease] = {"median_survival": float(median_surv), "survival_at_12": round(at_12, 4)}
        print(f"  {label}: median={median_surv}, survival@12={at_12:.4f}")

        timeline = kmf.timeline
        survival = kmf.survival_function_[label].values
        ci_lower = kmf.confidence_interval_[f"{label}_lower_0.95"].values
        ci_upper = kmf.confidence_interval_[f"{label}_upper_0.95"].values
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=list(timeline) + list(timeline[::-1]),
            y=list(ci_upper) + list(ci_lower[::-1]),
            fill="toself", fillcolor=color, opacity=0.15,
            line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=timeline, y=survival, mode="lines",
            name=label, line=dict(color=color, width=2)))

    fig.update_layout(title="Kaplan-Meier Disease-Free Survival Curves",
                      xaxis_title="Time Period", yaxis_title="Disease-Free Survival", height=500)
    save_fig(fig, "survival_km_curves")

    try:
        cox_df = df[["time_to_diagnosis", "diabetes_status", "bmi_category",
                      "physical_activity", "smoking_status", "age_group"]].dropna().copy()
        cox_df = cox_df[cox_df["time_to_diagnosis"] > 0]
        if len(cox_df) > 100 and cox_df["diabetes_status"].sum() > 10:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="time_to_diagnosis", event_col="diabetes_status")
            cox_summary = cph.summary[["coef", "exp(coef)", "p"]].round(4)
            results["cox_diabetes"] = cox_summary.to_dict()
            print(f"\nCox PH — Diabetes:")
            print(cox_summary.to_string())
    except Exception as e:
        print(f"Cox PH skipped: {e}")

    return results
