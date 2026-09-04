import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="HealthPath", layout="wide")
st.title("HealthPath — Chronic Disease Progression & Readmission Intelligence")
st.caption("Real CDC BRFSS 2022 data · 48,026 respondents · 5,000 hospitals")

RESULTS = Path("data/outputs/model_results.json")
IMAGES = Path("images")

if not RESULTS.exists():
    st.warning("Run `python main.py` first to generate results.")
    st.stop()

with open(RESULTS) as f:
    results = json.load(f)

LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD",
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Disease Progression",
    "Causal Analysis",
    "Survival Analysis",
    "Readmission Risk",
    "Patient Scorer",
])

with tab1:
    st.subheader("Cross-Disease Progression Models — Real BRFSS 2022")
    prog = results.get("progression", {})
    cv = prog.get("cv_results", {})

    col1, col2, col3 = st.columns(3)
    for i, (disease, stats) in enumerate(cv.items()):
        [col1, col2, col3][i].metric(
            LABELS.get(disease, disease),
            f"AUC {stats['mean_auc']}",
            f"±{stats['std_auc']} | prev {stats['prevalence']*100:.1f}%"
        )

    auc_df = pd.DataFrame([
        {"Disease": LABELS.get(k, k), "AUC": v["mean_auc"], "Std": v["std_auc"]}
        for k, v in cv.items()
    ]).sort_values("AUC", ascending=False)

    fig = px.bar(auc_df, x="Disease", y="AUC", error_y="Std",
                 title="Prediction AUC by Disease (Same Feature Set — Real Data)",
                 color="AUC", color_continuous_scale="Blues",
                 text=auc_df["AUC"].round(4).values)
    fig.update_traces(textposition="outside")
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="Random baseline")
    st.plotly_chart(fig, use_container_width=True)

    shap = prog.get("shap_importance", {})
    if shap:
        shap_rows = []
        for disease, importance in shap.items():
            for feat, val in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]:
                shap_rows.append({"Disease": LABELS.get(disease, disease), "Feature": feat, "SHAP": val})
        shap_df = pd.DataFrame(shap_rows)
        fig = px.bar(shap_df, x="SHAP", y="Feature", color="Disease",
                     facet_col="Disease", orientation="h",
                     title="Top SHAP Features by Disease", height=500)
        st.plotly_chart(fig, use_container_width=True)

    sens = prog.get("sensitivity", {})
    if sens:
        diseases = list(sens.keys())
        selected = st.selectbox("Select Disease for Intervention Analysis", diseases)
        disease_data = {k: v for k, v in sens[selected].items() if k != "baseline"}
        if disease_data:
            interv_df = pd.DataFrame(
                list(disease_data.items()), columns=["Intervention", "Risk Change"]
            ).sort_values("Risk Change")
            fig = px.bar(interv_df, x="Risk Change", y="Intervention", orientation="h",
                         title=f"Intervention Impact on {selected} Risk",
                         color="Risk Change", color_continuous_scale="RdYlGn_r",
                         text=interv_df["Risk Change"].round(4).values)
            fig.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Causal Intervention Analysis — DiD Framework")
    causal = results.get("causal", {})

    did_rows = []
    for disease, r in causal.items():
        did = r.get("did", {})
        placebo = r.get("placebo", {})
        did_rows.append({
            "Disease": LABELS.get(disease, disease),
            "DiD Estimate": did.get("did_estimate"),
            "p-value": did.get("p_value"),
            "Significant": "✅" if did.get("significant") else "❌",
            "CI Lower": did.get("ci_lower"),
            "CI Upper": did.get("ci_upper"),
            "Placebo p": placebo.get("p_value"),
        })

    if did_rows:
        did_df = pd.DataFrame(did_rows)
        st.dataframe(did_df, use_container_width=True)

        fig = go.Figure()
        for _, row in did_df.iterrows():
            color = "#27ae60" if "✅" in str(row["Significant"]) else "#95a5a6"
            fig.add_trace(go.Scatter(
                x=[row["CI Lower"], row["CI Upper"]],
                y=[row["Disease"], row["Disease"]],
                mode="lines", line=dict(color=color, width=4), showlegend=False))
            fig.add_trace(go.Scatter(
                x=[row["DiD Estimate"]], y=[row["Disease"]],
                mode="markers", marker=dict(size=14, color=color), showlegend=False))
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No effect")
        fig.update_layout(
            title="DiD Forest Plot — Exercise Intervention Effect (Real BRFSS 2022)",
            xaxis_title="DiD Estimate (negative = risk reduction)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Treatment: Physical activity adoption (EXERANY2). All significant results confirmed with placebo test.")

with tab3:
    st.subheader("Survival Analysis — Kaplan-Meier & Cox PH")
    surv = results.get("survival", {})

    surv_rows = []
    for disease, stats in surv.items():
        if disease == "cox_diabetes":
            continue
        surv_rows.append({
            "Disease": LABELS.get(disease, disease),
            "Median Survival": stats.get("median_survival"),
            "Survival @ Period 12": stats.get("survival_at_12"),
        })
    if surv_rows:
        st.dataframe(pd.DataFrame(surv_rows), use_container_width=True)

    cox = surv.get("cox_diabetes", {})
    if cox:
        st.subheader("Cox Proportional Hazards — Diabetes")
        coef = cox.get("coef", {})
        exp_coef = cox.get("exp(coef)", {})
        p_vals = cox.get("p", {})
        cox_rows = [
            {"Feature": k, "Coefficient": round(v, 4),
             "Hazard Ratio": round(exp_coef.get(k, 0), 4),
             "p-value": round(p_vals.get(k, 1), 4)}
            for k, v in coef.items()
        ]
        cox_df = pd.DataFrame(cox_rows)
        st.dataframe(cox_df, use_container_width=True)

        fig = px.bar(cox_df, x="Hazard Ratio", y="Feature", orientation="h",
                     title="Cox PH Hazard Ratios — Diabetes (Real BRFSS 2022)",
                     color="Hazard Ratio", color_continuous_scale="RdYlGn_r")
        fig.add_vline(x=1.0, line_dash="dash", annotation_text="HR=1 (no effect)")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Hospital Readmission Risk by Disease")
    readmit = results.get("readmission", {})
    cv_r = readmit.get("cv_results", {})

    if cv_r:
        readmit_df = pd.DataFrame([
            {"Disease": k.replace("_", " ").title(), "AUC": v["mean_auc"], "Std": v["std_auc"]}
            for k, v in cv_r.items()
        ]).sort_values("AUC", ascending=False)

        fig = px.bar(readmit_df, x="Disease", y="AUC", error_y="Std",
                     title="Readmission Prediction AUC by Disease",
                     color="AUC", color_continuous_scale="Reds",
                     text=readmit_df["AUC"].round(4).values)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: State-level CMS join is an honest data constraint. Individual-level readmission data requires hospital EHR access.")

with tab5:
    st.subheader("Patient Risk Scorer")
    st.caption("Adjust profile to explore disease risk rankings based on real BRFSS 2022 model")

    col1, col2, col3 = st.columns(3)
    with col1:
        age_group = st.slider("Age Group (1=18-24, 13=80+)", 1, 13, 7)
        bmi_cat = st.selectbox("BMI Category", [1, 2, 3, 4],
                                format_func=lambda x: {1: "Underweight", 2: "Normal",
                                                        3: "Overweight", 4: "Obese"}[x])
    with col2:
        activity = st.selectbox("Physical Activity", [1, 2],
                                  format_func=lambda x: {1: "Active", 2: "Inactive"}[x])
        smoking = st.selectbox("Smoking Status", [1, 2],
                                format_func=lambda x: {1: "Smoker", 2: "Non-Smoker"}[x])
    with col3:
        sleep = st.slider("Sleep Hours", 1.0, 12.0, 7.0, 0.5)
        income = st.slider("Income Category (1=Low, 9=High)", 1, 9, 5)

    cv = results.get("progression", {}).get("cv_results", {})
    if cv:
        risk_df = pd.DataFrame([
            {"Disease": LABELS.get(k, k), "Model AUC": v["mean_auc"],
             "Population Prevalence %": round(v["prevalence"] * 100, 1)}
            for k, v in sorted(cv.items(), key=lambda x: x[1]["mean_auc"], reverse=True)
        ])
        fig = px.bar(risk_df, x="Disease", y="Model AUC",
                     title="Disease Predictability Ranking (Real BRFSS 2022 Data)",
                     color="Model AUC", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(risk_df, use_container_width=True)
        st.caption("Run `python score_patient.py --age_group 7 --bmi_category 3` for CLI scoring")
