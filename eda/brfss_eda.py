import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def run_brfss_eda(df):
    print("\n=== BRFSS EDA ===")
    print(f"Shape: {df.shape}")
    print(f"Total respondents: {len(df):,}")

    print(f"\nNumerical summary:")
    print(df.describe().round(3).to_string())

    print(f"\nMissing values:")
    print(df.isnull().sum().to_string())

    print(f"\nData types:")
    print(df.dtypes.to_string())

    for col in df.select_dtypes(include=[np.number]).columns[:10]:
        s = df[col].dropna()
        print(f"\n{col}: mean={s.mean():.3f}, median={s.median():.3f}, "
              f"std={s.std():.3f}, skew={s.skew():.3f}, kurtosis={s.kurtosis():.3f}, "
              f"min={s.min():.1f}, max={s.max():.1f}")

    disease_rates = {DISEASE_LABELS[d]: round(df[d].mean() * 100, 2) for d in DISEASES if d in df.columns}
    print(f"\nDisease prevalence: {disease_rates}")

    fig = px.bar(
        x=list(disease_rates.keys()), y=list(disease_rates.values()),
        title="Disease Prevalence in Real BRFSS 2022 Data (%)",
        labels={"x": "Disease", "y": "Prevalence (%)"},
        color=list(disease_rates.values()), color_continuous_scale="Reds",
        text=[f"{v:.1f}%" for v in disease_rates.values()],
    )
    fig.update_traces(textposition="outside")
    save_fig(fig, "brfss_disease_prevalence")

    avail_diseases = [d for d in DISEASES if d in df.columns]
    corr = df[avail_diseases].corr()
    fig = px.imshow(corr, title="Disease Co-occurrence Correlation",
                    color_continuous_scale="RdBu_r", text_auto=".3f",
                    x=[DISEASE_LABELS[d] for d in avail_diseases],
                    y=[DISEASE_LABELS[d] for d in avail_diseases])
    save_fig(fig, "brfss_disease_correlation")

    if "bmi_category" in df.columns:
        bmi_disease = df.groupby("bmi_category")[avail_diseases].mean() * 100
        bmi_disease.index.name = "bmi_category"
        bmi_labels = {1: "Underweight", 2: "Normal", 3: "Overweight", 4: "Obese"}
        bmi_disease.index = [bmi_labels.get(i, str(i)) for i in bmi_disease.index]
        bmi_disease.index.name = "bmi_category"
        fig = px.bar(
            bmi_disease.reset_index().melt(id_vars="bmi_category"),
            x="bmi_category", y="value", color="variable",
            title="Disease Prevalence by BMI Category (%)",
            labels={"bmi_category": "BMI Category", "value": "Prevalence (%)", "variable": "Disease"},
            barmode="group",
        )
        save_fig(fig, "brfss_bmi_disease")

    if "physical_activity" in df.columns:
        act_disease = df.groupby("physical_activity")[avail_diseases].mean() * 100
        act_disease.index.name = "physical_activity"
        act_labels = {1: "Active", 2: "Inactive"}
        act_disease.index = [act_labels.get(i, str(i)) for i in act_disease.index]
        act_disease.index.name = "physical_activity"
        fig = px.bar(
            act_disease.reset_index().melt(id_vars="physical_activity"),
            x="physical_activity", y="value", color="variable",
            title="Disease Prevalence by Physical Activity Level (%)",
            labels={"physical_activity": "Activity Level", "value": "Prevalence (%)", "variable": "Disease"},
            barmode="group",
        )
        save_fig(fig, "brfss_activity_disease")

    if "smoking_status" in df.columns:
        smoke_disease = df.groupby("smoking_status")[avail_diseases].mean() * 100
        smoke_disease.index.name = "smoking_status"
        smoke_labels = {1: "Smoker", 2: "Non-Smoker"}
        smoke_disease.index = [smoke_labels.get(i, str(i)) for i in smoke_disease.index]
        smoke_disease.index.name = "smoking_status"
        fig = px.bar(
            smoke_disease.reset_index().melt(id_vars="smoking_status"),
            x="smoking_status", y="value", color="variable",
            title="Disease Prevalence by Smoking Status (%)",
            labels={"smoking_status": "Smoking Status", "value": "Prevalence (%)", "variable": "Disease"},
            barmode="group",
        )
        save_fig(fig, "brfss_smoking_disease")

    if "sleep_hours" in df.columns:
        fig = px.histogram(
            df, x="sleep_hours", color_discrete_sequence=["#3498db"],
            title="Sleep Hours Distribution (Real BRFSS 2022)",
            nbins=20, labels={"sleep_hours": "Sleep Hours"},
        )
        fig.add_vline(x=df["sleep_hours"].mean(), line_dash="dash",
                      annotation_text=f"Mean: {df['sleep_hours'].mean():.1f}h")
        save_fig(fig, "brfss_sleep_distribution")

    if "age_group" in df.columns:
        age_disease = df.groupby("age_group")[avail_diseases].mean() * 100
        age_disease.index.name = "age_group"
        fig = px.line(
            age_disease.reset_index().melt(id_vars="age_group"),
            x="age_group", y="value", color="variable",
            title="Disease Prevalence by Age Group (%)",
            labels={"age_group": "Age Group (1=18-24, 13=80+)", "value": "Prevalence (%)", "variable": "Disease"},
            markers=True,
        )
        save_fig(fig, "brfss_age_disease")

    if "income_category" in df.columns:
        income_disease = df.groupby("income_category")[avail_diseases].mean() * 100
        income_disease.index.name = "income_category"
        fig = px.line(
            income_disease.reset_index().melt(id_vars="income_category"),
            x="income_category", y="value", color="variable",
            title="Disease Prevalence by Income Category (%)",
            labels={"income_category": "Income (1=Low, 8=High)", "value": "Prevalence (%)", "variable": "Disease"},
            markers=True,
        )
        save_fig(fig, "brfss_income_disease")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 2:
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix, title="Full Feature Correlation Matrix",
                        color_continuous_scale="RdBu_r", text_auto=".2f")
        fig.update_layout(height=900)
        save_fig(fig, "brfss_full_correlation")

    if "general_health" in df.columns:
        gh_disease = df.groupby("general_health")[avail_diseases].mean() * 100
        gh_disease.index.name = "general_health"
        gh_labels = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
        gh_disease.index = [gh_labels.get(i, str(i)) for i in gh_disease.index]
        gh_disease.index.name = "general_health"
        fig = px.bar(
            gh_disease.reset_index().melt(id_vars="general_health"),
            x="general_health", y="value", color="variable",
            title="Disease Prevalence by Self-Reported General Health (%)",
            labels={"general_health": "General Health", "value": "Prevalence (%)", "variable": "Disease"},
            barmode="group",
        )
        save_fig(fig, "brfss_general_health_disease")

    if "sex" in df.columns:
        sex_disease = df.groupby("sex")[avail_diseases].mean() * 100
        sex_disease.index.name = "sex"
        sex_labels = {1: "Male", 2: "Female"}
        sex_disease.index = [sex_labels.get(i, str(i)) for i in sex_disease.index]
        sex_disease.index.name = "sex"
        fig = px.bar(
            sex_disease.reset_index().melt(id_vars="sex"),
            x="sex", y="value", color="variable",
            title="Disease Prevalence by Sex (%)",
            labels={"sex": "Sex", "value": "Prevalence (%)", "variable": "Disease"},
            barmode="group",
        )
        save_fig(fig, "brfss_sex_disease")

    print("\nBRFSS EDA complete — all plots saved")
    return df
