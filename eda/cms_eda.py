import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"
IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES_DIR / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def run_cms_eda(df):
    print("\n=== CMS EDA ===")
    print(f"Shape: {df.shape}")
    print(f"Unique hospitals: {df['hospital_id'].nunique():,}")
    print(f"\nNumerical summary:")
    print(df.describe().round(4).to_string())
    print(f"\nMissing values:")
    print(df.isnull().sum().to_string())

    for col in ["readmission_rate_30day", "n_cases"]:
        s = df[col].dropna()
        print(f"\n{col}: mean={s.mean():.4f}, median={s.median():.4f}, "
              f"std={s.std():.4f}, skew={s.skew():.3f}, "
              f"min={s.min():.4f}, max={s.max():.4f}")

    disease_readmit = df.groupby("disease")["readmission_rate_30day"].agg(["mean", "std", "median"])
    print(f"\nReadmission rates by disease:")
    print(disease_readmit.round(4).to_string())

    fig = px.box(df, x="disease", y="readmission_rate_30day",
                 title="30-Day Readmission Rate by Disease",
                 color="disease", labels={"disease": "Disease", "readmission_rate_30day": "Readmission Rate"})
    save_fig(fig, "cms_readmit_by_disease")

    fig = px.bar(disease_readmit.reset_index(), x="disease", y="mean", error_y="std",
                 title="Mean 30-Day Readmission Rate by Disease",
                 color="mean", color_continuous_scale="Reds",
                 text=disease_readmit["mean"].round(3).values)
    fig.update_traces(textposition="outside")
    save_fig(fig, "cms_readmit_mean_by_disease")

    tier_disease = df.groupby(["quality_tier", "disease"])["readmission_rate_30day"].mean().reset_index()
    fig = px.line(tier_disease, x="quality_tier", y="readmission_rate_30day", color="disease",
                  title="Readmission Rate by Hospital Quality Tier",
                  labels={"quality_tier": "Quality Tier (1=Worst, 5=Best)", "readmission_rate_30day": "Readmission Rate"},
                  markers=True)
    save_fig(fig, "cms_quality_tier_disease")

    state_readmit = df.groupby("state")["readmission_rate_30day"].mean().reset_index()
    valid_states = state_readmit[state_readmit["state"].str.len() == 2]
    if len(valid_states) > 5:
        fig = px.choropleth(valid_states, locations="state", locationmode="USA-states",
                            color="readmission_rate_30day", scope="usa",
                            title="Mean 30-Day Readmission Rate by State",
                            color_continuous_scale="Reds",
                            labels={"readmission_rate_30day": "Readmission Rate"})
        save_fig(fig, "cms_state_map")

    fig = px.histogram(df, x="readmission_rate_30day", color="disease",
                       title="Readmission Rate Distribution by Disease",
                       nbins=30, barmode="overlay", opacity=0.7,
                       labels={"readmission_rate_30day": "30-Day Readmission Rate"})
    fig.add_vline(x=df["readmission_rate_30day"].mean(), line_dash="dash",
                  annotation_text=f"Mean: {df['readmission_rate_30day'].mean():.3f}")
    save_fig(fig, "cms_readmit_histogram")

    fig = px.scatter(df, x="n_cases", y="readmission_rate_30day", color="disease",
                     title="Case Volume vs Readmission Rate",
                     labels={"n_cases": "Number of Cases", "readmission_rate_30day": "Readmission Rate"},
                     opacity=0.4, trendline="ols")
    save_fig(fig, "cms_volume_readmit_scatter")

    teach_disease = df.groupby(["teaching_hospital", "disease"])["readmission_rate_30day"].mean().reset_index()
    teach_disease["teaching_hospital"] = teach_disease["teaching_hospital"].map({0: "Non-Teaching", 1: "Teaching"})
    fig = px.bar(teach_disease, x="disease", y="readmission_rate_30day", color="teaching_hospital",
                 title="Readmission Rate: Teaching vs Non-Teaching Hospitals",
                 barmode="group", labels={"readmission_rate_30day": "Readmission Rate"})
    save_fig(fig, "cms_teaching_disease")

    print("\nCMS EDA complete — all plots saved")
    return df
