"""
Statistical Analysis Module — HealthPath
Additive enhancement: zero changes to existing pipeline code.

Add to main.py after [7/7]:

    from analysis.statistical import run_statistical_analysis
    run_statistical_analysis(brfss_df, cms_df, progression_results, readmission_results, causal_results)
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("data/outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_COLS = ["diabetes_status", "heart_disease_status", "copd_status"]
DISEASE_LABELS = {"diabetes_status": "Diabetes", "heart_disease_status": "Heart Disease", "copd_status": "COPD"}

FEATURE_COLS = ["bmi_category", "age_group", "income_level", "sleep_hours",
                "exercise_days", "smoking_status", "general_health"]


def save_fig(fig, name):
    import os
    os.makedirs("images", exist_ok=True)
    try:
        fig.write_image(f"images/stat_{name}.png", width=1200, height=600)
    except Exception as e:
        print(f"    Could not save stat_{name}.png: {e}")
    fig.write_html(str(OUTPUT_DIR / f"stat_{name}.html"))


def describe_series(series, label):
    s = series.dropna()
    if len(s) < 4:
        return {}
    _, p_ks = stats.kstest(s, stats.norm(loc=s.mean(), scale=s.std()).cdf)
    _, p_jb = stats.jarque_bera(s)
    return {
        "label": label,
        "n": int(len(s)),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "is_normal_ks": bool(p_ks > 0.05),
        "is_normal_jb": bool(p_jb > 0.05),
    }


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    if len(values) < 3:
        return None, None
    boots = [np.mean(np.random.choice(values, len(values), replace=True))
             for _ in range(n_bootstrap)]
    alpha = 1 - ci
    return (round(float(np.percentile(boots, alpha/2*100)), 4),
            round(float(np.percentile(boots, (1-alpha/2)*100)), 4))


def disease_prevalence_analysis(brfss_df):
    results = {}
    prevalence_data = []

    for col in DISEASE_COLS:
        if col not in brfss_df.columns:
            continue
        prev = brfss_df[col].mean()
        ci_low, ci_high = bootstrap_ci(brfss_df[col].values)
        prevalence_data.append({
            "disease": DISEASE_LABELS[col],
            "prevalence": round(float(prev * 100), 2),
            "ci_lower": round(float(ci_low * 100), 2) if ci_low else None,
            "ci_upper": round(float(ci_high * 100), 2) if ci_high else None,
            "n_cases": int(brfss_df[col].sum()),
            "n_total": int(len(brfss_df)),
        })

    results["prevalence"] = prevalence_data

    fig = go.Figure()
    diseases = [d["disease"] for d in prevalence_data]
    prevs = [d["prevalence"] for d in prevalence_data]
    ci_lows = [d["prevalence"] - d["ci_lower"] if d["ci_lower"] else 0 for d in prevalence_data]
    ci_highs = [d["ci_upper"] - d["prevalence"] if d["ci_upper"] else 0 for d in prevalence_data]

    fig.add_trace(go.Bar(
        x=diseases, y=prevs,
        marker_color=["#e74c3c", "#3498db", "#2ecc71"],
        error_y=dict(type="data", symmetric=False,
                     array=ci_highs, arrayminus=ci_lows, visible=True),
        text=[f"{p:.1f}%" for p in prevs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Disease Prevalence with 95% Bootstrap CIs — BRFSS 2022",
        yaxis_title="Prevalence (%)", height=450,
    )
    save_fig(fig, "disease_prevalence_ci")
    return results


def feature_distribution_analysis(brfss_df):
    results = {}
    valid_features = [f for f in FEATURE_COLS if f in brfss_df.columns]

    if not valid_features:
        return results

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f.replace("_", " ").title() for f in valid_features[:8]]
    )

    for i, feat in enumerate(valid_features[:8]):
        row, col = i // 4 + 1, i % 4 + 1
        fig.add_trace(go.Histogram(
            x=brfss_df[feat].dropna(),
            name=feat, showlegend=False,
            marker_color="#3498db", opacity=0.7,
        ), row=row, col=col)
        results[feat] = describe_series(brfss_df[feat].dropna(), feat)

    fig.update_layout(height=600, title_text="Feature Distributions — BRFSS 2022")
    save_fig(fig, "feature_distributions")
    return results


def auc_bootstrap_analysis(progression_results, readmission_results):
    results = {}

    all_aucs = []
    for label, res_dict in [("Progression", progression_results), ("Readmission", readmission_results)]:
        cv = res_dict.get("cv_results", {})
        for disease, stats_d in cv.items():
            mean_auc = stats_d.get("mean_auc", 0)
            std_auc = stats_d.get("std_auc", 0)
            disease_label = DISEASE_LABELS.get(disease, disease)
            all_aucs.append({
                "model": label,
                "disease": disease_label,
                "mean_auc": float(mean_auc),
                "std_auc": float(std_auc),
                "ci_lower": round(float(mean_auc) - 1.96 * float(std_auc), 4),
                "ci_upper": round(float(mean_auc) + 1.96 * float(std_auc), 4),
            })

    results["auc_summary"] = all_aucs

    if all_aucs:
        fig = go.Figure()
        for model_type in ["Progression", "Readmission"]:
            subset = [a for a in all_aucs if a["model"] == model_type]
            if not subset:
                continue
            diseases = [a["disease"] for a in subset]
            means = [a["mean_auc"] for a in subset]
            ci_low = [a["mean_auc"] - a["ci_lower"] for a in subset]
            ci_high = [a["ci_upper"] - a["mean_auc"] for a in subset]
            fig.add_trace(go.Bar(
                name=model_type, x=diseases, y=means,
                error_y=dict(type="data", symmetric=False,
                             array=ci_high, arrayminus=ci_low, visible=True),
                text=[f"{m:.3f}" for m in means],
                textposition="outside",
            ))

        fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                      annotation_text="Clinical threshold (0.70)")
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                      annotation_text="Strong performance (0.80)")
        fig.update_layout(
            title="Model AUC with 95% Confidence Intervals",
            yaxis_title="AUC", barmode="group",
            yaxis=dict(range=[0.5, 1.0]), height=450,
        )
        save_fig(fig, "model_auc_ci")

    return results


def correlation_analysis(brfss_df):
    results = {}
    valid_features = [f for f in FEATURE_COLS + DISEASE_COLS if f in brfss_df.columns]

    if len(valid_features) < 3:
        return results

    corr_matrix = brfss_df[valid_features].corr(method="spearman")
    results["spearman_correlation"] = corr_matrix.round(4).to_dict()

    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=[f.replace("_", " ") for f in corr_matrix.columns],
        y=[f.replace("_", " ") for f in corr_matrix.index],
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Spearman Correlation Matrix — Features & Disease Outcomes",
        height=600,
    )
    save_fig(fig, "correlation_matrix")
    return results


def run_statistical_analysis(brfss_df, cms_df, progression_results,
                              readmission_results, causal_results):
    print("\n=== Statistical Analysis Module ===")
    np.random.seed(42)
    all_results = {}

    print("  Disease prevalence with bootstrap CIs...")
    all_results["prevalence"] = disease_prevalence_analysis(brfss_df)

    print("  Feature distribution analysis...")
    all_results["features"] = feature_distribution_analysis(brfss_df)

    print("  AUC bootstrap confidence intervals...")
    all_results["auc"] = auc_bootstrap_analysis(progression_results, readmission_results)

    print("  Spearman correlation matrix...")
    all_results["correlation"] = correlation_analysis(brfss_df)

    output_path = OUTPUT_DIR / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"  Results saved: {output_path}")
    print(f"  Plots saved: images/stat_*.png")
    return all_results
