"""
Feature Signal Decay Analysis Module — HealthPath
Additive enhancement: zero changes to existing pipeline code.

Add to main.py after run_progression_models():

    from analysis.factor_decay import run_factor_decay_analysis
    run_factor_decay_analysis(brfss_df, progression_results, readmission_results)
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

OUTPUT_DIR = Path("data/outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_COLS = ["diabetes_status", "heart_disease_status", "copd_status"]
DISEASE_LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD"
}

FEATURE_COLS = [
    "bmi_category", "age_group", "income_level", "sleep_hours",
    "exercise_days", "smoking_status", "general_health"
]


def save_fig(fig, name):
    import os
    os.makedirs("images", exist_ok=True)
    try:
        fig.write_image(f"images/decay_{name}.png", width=1200, height=600)
        print(f"    Saved images/decay_{name}.png")
    except Exception as e:
        print(f"    Could not save decay_{name}.png: {e}")
    fig.write_html(str(OUTPUT_DIR / f"decay_{name}.html"))


def compute_feature_importance_stability(progression_results):
    """
    Measure model AUC stability across CV folds.
    How consistent is performance across folds per disease?
    """
    results = {}
    stability_records = []

    cv_results = progression_results.get("cv_results", {})
    for disease, cv in cv_results.items():
        label = DISEASE_LABELS.get(disease, disease)
        mean_auc = cv.get("mean_auc", 0)
        std_auc = cv.get("std_auc", 0)
        cv_ratio = round(float(std_auc / mean_auc) if mean_auc > 0 else 0, 4)
        stability_records.append({
            "disease": label,
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc),
            "cv": cv_ratio,
            "stable": bool(cv_ratio < 0.05),
        })

    if stability_records:
        results["auc_stability"] = stability_records
        fig = go.Figure()
        diseases = [r["disease"] for r in stability_records]
        means = [r["mean_auc"] for r in stability_records]
        stds = [r["std_auc"] for r in stability_records]
        colors = ["#2ecc71" if r["stable"] else "#e74c3c" for r in stability_records]

        fig.add_trace(go.Bar(
            x=diseases, y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=colors,
            text=[f"CV={r['cv']:.3f}" for r in stability_records],
            textposition="outside",
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                      annotation_text="Strong (0.80)")
        fig.update_layout(
            title="Model AUC Stability Across CV Folds (green = stable CV<5%)",
            yaxis_title="AUC", yaxis=dict(range=[0.5, 1.0]),
            height=450,
        )
        save_fig(fig, "auc_stability")

    return results


def compute_feature_correlation_decay(brfss_df):
    """
    How does feature-disease Spearman correlation change across age groups?
    Shows which features lose predictive signal at older/younger ages.
    """
    results = {}

    if "age_group" not in brfss_df.columns:
        return results

    age_groups = sorted(brfss_df["age_group"].dropna().unique())
    decay_records = []

    for disease_col in DISEASE_COLS:
        if disease_col not in brfss_df.columns:
            continue
        label = DISEASE_LABELS[disease_col]

        for feat in FEATURE_COLS:
            if feat not in brfss_df.columns or feat == "age_group":
                continue

            age_corrs = []
            for age in age_groups:
                subset = brfss_df[brfss_df["age_group"] == age][[feat, disease_col]].dropna()
                if len(subset) < 30:
                    continue
                try:
                    corr, p = stats.spearmanr(subset[feat], subset[disease_col])
                    age_corrs.append({
                        "age_group": int(age),
                        "correlation": round(float(corr), 4),
                        "p_value": round(float(p), 4),
                        "n": int(len(subset)),
                    })
                except Exception:
                    continue

            if len(age_corrs) >= 3:
                corr_values = [r["correlation"] for r in age_corrs]
                decay_records.append({
                    "disease": label,
                    "feature": feat,
                    "age_correlations": age_corrs,
                    "mean_corr": round(float(np.mean(corr_values)), 4),
                    "corr_std": round(float(np.std(corr_values)), 4),
                    "corr_range": round(float(np.max(corr_values) - np.min(corr_values)), 4),
                    "stable": bool(np.std(corr_values) < 0.05),
                })

    results["feature_correlation_decay"] = decay_records

    if decay_records:
        top_features = sorted(decay_records, key=lambda x: abs(x["mean_corr"]), reverse=True)[:6]
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{r['feature']} → {r['disease']}" for r in top_features]
        )
        colors = {"Diabetes": "#e74c3c", "Heart Disease": "#3498db", "COPD": "#2ecc71"}

        for i, record in enumerate(top_features):
            row, col = i // 3 + 1, i % 3 + 1
            age_corrs = record["age_correlations"]
            ages = [r["age_group"] for r in age_corrs]
            corrs = [r["correlation"] for r in age_corrs]
            fig.add_trace(go.Scatter(
                x=ages, y=corrs,
                mode="lines+markers",
                line=dict(color=colors.get(record["disease"], "#9b59b6"), width=2),
                showlegend=False,
            ), row=row, col=col)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

        fig.update_layout(height=600,
                          title_text="Feature-Disease Correlation Decay Across Age Groups")
        save_fig(fig, "feature_correlation_decay")

    return results


def compute_prevalence_gradient(brfss_df):
    """
    How disease prevalence rises or falls across risk factor levels.
    Strongest gradients = most actionable risk factors.
    """
    results = {}
    gradient_records = []

    for disease_col in DISEASE_COLS:
        if disease_col not in brfss_df.columns:
            continue
        label = DISEASE_LABELS[disease_col]

        for feat in ["bmi_category", "age_group", "income_level", "exercise_days"]:
            if feat not in brfss_df.columns:
                continue

            groups = sorted(brfss_df[feat].dropna().unique())
            if len(groups) < 3:
                continue

            prevalences = []
            for g in groups:
                subset = brfss_df[brfss_df[feat] == g]
                if len(subset) < 20:
                    continue
                prev = subset[disease_col].mean()
                prevalences.append({
                    "group": int(g),
                    "prevalence": round(float(prev * 100), 2),
                    "n": int(len(subset)),
                })

            if len(prevalences) >= 3:
                prev_vals = [p["prevalence"] for p in prevalences]
                gradient = prev_vals[-1] - prev_vals[0]
                gradient_records.append({
                    "disease": label,
                    "feature": feat,
                    "prevalences": prevalences,
                    "gradient": round(float(gradient), 2),
                    "monotonic": bool(
                        all(prev_vals[i] <= prev_vals[i+1] for i in range(len(prev_vals)-1)) or
                        all(prev_vals[i] >= prev_vals[i+1] for i in range(len(prev_vals)-1))
                    ),
                })

    results["prevalence_gradients"] = gradient_records

    if gradient_records:
        top = sorted(gradient_records, key=lambda x: abs(x["gradient"]), reverse=True)[:6]
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f"{r['feature']} → {r['disease']} (Δ={r['gradient']:.1f}%)"
                            for r in top]
        )
        palette = {"Diabetes": "#e74c3c", "Heart Disease": "#3498db", "COPD": "#2ecc71"}

        for i, record in enumerate(top):
            row, col = i // 3 + 1, i % 3 + 1
            groups = [p["group"] for p in record["prevalences"]]
            prevs = [p["prevalence"] for p in record["prevalences"]]
            color = palette.get(record["disease"], "#9b59b6")
            fig.add_trace(go.Scatter(
                x=groups, y=prevs,
                mode="lines+markers",
                fill="tozeroy",
                line=dict(color=color, width=2),
                showlegend=False,
            ), row=row, col=col)

        fig.update_layout(height=600,
                          title_text="Disease Prevalence Gradient Across Risk Factor Levels")
        save_fig(fig, "prevalence_gradients")

    return results


def estimate_feature_half_life(brfss_df):
    """
    Fit exponential growth to prevalence vs risk factor level.
    Estimate at which level disease risk doubles from baseline.
    """
    results = {}
    half_life_records = []

    for disease_col in DISEASE_COLS:
        if disease_col not in brfss_df.columns:
            continue
        label = DISEASE_LABELS[disease_col]
        baseline_prev = float(brfss_df[disease_col].mean())

        for feat in ["bmi_category", "age_group"]:
            if feat not in brfss_df.columns:
                continue

            groups = sorted(brfss_df[feat].dropna().unique())
            points = []
            for g in groups:
                subset = brfss_df[brfss_df[feat] == g]
                if len(subset) > 20:
                    points.append((int(g), float(subset[disease_col].mean())))

            if len(points) < 4:
                continue

            x = np.array([p[0] for p in points], dtype=float)
            y = np.array([p[1] for p in points], dtype=float)

            try:
                def exp_growth(x, a, b):
                    return a * np.exp(b * x)

                popt, _ = curve_fit(exp_growth, x, y, p0=[y[0], 0.1], maxfev=5000)
                a, b = popt

                if b > 0:
                    doubling_level = np.log(2) / b
                    half_life_records.append({
                        "disease": label,
                        "feature": feat,
                        "doubling_level": round(float(doubling_level), 2),
                        "growth_rate": round(float(b), 4),
                        "baseline_prevalence_pct": round(baseline_prev * 100, 2),
                    })
            except Exception:
                continue

    results["risk_doubling_levels"] = half_life_records

    if half_life_records:
        fig = go.Figure()
        for record in half_life_records:
            fig.add_trace(go.Bar(
                name=f"{record['disease']} — {record['feature']}",
                x=[f"{record['disease']}\n({record['feature']})"],
                y=[record["doubling_level"]],
                text=[f"Doubles at level {record['doubling_level']:.1f}"],
                textposition="outside",
            ))
        fig.update_layout(
            title="Risk Factor Level at Which Disease Prevalence Doubles",
            yaxis_title="Risk Factor Level",
            height=450,
            showlegend=False,
        )
        save_fig(fig, "risk_doubling_levels")

    return results


def run_factor_decay_analysis(brfss_df, progression_results, readmission_results):
    print("\n=== Feature Signal Decay Module ===")
    np.random.seed(42)
    all_results = {}

    print("  [1/4] AUC stability across CV folds...")
    all_results["stability"] = compute_feature_importance_stability(progression_results)

    print("  [2/4] Feature-disease correlation decay across age groups...")
    all_results["correlation_decay"] = compute_feature_correlation_decay(brfss_df)

    print("  [3/4] Disease prevalence gradients across risk factors...")
    all_results["prevalence_gradients"] = compute_prevalence_gradient(brfss_df)

    print("  [4/4] Risk factor doubling levels (exponential fit)...")
    all_results["half_life"] = estimate_feature_half_life(brfss_df)

    output_path = OUTPUT_DIR / "factor_decay.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"  Results saved: {output_path}")
    print(f"  Plots: images/decay_auc_stability.png, decay_feature_correlation_decay.png,")
    print(f"         decay_prevalence_gradients.png, decay_risk_doubling_levels.png")
    return all_results
