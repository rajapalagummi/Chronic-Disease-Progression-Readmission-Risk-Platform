

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

OUTPUT_DIR = Path("data/outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DISEASE_COLS = ["diabetes_status", "heart_disease_status", "copd_status"]
DISEASE_LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD",
}

RISK_FACTORS = [
    "bmi_category", "smoking_status", "physical_activity",
    "sleep_hours", "income_category", "general_health",
]

DEMOGRAPHIC_COLS = ["age_group", "income_level", "general_health"]


def save_fig(fig, name):
    import os
    os.makedirs("images", exist_ok=True)
    try:
        fig.write_image(f"images/eda_{name}.png", width=1200, height=650)
        print(f"    Saved images/eda_{name}.png")
    except Exception as e:
        print(f"    Could not save eda_{name}.png: {e}")
    fig.write_html(str(OUTPUT_DIR / f"eda_{name}.html"))


# ─────────────────────────────────────────────
# 1. DEMOGRAPHIC INTERSECTIONALITY HEATMAP
# ─────────────────────────────────────────────

def compute_demographic_intersectionality(brfss_df):
    """
    For each disease, compute prevalence across age × income intersections.
    Reveals which demographic pockets carry the highest burden.
    """
    results = {}
    records = []

    age_col = "age_group"
    inc_col = "income_category"

    for disease_col in DISEASE_COLS:
        if not all(c in brfss_df.columns for c in [age_col, inc_col, disease_col]):
            continue
        label = DISEASE_LABELS[disease_col]

        pivot_data = []
        ages = sorted(brfss_df[age_col].dropna().unique())
        incomes = sorted(brfss_df[inc_col].dropna().unique())

        for age in ages:
            row_vals = []
            for inc in incomes:
                subset = brfss_df[
                    (brfss_df[age_col] == age) & (brfss_df[inc_col] == inc)
                ][disease_col].dropna()
                if len(subset) >= 20:
                    row_vals.append(round(float(subset.mean() * 100), 2))
                else:
                    row_vals.append(None)
            pivot_data.append(row_vals)

        records.append({
            "disease": label,
            "age_groups": [int(a) for a in ages],
            "income_levels": [int(i) for i in incomes],
            "prevalence_matrix": pivot_data,
        })

    results["demographic_intersectionality"] = records

    if records:
        fig = make_subplots(
            rows=1, cols=len(records),
            subplot_titles=[r["disease"] for r in records],
        )

        colorscales = ["Reds", "Blues", "Greens"]
        for i, record in enumerate(records):
            z = record["prevalence_matrix"]
            x_labels = [str(v) for v in record["income_levels"]]
            y_labels = [str(v) for v in record["age_groups"]]

            z_clean = [
                [v if v is not None else 0 for v in row]
                for row in z
            ]

            fig.add_trace(
                go.Heatmap(
                    z=z_clean,
                    x=x_labels,
                    y=y_labels,
                    colorscale=colorscales[i % len(colorscales)],
                    showscale=True,
                    colorbar=dict(x=0.3 + i * 0.35, len=0.9),
                    name=record["disease"],
                    text=[[f"{v:.1f}%" if v else "" for v in row] for row in z],
                    texttemplate="%{text}",
                    textfont={"size": 8},
                ),
                row=1, col=i + 1,
            )
            fig.update_xaxes(title_text="Income Level", row=1, col=i + 1)
            fig.update_yaxes(title_text="Age Group", row=1, col=i + 1)

        fig.update_layout(
            title_text="Disease Prevalence (%) by Age × Income Intersection",
            height=520,
        )
        save_fig(fig, "demographic_intersectionality")

    return results


# ─────────────────────────────────────────────
# 2. COMORBIDITY NETWORK (chord-style bar pairs)
# ─────────────────────────────────────────────

def compute_comorbidity_patterns(brfss_df):
    """
    Compute co-occurrence rates between disease pairs and risk factors.
    Which combinations cluster together?
    """
    results = {}
    records = []

    available = [c for c in DISEASE_COLS if c in brfss_df.columns]
    if len(available) < 2:
        return results

    # Pairwise disease co-occurrence
    pair_records = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            d1, d2 = available[i], available[j]
            sub = brfss_df[[d1, d2]].dropna()
            if len(sub) < 50:
                continue
            s1 = sub[d1].astype(bool)
            s2 = sub[d2].astype(bool)
            both = float((s1 & s2).mean() * 100)
            d1_only = float((s1 & ~s2).mean() * 100)
            d2_only = float((~s1 & s2).mean() * 100)
            neither = float((~s1 & ~s2).mean() * 100)
            # Lift: how much more common is the pair vs expected?
            p1 = sub[d1].mean()
            p2 = sub[d2].mean()
            expected = float(p1 * p2 * 100)
            lift = round(both / expected, 3) if expected > 0 else None
            pair_records.append({
                "disease_1": DISEASE_LABELS[d1],
                "disease_2": DISEASE_LABELS[d2],
                "co_prevalence_pct": round(both, 2),
                "d1_only_pct": round(d1_only, 2),
                "d2_only_pct": round(d2_only, 2),
                "neither_pct": round(neither, 2),
                "lift": lift,
            })

    # Risk factor co-occurrence with disease
    risk_disease_records = []
    for disease_col in available:
        label = DISEASE_LABELS[disease_col]
        disease_pos = brfss_df[brfss_df[disease_col] == 1]
        disease_neg = brfss_df[brfss_df[disease_col] == 0]

        for rf in RISK_FACTORS:
            if rf not in brfss_df.columns:
                continue
            try:
                pos_mean = float(disease_pos[rf].mean())
                neg_mean = float(disease_neg[rf].mean())
                ratio = round(pos_mean / neg_mean, 3) if neg_mean > 0 else None
                risk_disease_records.append({
                    "disease": label,
                    "risk_factor": rf,
                    "mean_with_disease": round(pos_mean, 3),
                    "mean_without_disease": round(neg_mean, 3),
                    "ratio": ratio,
                })
            except Exception:
                continue

    records = {"disease_pairs": pair_records, "risk_disease": risk_disease_records}
    results["comorbidity_patterns"] = records

    # Plot: stacked bar of co-occurrence vs solo occurrence per pair
    if pair_records:
        labels = [f"{r['disease_1']} +\n{r['disease_2']}" for r in pair_records]
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Both",
            x=labels,
            y=[r["co_prevalence_pct"] for r in pair_records],
            marker_color="#c0392b",
            text=[f"Lift {r['lift']}×" for r in pair_records],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            name="D1 Only",
            x=labels,
            y=[r["d1_only_pct"] for r in pair_records],
            marker_color="#e67e22",
        ))
        fig.add_trace(go.Bar(
            name="D2 Only",
            x=labels,
            y=[r["d2_only_pct"] for r in pair_records],
            marker_color="#3498db",
        ))

        # Risk factor ratio subplot
        if risk_disease_records:
            top_rf = sorted(
                [r for r in risk_disease_records if r["ratio"] is not None],
                key=lambda x: abs(x["ratio"] - 1),
                reverse=True,
            )[:9]

            fig2 = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "Disease Co-occurrence & Lift",
                    "Risk Factor Mean Ratio (with vs without disease)",
                ],
                column_widths=[0.4, 0.6],
            )

            for trace in [
                go.Bar(
                    name="Both",
                    x=labels,
                    y=[r["co_prevalence_pct"] for r in pair_records],
                    marker_color="#c0392b",
                    text=[f"Lift {r['lift']}×" for r in pair_records],
                    textposition="inside",
                ),
                go.Bar(
                    name="D1 Only",
                    x=labels,
                    y=[r["d1_only_pct"] for r in pair_records],
                    marker_color="#e67e22",
                ),
                go.Bar(
                    name="D2 Only",
                    x=labels,
                    y=[r["d2_only_pct"] for r in pair_records],
                    marker_color="#3498db",
                ),
            ]:
                fig2.add_trace(trace, row=1, col=1)

            rf_names = [f"{r['risk_factor']}\n({r['disease']})" for r in top_rf]
            ratios = [r["ratio"] for r in top_rf]
            colors = ["#e74c3c" if r > 1 else "#2ecc71" for r in ratios]

            fig2.add_trace(
                go.Bar(
                    x=rf_names,
                    y=ratios,
                    marker_color=colors,
                    showlegend=False,
                    text=[f"{r:.2f}×" for r in ratios],
                    textposition="outside",
                ),
                row=1, col=2,
            )
            fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
            fig2.update_layout(
                title_text="Comorbidity Patterns: Disease Pairs & Risk Factor Ratios",
                barmode="stack",
                height=520,
            )
            save_fig(fig2, "comorbidity_patterns")
        else:
            fig.update_layout(
                title_text="Disease Co-occurrence Patterns",
                barmode="stack",
                height=450,
            )
            save_fig(fig, "comorbidity_patterns")

    return results


# ─────────────────────────────────────────────
# 3. HEALTH TRAJECTORY — GENERAL HEALTH × AGE
# ─────────────────────────────────────────────

def compute_health_trajectory(brfss_df):
    """
    How does self-reported general health and disease burden evolve across age groups?
    Shows the deterioration curve and inflection points.
    """
    results = {}

    if "age_group" not in brfss_df.columns:
        return results

    ages = sorted(brfss_df["age_group"].dropna().unique())
    trajectory_records = []

    for age in ages:
        subset = brfss_df[brfss_df["age_group"] == age]
        if len(subset) < 30:
            continue

        rec = {"age_group": int(age), "n": int(len(subset))}

        if "general_health" in subset.columns:
            rec["mean_general_health"] = round(float(subset["general_health"].mean()), 3)

        for disease_col in DISEASE_COLS:
            if disease_col in subset.columns:
                label = DISEASE_LABELS[disease_col]
                rec[f"prevalence_{label}"] = round(
                    float(subset[disease_col].mean() * 100), 2
                )

        for rf in ["bmi_category", "physical_activity", "sleep_hours", "smoking_status"]:
            if rf in subset.columns:
                rec[f"mean_{rf}"] = round(float(subset[rf].mean()), 3)

        trajectory_records.append(rec)

    results["health_trajectory"] = trajectory_records

    if trajectory_records:
        df_traj = pd.DataFrame(trajectory_records)
        age_x = df_traj["age_group"].tolist()

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Disease Prevalence by Age",
                "Self-Reported General Health by Age",
                "Mean Exercise Days by Age",
                "Mean BMI Category by Age",
            ],
        )

        disease_colors = {
            "Diabetes": "#e74c3c",
            "Heart Disease": "#3498db",
            "COPD": "#2ecc71",
        }

        # Row 1 Col 1 — disease prevalence
        for disease_col in DISEASE_COLS:
            label = DISEASE_LABELS[disease_col]
            col_name = f"prevalence_{label}"
            if col_name in df_traj.columns:
                fig.add_trace(
                    go.Scatter(
                        x=age_x,
                        y=df_traj[col_name].tolist(),
                        mode="lines+markers",
                        name=label,
                        line=dict(color=disease_colors[label], width=2),
                    ),
                    row=1, col=1,
                )

        # Row 1 Col 2 — general health
        if "mean_general_health" in df_traj.columns:
            fig.add_trace(
                go.Scatter(
                    x=age_x,
                    y=df_traj["mean_general_health"].tolist(),
                    mode="lines+markers",
                    name="General Health",
                    line=dict(color="#9b59b6", width=2),
                    showlegend=False,
                ),
                row=1, col=2,
            )

        # Row 2 Col 1 — exercise
        if "mean_physical_activity" in df_traj.columns:
            fig.add_trace(
                go.Scatter(
                    x=age_x,
                    y=df_traj["mean_physical_activity"].tolist(),
                    mode="lines+markers",
                    name="Exercise Days",
                    line=dict(color="#27ae60", width=2),
                    showlegend=False,
                ),
                row=2, col=1,
            )

        # Row 2 Col 2 — BMI
        if "mean_bmi_category" in df_traj.columns:
            fig.add_trace(
                go.Scatter(
                    x=age_x,
                    y=df_traj["mean_bmi_category"].tolist(),
                    mode="lines+markers",
                    name="BMI Category",
                    line=dict(color="#e67e22", width=2),
                    showlegend=False,
                ),
                row=2, col=2,
            )

        fig.update_layout(
            title_text="Health Trajectory Across Age Groups",
            height=620,
        )
        save_fig(fig, "health_trajectory")

    return results


# ─────────────────────────────────────────────
# 4. RISK FACTOR INTERACTION HEATMAP
# ─────────────────────────────────────────────

def compute_risk_factor_interactions(brfss_df):
    """
    Spearman correlation matrix across all risk factors and disease indicators.
    Reveals which risk factors co-occur and which are independent signals.
    """
    results = {}

    from scipy import stats

    all_cols = [c for c in RISK_FACTORS + DISEASE_COLS if c in brfss_df.columns]
    if len(all_cols) < 4:
        return results

    n = len(all_cols)
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.ones((n, n))

    sub = brfss_df[all_cols].dropna()
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif j > i:
                try:
                    r, p = stats.spearmanr(sub.iloc[:, i], sub.iloc[:, j])
                    corr_matrix[i, j] = round(float(r), 4)
                    corr_matrix[j, i] = round(float(r), 4)
                    pval_matrix[i, j] = round(float(p), 4)
                    pval_matrix[j, i] = round(float(p), 4)
                except Exception:
                    pass

    labels = []
    for c in all_cols:
        if c in DISEASE_LABELS:
            labels.append(DISEASE_LABELS[c])
        else:
            labels.append(c.replace("_", " ").title())

    corr_records = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_records.append({
                "var_a": labels[i],
                "var_b": labels[j],
                "spearman_r": float(corr_matrix[i, j]),
                "p_value": float(pval_matrix[i, j]),
                "significant": bool(pval_matrix[i, j] < 0.05),
            })

    results["risk_factor_interactions"] = {
        "variables": labels,
        "correlation_matrix": corr_matrix.tolist(),
        "pairwise_records": corr_records,
    }

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.tolist(),
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{corr_matrix[i][j]:.2f}" for j in range(n)] for i in range(n)],
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Spearman r"),
        )
    )

    # Significance overlay — mark non-significant cells
    sig_x, sig_y = [], []
    for i in range(n):
        for j in range(n):
            if i != j and pval_matrix[i, j] >= 0.05:
                sig_x.append(labels[j])
                sig_y.append(labels[i])

    if sig_x:
        fig.add_trace(
            go.Scatter(
                x=sig_x,
                y=sig_y,
                mode="markers",
                marker=dict(symbol="x", size=10, color="gray", opacity=0.5),
                name="p≥0.05 (NS)",
                showlegend=True,
            )
        )

    fig.update_layout(
        title="Risk Factor × Disease Spearman Correlation Matrix (✕ = non-significant)",
        height=560,
        xaxis=dict(tickangle=45),
    )
    save_fig(fig, "risk_factor_interactions")

    return results


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_advanced_eda(brfss_df):
    print("\n=== Advanced EDA Module ===")
    np.random.seed(42)
    all_results = {}

    print("  [1/4] Demographic intersectionality heatmap (age × income)...")
    all_results["intersectionality"] = compute_demographic_intersectionality(brfss_df)

    print("  [2/4] Comorbidity patterns & risk factor ratios...")
    all_results["comorbidity"] = compute_comorbidity_patterns(brfss_df)

    print("  [3/4] Health trajectory across age groups...")
    all_results["trajectory"] = compute_health_trajectory(brfss_df)

    print("  [4/4] Risk factor interaction correlation matrix...")
    all_results["interactions"] = compute_risk_factor_interactions(brfss_df)

    output_path = OUTPUT_DIR / "advanced_eda.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"  Results saved: {output_path}")
    print(f"  Plots: images/eda_demographic_intersectionality.png,")
    print(f"         eda_comorbidity_patterns.png,")
    print(f"         eda_health_trajectory.png,")
    print(f"         eda_risk_factor_interactions.png")
    return all_results
