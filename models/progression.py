import pandas as pd
import numpy as np
import shap
import mlflow
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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
FEATURE_COLS = [
    "age_group", "sex", "bmi_category", "physical_activity", "smoking_status",
    "alcohol_days", "sleep_hours", "income_category", "education_category",
    "health_plan", "medical_cost_barrier", "exercise_any", "last_checkup",
    "general_health", "physical_health_days", "mental_health_days",
    "heart_attack_history", "stroke_history", "kidney_disease", "depression", "arthritis",
]
MODIFIABLE = ["bmi_category", "physical_activity", "smoking_status",
               "sleep_hours", "exercise_any", "last_checkup"]


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        fig.write_image(str(IMAGES_DIR / f"{name}.png"), width=1200, height=600)
    except Exception:
        pass


def prepare_features(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    return X.values, available


def train_xgb(X, y):
    pos = max(y.sum(), 1)
    neg = len(y) - pos
    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        scale_pos_weight=neg / pos, random_state=42,
        eval_metric="auc", verbosity=0,
    )
    model.fit(X, y)
    return model


def cross_validate_disease(df):
    X, features = prepare_features(df)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for disease in DISEASES:
        if disease not in df.columns:
            continue
        y = df[disease].fillna(0).astype(int).values
        if y.sum() < 20:
            continue
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            model = train_xgb(X[train_idx], y[train_idx])
            pred = model.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(y[test_idx])) > 1:
                aucs.append(roc_auc_score(y[test_idx], pred))
        if aucs:
            results[disease] = {
                "mean_auc": round(np.mean(aucs), 4),
                "std_auc": round(np.std(aucs), 4),
                "prevalence": round(float(y.mean()), 4),
                "label": DISEASE_LABELS[disease],
            }
            print(f"  {DISEASE_LABELS[disease]}: AUC={results[disease]['mean_auc']} ± {results[disease]['std_auc']}")
    return results, features


def train_all_models(df):
    X, features = prepare_features(df)
    models = {}
    shap_importance = {}

    for disease in DISEASES:
        if disease not in df.columns:
            continue
        y = df[disease].fillna(0).astype(int).values
        if y.sum() < 20:
            continue
        model = train_xgb(X, y)
        models[disease] = model

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X[:2000])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_importance[disease] = dict(zip(features, [round(float(v), 4) for v in mean_abs]))

    return models, shap_importance, features, X


def compute_sensitivity(models, df, features):
    X_base, _ = prepare_features(df)
    sensitivity = {}
    for disease, model in models.items():
        base_prob = model.predict_proba(X_base)[:, 1].mean()
        disease_sens = {"baseline": round(float(base_prob), 4)}
        for factor in MODIFIABLE:
            if factor not in features:
                continue
            X_mod = X_base.copy()
            idx = features.index(factor)
            col_vals = df[factor].dropna()
            X_mod[:, idx] = col_vals.quantile(0.25)
            mod_prob = model.predict_proba(X_mod)[:, 1].mean()
            disease_sens[factor] = round(float(mod_prob - base_prob), 4)
        sensitivity[DISEASE_LABELS[disease]] = disease_sens
    return sensitivity


def plot_results(cv_results, shap_importance):
    auc_data = pd.DataFrame([
        {"Disease": v["label"], "AUC": v["mean_auc"], "Std": v["std_auc"]}
        for v in cv_results.values()
    ]).sort_values("AUC", ascending=False)

    fig = px.bar(auc_data, x="Disease", y="AUC", error_y="Std",
                 title="Cross-Disease Prediction AUC (Same Feature Set — Real BRFSS 2022)",
                 color="AUC", color_continuous_scale="Blues", text=auc_data["AUC"].round(4).values)
    fig.update_traces(textposition="outside")
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="Random baseline")
    save_fig(fig, "model_cross_disease_auc")

    shap_rows = []
    for disease, importance in shap_importance.items():
        for feat, val in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]:
            shap_rows.append({"Disease": DISEASE_LABELS.get(disease, disease), "Feature": feat, "SHAP": val})
    shap_df = pd.DataFrame(shap_rows)
    fig = px.bar(shap_df, x="SHAP", y="Feature", color="Disease",
                 facet_col="Disease", orientation="h",
                 title="Top SHAP Feature Importance by Disease", height=700)
    save_fig(fig, "model_shap_by_disease")


def run_progression_models(brfss_df):
    mlflow.set_experiment("healthpath_progression")
    print("\n=== Module 3: Cross-Disease Progression Models ===")

    with mlflow.start_run(run_name="cross_disease_xgboost"):
        cv_results, features = cross_validate_disease(brfss_df)
        models, shap_importance, features, X = train_all_models(brfss_df)

        for disease, result in cv_results.items():
            mlflow.log_metric(f"{DISEASE_LABELS[disease]}_auc", result["mean_auc"])

        plot_results(cv_results, shap_importance)
        sensitivity = compute_sensitivity(models, brfss_df, features)

    return {
        "cv_results": cv_results,
        "models": models,
        "shap_importance": shap_importance,
        "sensitivity": sensitivity,
        "features": features,
        "X": X,
    }
