import json
import mlflow
from pathlib import Path

from data.ingest.brfss import load_brfss
from data.ingest.cms import load_cms
from data.quality.dq_pipeline import run_dq_pipeline
from eda.brfss_eda import run_brfss_eda
from eda.cms_eda import run_cms_eda
from analysis.causal_survival import run_survival_analysis, run_causal_analysis
from models.progression import run_progression_models
from models.readmission import run_readmission_models

Path("data/outputs").mkdir(parents=True, exist_ok=True)
Path("images").mkdir(exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlruns.db")

print("=== HealthPath — Chronic Disease Progression & Readmission Intelligence ===")
print("Using real BRFSS 2022 and CMS Hospital Compare data\n")

print("[1/7] Loading BRFSS 2022 data...")
brfss_df = load_brfss()
print(f"BRFSS loaded: {len(brfss_df):,} respondents, {brfss_df.shape[1]} features")

print("\n[2/7] Loading CMS Hospital Compare data...")
cms_df = load_cms()
print(f"CMS loaded: {cms_df['hospital_id'].nunique():,} hospitals, {len(cms_df):,} rows")

print("\n[3/7] Data quality pipeline...")
run_dq_pipeline(brfss_df, cms_df)

print("\n[4/7] BRFSS EDA...")
run_brfss_eda(brfss_df)

print("\n[5/7] CMS EDA...")
run_cms_eda(cms_df)

print("\n[6/7] Survival and causal analysis...")
survival_results = run_survival_analysis(brfss_df)
causal_results, did_df = run_causal_analysis(brfss_df)

print("\n[7/7] Progression and readmission models...")
progression_results = run_progression_models(brfss_df)
readmission_results = run_readmission_models(brfss_df, cms_df)

from analysis.statistical import run_statistical_analysis
run_statistical_analysis(brfss_df, cms_df, progression_results, readmission_results, causal_results)

from analysis.factor_decay import run_factor_decay_analysis
run_factor_decay_analysis(brfss_df, progression_results, readmission_results)

from analysis.advanced_eda import run_advanced_eda
print(brfss_df.columns.tolist())
run_advanced_eda(brfss_df)


DISEASE_LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD",
}

results = {
    "progression": {
        "cv_results": progression_results["cv_results"],
        "shap_importance": progression_results["shap_importance"],
        "sensitivity": progression_results["sensitivity"],
    },
    "causal": {
        disease: {"did": r["did"], "placebo": r["placebo"]}
        for disease, r in causal_results.items()
    },
    "survival": survival_results,
    "readmission": {
        "cv_results": readmission_results["cv_results"],
        "sensitivity": readmission_results["sensitivity"],
    },
}

with open("data/outputs/model_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("=== FINAL NUMBERS ===")
print("=" * 60)

print("\nData:")
print(f"  BRFSS respondents: {len(brfss_df):,}")
print(f"  CMS hospitals: {cms_df['hospital_id'].nunique():,}")

print("\nDisease Prevalence (real BRFSS 2022):")
for col in ["diabetes_status", "heart_disease_status", "copd_status"]:
    if col in brfss_df.columns:
        print(f"  {DISEASE_LABELS[col]}: {brfss_df[col].mean()*100:.1f}%")

print("\nDisease Progression AUC:")
for disease, stats in progression_results["cv_results"].items():
    print(f"  {DISEASE_LABELS.get(disease, disease)}: {stats['mean_auc']} ± {stats['std_auc']}")

print("\nCausal DiD Results (Exercise Intervention):")
for disease, r in causal_results.items():
    did = r["did"]
    print(f"  {DISEASE_LABELS.get(disease, disease)}: DiD={did['did_estimate']}, p={did['p_value']}, sig={did['significant']}")

print("\nReadmission Model AUC:")
for disease, stats in readmission_results["cv_results"].items():
    print(f"  {disease}: {stats['mean_auc']} ± {stats['std_auc']}")

print("\nOutputs:")
print("  data/outputs/data_quality_report.html")
print("  data/outputs/model_results.json")
print("  images/ — all visualizations")
print("\nRun: streamlit run dashboard/app.py")
print("Run: python score_patient.py --age_group 7 --bmi_category 3")
