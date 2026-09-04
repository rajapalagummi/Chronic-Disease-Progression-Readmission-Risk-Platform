import argparse
import json
from pathlib import Path

RESULTS_PATH = Path("data/outputs/model_results.json")
DISEASE_LABELS = {
    "diabetes_status": "Diabetes",
    "heart_disease_status": "Heart Disease",
    "copd_status": "COPD",
}


def main():
    parser = argparse.ArgumentParser(description="HealthPath Patient Risk Scorer")
    parser.add_argument("--age_group", type=int, default=7, help="Age group 1=18-24, 13=80+")
    parser.add_argument("--bmi_category", type=int, default=3, help="BMI: 1=Underweight, 2=Normal, 3=Overweight, 4=Obese")
    parser.add_argument("--smoking_status", type=int, default=2, help="1=Smoker, 2=Non-Smoker")
    parser.add_argument("--physical_activity", type=int, default=1, help="1=Active, 2=Inactive")
    parser.add_argument("--sleep_hours", type=float, default=7.0, help="Hours of sleep per night")
    parser.add_argument("--income_category", type=int, default=4, help="Income 1=Low, 9=High")
    args = parser.parse_args()

    print("\n=== HealthPath Patient Risk Assessment ===")
    print(f"\nPatient Profile:")
    print(f"  Age group: {args.age_group} (1=18-24, 13=80+)")
    print(f"  BMI category: {args.bmi_category} (1=Underweight, 2=Normal, 3=Overweight, 4=Obese)")
    print(f"  Smoking: {'Yes' if args.smoking_status == 1 else 'No'}")
    print(f"  Physical activity: {'Active' if args.physical_activity == 1 else 'Inactive'}")
    print(f"  Sleep: {args.sleep_hours}h/night")
    print(f"  Income category: {args.income_category}/9")

    if not RESULTS_PATH.exists():
        print("\nNo model results found. Run: python main.py")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    prog = results.get("progression", {})
    cv = prog.get("cv_results", {})

    print("\nDisease Risk Profile (from real BRFSS 2022 model):")
    print(f"{'Disease':<20} {'Model AUC':<12} {'Population Prevalence'}")
    print("-" * 50)
    for disease, stats in sorted(cv.items(), key=lambda x: x[1]["mean_auc"], reverse=True):
        label = DISEASE_LABELS.get(disease, disease)
        prev = stats.get("prevalence", 0) * 100
        print(f"{label:<20} {stats['mean_auc']:<12} {prev:.1f}%")

    causal = results.get("causal", {})
    print("\nCausal Effect of Exercise on Disease Risk (DiD estimates):")
    for disease, r in causal.items():
        did = r.get("did", {})
        label = DISEASE_LABELS.get(disease, disease)
        est = did.get("did_estimate", 0)
        p = did.get("p_value", 1)
        direction = "reduces" if est < 0 else "increases"
        print(f"  {label}: exercise {direction} risk by {abs(est)*100:.2f}pp (p={p})")

    print("\nFor full interactive scoring, run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
