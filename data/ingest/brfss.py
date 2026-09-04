import pyreadstat
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

COL_MAP = {
    "DIABETE4": "diabetes_raw",
    "CVDCRHD4": "heart_disease_raw",
    "CHCCOPD3": "copd_raw",
    "CVDINFR4": "heart_attack_history",
    "CVDSTRK3": "stroke_history",
    "CHCKDNY2": "kidney_disease",
    "ADDEPEV3": "depression",
    "HAVARTH4": "arthritis",
    "_BMI5CAT": "bmi_category",
    "_TOTINDA": "physical_activity",
    "SMOKE100": "smoking_status",
    "ALCDAY4": "alcohol_raw",
    "SLEPTIM1": "sleep_hours",
    "MEDCOST1": "medical_cost_barrier",
    "_INCOMG1": "income_category",
    "_EDUCAG": "education_category",
    "_AGEG5YR": "age_group",
    "_SEX": "sex",
    "_STATE": "state",
    "CHECKUP1": "last_checkup",
    "EXERANY2": "exercise_any",
    "_HLTHPLN": "health_plan",
    "GENHLTH": "general_health",
    "PHYSHLTH": "physical_health_days",
    "MENTHLTH": "mental_health_days",
    "INCOME3": "income_raw",
    "EDUCA": "education_raw",
}

DISEASE_BINARY = {
    "diabetes_raw": ("diabetes_status", [1, 2]),
    "heart_disease_raw": ("heart_disease_status", [1]),
    "copd_raw": ("copd_status", [1]),
}

REFUSAL_CODES = [7, 8, 9, 77, 88, 99, 777, 888, 999]


def clean_brfss(df):
    df = df.rename(columns=COL_MAP)

    for raw_col, (clean_col, positive_vals) in DISEASE_BINARY.items():
        if raw_col in df.columns:
            df[clean_col] = df[raw_col].apply(
                lambda x: 1 if x in positive_vals else (0 if x == 2 else np.nan)
            )

    df["hypertension_status"] = np.nan

    df["diabetes_status"] = df["diabetes_raw"].apply(
        lambda x: 1 if x in [1, 2] else (0 if x == 3 else np.nan)
    )

    for col in ["sleep_hours", "physical_health_days", "mental_health_days"]:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(REFUSAL_CODES), np.nan)
            df[col] = df[col].where(df[col] <= 30, np.nan)

    if "alcohol_raw" in df.columns:
        def parse_alcohol(x):
            if pd.isna(x) or x in [888, 777, 999]:
                return 0
            x = int(x)
            if x == 888:
                return 0
            elif x >= 200:
                return (x - 200) / 4.33
            elif x >= 100:
                return x - 100
            return np.nan
        df["alcohol_days"] = df["alcohol_raw"].apply(parse_alcohol)

    for col in ["bmi_category", "physical_activity", "smoking_status",
                "medical_cost_barrier", "health_plan", "exercise_any",
                "last_checkup", "general_health", "heart_attack_history",
                "stroke_history", "kidney_disease", "depression", "arthritis"]:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(REFUSAL_CODES), np.nan)

    keep = [
        "diabetes_status", "heart_disease_status", "copd_status",
        "bmi_category", "physical_activity", "smoking_status", "alcohol_days",
        "sleep_hours", "income_category", "education_category", "age_group",
        "sex", "state", "last_checkup", "exercise_any", "health_plan",
        "general_health", "physical_health_days", "mental_health_days",
        "heart_attack_history", "stroke_history", "kidney_disease",
        "depression", "arthritis",
    ]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()
    df = df.dropna(subset=["diabetes_status", "heart_disease_status", "copd_status"])
    df = df.reset_index(drop=True)

    return df


def load_brfss():
    cache = RAW_DIR / "brfss_clean.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"BRFSS loaded from cache: {len(df):,} rows, {df.shape[1]} columns")
        return df

    xpt_files = [f for f in RAW_DIR.iterdir() if f.suffix.upper() == ".XPT"]
    if not xpt_files:
        raise FileNotFoundError(f"No XPT file found in {RAW_DIR}. Place LLCP2022.XPT there.")

    xpt_path = xpt_files[0]
    print(f"Reading real BRFSS data from {xpt_path.name} ({xpt_path.stat().st_size/1e9:.2f} GB)...")
    raw, meta = pyreadstat.read_xport(str(xpt_path), encoding="latin1", row_limit=50000)
    print(f"Raw shape: {raw.shape}")

    df = clean_brfss(raw)
    df["survey_year"] = 2022
    df.to_parquet(cache, index=False)
    print(f"BRFSS cleaned and cached: {len(df):,} rows, {df.shape[1]} columns")
    return df
