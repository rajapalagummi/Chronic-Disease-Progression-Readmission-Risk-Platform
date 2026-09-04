# HealthPath — Data Dictionary

## Disease Outcome Definitions

| Metric | Source Column | Positive Code | Definition |
|---|---|---|---|
| diabetes_status | DIABETE4 | 1 or 2 | Ever told had diabetes or pre-diabetes |
| heart_disease_status | CVDCRHD4 | 1 | Ever told had coronary heart disease |
| copd_status | CHCCOPD3 | 1 | Ever told had COPD or emphysema |

## Lifestyle & Behavioral Features

| Feature | Source Column | Values | Notes |
|---|---|---|---|
| bmi_category | _BMI5CAT | 1=Underweight, 2=Normal, 3=Overweight, 4=Obese | Computed from self-reported height/weight |
| physical_activity | _TOTINDA | 1=Active, 2=Inactive | Any physical activity in past 30 days |
| smoking_status | SMOKE100 | 1=Smoker, 2=Non-Smoker | Smoked ≥100 cigarettes lifetime |
| alcohol_days | ALCDAY4 | Continuous: 0-7 avg days/week | Parsed from BRFSS coded format |
| sleep_hours | SLEPTIM1 | Continuous: 1-24 hours | Hours sleep per 24-hour period |
| exercise_any | EXERANY2 | 1=Yes, 2=No | Any physical activity past 30 days |
| last_checkup | CHECKUP1 | 1=<1yr, 2=1-2yr, 3=2-5yr, 4=5+yr | Time since last routine checkup |
| general_health | GENHLTH | 1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor | Self-reported general health |

## Socioeconomic Features

| Feature | Source Column | Values |
|---|---|---|
| income_category | _INCOMG1 | 1=<$15K, 2=$15-25K, 3=$25-35K, 4=$35-50K, 5=≥$50K |
| education_category | _EDUCAG | 1=No HS, 2=HS grad, 3=Some college, 4=College grad |
| age_group | _AGEG5YR | 1=18-24, 2=25-29, ..., 13=80+ |
| sex | _SEX | 1=Male, 2=Female |
| health_plan | _HLTHPLN | 1=Yes, 2=No |
| medical_cost_barrier | MEDCOST1 | 1=Yes (could not see doctor due to cost), 2=No |

## Hospital & Readmission Metrics

| Metric | Source | Definition |
|---|---|---|
| readmission_rate_30day | CMS Hospital Compare | Proportion readmitted within 30 days of discharge |
| high_readmit_{disease} | Derived | 1 if hospital-state rate above 50th percentile |
| quality_tier | CMS Star Ratings | 1=Worst, 5=Best overall hospital rating |

## Derived Analytical Metrics

| Metric | Definition | Formula |
|---|---|---|
| Sedentary | Zero physical activity in past 30 days | physical_activity == 2 |
| High Readmission Risk | Above population median readmission rate | readmission_rate > quantile(0.50) |
| 30-Day Readmission Window | Calendar days from discharge to readmission | discharge_date + 30 days |
| DiD Estimate | Causal effect of exercise treatment | (treat_post - treat_pre) - (ctrl_post - ctrl_pre) |

## Data Quality Thresholds

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Rule compliance | ≥95% | 80-95% | <80% |
| Null rate | <5% | 5-20% | >20% |

## BRFSS Refusal/Unknown Code Handling

| Code | Meaning | Treatment |
|---|---|---|
| 7, 77, 777 | Don't know/Not sure | Replaced with NaN, imputed with median |
| 9, 99, 999 | Refused | Replaced with NaN, imputed with median |
| 888 | None/Never (alcohol) | Recoded as 0 days |
| BLANK | Missing/Not applicable | Imputed with column median |
