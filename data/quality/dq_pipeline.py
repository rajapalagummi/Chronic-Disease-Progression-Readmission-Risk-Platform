import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from jinja2 import Template
from datetime import datetime

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRFSS_RULES = [
    ("sleep_hours", "sleep_hours >= 3 AND sleep_hours <= 12", "Sleep hours 3-12"),
    ("bmi_category", "bmi_category IN (1,2,3,4)", "BMI category 1-4"),
    ("age_group", "age_group >= 1 AND age_group <= 13", "Age group 1-13"),
    ("diabetes_status", "diabetes_status IN (0,1)", "Diabetes binary"),
    ("heart_disease_status", "heart_disease_status IN (0,1)", "Heart disease binary"),
    ("copd_status", "copd_status IN (0,1)", "COPD binary"),
    ("bmi_category", "bmi_category IS NOT NULL", "BMI not null"),
    ("physical_activity", "physical_activity IN (1,2)", "Physical activity 1-2"),
    ("smoking_status", "smoking_status IN (1,2)", "Smoking binary"),
]

CMS_RULES = [
    ("readmission_rate_30day", "readmission_rate_30day >= 0.05 AND readmission_rate_30day <= 0.50", "Readmission rate 5-50%"),
    ("quality_tier", "quality_tier >= 1 AND quality_tier <= 5", "Quality tier 1-5"),
    ("n_cases", "n_cases >= 1", "Case count positive"),
    ("disease", "disease IN ('diabetes','heart_failure','copd','heart_attack')", "Valid disease"),
]


def run_dq_checks(con, table, rules):
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    results = []
    for col, rule, description in rules:
        try:
            passing = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {rule}").fetchone()[0]
        except Exception:
            passing = 0
        failing = total - passing
        pass_rate = passing / total * 100 if total > 0 else 0
        results.append({
            "table": table, "column": col, "description": description,
            "total_rows": total, "passing_rows": passing,
            "failing_rows": failing, "pass_rate_pct": round(pass_rate, 2),
            "status": "PASS" if pass_rate >= 95 else "WARN" if pass_rate >= 80 else "FAIL",
        })

    for col in con.execute(f"PRAGMA table_info({table})").df()["name"].tolist():
        try:
            null_count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
            null_pct = null_count / total * 100
            results.append({
                "table": table, "column": col, "description": f"Null check: {col}",
                "total_rows": total, "passing_rows": total - null_count,
                "failing_rows": null_count, "pass_rate_pct": round(100 - null_pct, 2),
                "status": "PASS" if null_pct < 5 else "WARN" if null_pct < 20 else "FAIL",
            })
        except Exception:
            continue
    return pd.DataFrame(results)


def check_volume_drift(con):
    try:
        by_year = con.execute("""
            SELECT survey_year, COUNT(*) as n_records
            FROM brfss GROUP BY survey_year ORDER BY survey_year
        """).df()
        if len(by_year) > 1:
            by_year["pct_change"] = by_year["n_records"].pct_change() * 100
            by_year["volume_alert"] = by_year["pct_change"].abs() > 20
        return by_year
    except Exception:
        return pd.DataFrame()


def generate_html_report(brfss_results, cms_results, volume_drift):
    template_str = """
<!DOCTYPE html><html><head><title>HealthPath DQ Report</title>
<style>
body{font-family:Arial,sans-serif;margin:40px;background:#f5f5f5}
h1{color:#2c3e50}h2{color:#34495e;margin-top:30px}
table{border-collapse:collapse;width:100%;background:white;margin-bottom:20px}
th{background:#2c3e50;color:white;padding:10px;text-align:left}
td{padding:8px 10px;border-bottom:1px solid #ddd}
tr:hover{background:#f0f0f0}
.PASS{color:#27ae60;font-weight:bold}.WARN{color:#f39c12;font-weight:bold}.FAIL{color:#e74c3c;font-weight:bold}
.summary{background:white;padding:20px;border-radius:5px;margin-bottom:20px}
.metric{display:inline-block;margin:10px 20px;text-align:center}
.metric-value{font-size:28px;font-weight:bold;color:#2c3e50}
.metric-label{font-size:12px;color:#888}
</style></head><body>
<h1>HealthPath — Data Quality Report</h1>
<p>Generated: {{ timestamp }}</p>
<div class="summary">
<div class="metric"><div class="metric-value">{{ total_checks }}</div><div class="metric-label">Total Checks</div></div>
<div class="metric"><div class="metric-value" style="color:#27ae60">{{ passing }}</div><div class="metric-label">Passing</div></div>
<div class="metric"><div class="metric-value" style="color:#f39c12">{{ warnings }}</div><div class="metric-label">Warnings</div></div>
<div class="metric"><div class="metric-value" style="color:#e74c3c">{{ failing }}</div><div class="metric-label">Failing</div></div>
</div>
<h2>BRFSS 2022 Data Quality</h2>
<table><tr><th>Column</th><th>Check</th><th>Total</th><th>Failing</th><th>Pass Rate</th><th>Status</th></tr>
{% for _, row in brfss_results.iterrows() %}
<tr><td>{{ row.column }}</td><td>{{ row.description }}</td><td>{{ row.total_rows }}</td>
<td>{{ row.failing_rows }}</td><td>{{ row.pass_rate_pct }}%</td>
<td class="{{ row.status }}">{{ row.status }}</td></tr>
{% endfor %}</table>
<h2>CMS Hospital Compare Data Quality</h2>
<table><tr><th>Column</th><th>Check</th><th>Total</th><th>Failing</th><th>Pass Rate</th><th>Status</th></tr>
{% for _, row in cms_results.iterrows() %}
<tr><td>{{ row.column }}</td><td>{{ row.description }}</td><td>{{ row.total_rows }}</td>
<td>{{ row.failing_rows }}</td><td>{{ row.pass_rate_pct }}%</td>
<td class="{{ row.status }}">{{ row.status }}</td></tr>
{% endfor %}</table>
{% if volume_rows %}
<h2>Volume by Survey Year</h2>
<table><tr><th>Year</th><th>Records</th><th>Change %</th><th>Alert</th></tr>
{% for row in volume_rows %}
<tr><td>{{ row.year }}</td><td>{{ row.n }}</td><td>{{ row.pct }}</td><td>{{ row.alert }}</td></tr>
{% endfor %}</table>{% endif %}
</body></html>"""
    all_results = pd.concat([brfss_results, cms_results])

    volume_rows = []
    if not volume_drift.empty and "pct_change" in volume_drift.columns:
        for _, r in volume_drift.iterrows():
            pct = r["pct_change"]
            volume_rows.append(type("R", (), {
                "year": int(r["survey_year"]),
                "n": int(r["n_records"]),
                "pct": f"{float(pct):.1f}" if pct == pct else "N/A",
                "alert": "YES" if r.get("volume_alert", False) else "NO",
            })())

    html = Template(template_str).render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_checks=len(all_results),
        passing=len(all_results[all_results["status"] == "PASS"]),
        warnings=len(all_results[all_results["status"] == "WARN"]),
        failing=len(all_results[all_results["status"] == "FAIL"]),
        brfss_results=brfss_results, cms_results=cms_results, volume_rows=volume_rows,
    )
    path = OUTPUT_DIR / "data_quality_report.html"
    path.write_text(html)
    print(f"Data quality report: {path}")
    return html


def run_dq_pipeline(brfss_df, cms_df):
    con = duckdb.connect()
    con.execute("CREATE TABLE brfss AS SELECT * FROM brfss_df")
    con.execute("CREATE TABLE cms AS SELECT * FROM cms_df")

    print("\n=== Data Quality Pipeline ===")
    brfss_results = run_dq_checks(con, "brfss", BRFSS_RULES)
    cms_results = run_dq_checks(con, "cms", CMS_RULES)
    volume_drift = check_volume_drift(con)

    brfss_pass = len(brfss_results[brfss_results["status"] == "PASS"])
    cms_pass = len(cms_results[cms_results["status"] == "PASS"])
    total = len(brfss_results) + len(cms_results)
    print(f"BRFSS: {brfss_pass}/{len(brfss_results)} checks passing")
    print(f"CMS: {cms_pass}/{len(cms_results)} checks passing")
    print(f"Overall pass rate: {(brfss_pass + cms_pass) / total * 100:.1f}%")

    generate_html_report(brfss_results, cms_results, volume_drift)
    con.close()
    return brfss_results, cms_results
