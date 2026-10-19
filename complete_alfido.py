from pathlib import Path
import json, textwrap, csv

ROOT = Path.cwd()
(ROOT / "notebooks").mkdir(exist_ok=True)
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "outputs").mkdir(exist_ok=True)

def make_notebook(filename, title, cells):
    nb = {
        "cells": [
            {"cell_type": kind,
             "metadata": {},
             "source": textwrap.dedent(src).strip().splitlines(True)}
            for kind, src in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.x"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    (ROOT / "notebooks" / filename).write_text(
        json.dumps(nb, indent=2), encoding="utf-8"
    )

common = r'''
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
DATA = ROOT / "data" / "raw"
files = list(DATA.glob("*.csv")) + list(DATA.glob("*.xlsx")) + list(DATA.glob("*.json"))
print("Data files found:", [f.name for f in files])
print("Put the correct task dataset in data/raw/ before interpreting results.")

def load_data(keywords):
    matches = [f for f in files if any(k in f.name.lower() for k in keywords)]
    if not matches:
        print("DATASET MISSING for:", keywords)
        return None
    f = matches[0]
    print("Loading:", f.name)
    if f.suffix.lower() == ".csv":
        return pd.read_csv(f)
    if f.suffix.lower() == ".xlsx":
        return pd.read_excel(f)
    return pd.read_json(f)

def overview(df):
    if df is None:
        return
    print("Shape:", df.shape)
    display(df.head())
    display(df.isna().sum().sort_values(ascending=False).head(20).to_frame("missing"))
    display(df.describe(include="all").T)
'''

make_notebook("Task1_Zomato_Analysis.ipynb", "Zomato Restaurant Analysis", [
    ("markdown", "# Task 1 — Zomato Restaurant & Review Analysis\n\nAdd the official restaurant/review CSV or Excel file to `data/raw/`. Rename it to include `zomato` for automatic discovery. Verify column meanings before drawing conclusions."),
    ("code", common + "\ndf = load_data(['zomato', 'restaurant', 'review'])\noverview(df)"),
    ("markdown", "## Data quality and restaurant patterns"),
    ("code", r'''
if df is not None:
    print("Duplicate rows:", df.duplicated().sum())
    print("Columns:", list(df.columns))
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        display(numeric.corr(numeric_only=True))
        col = numeric.columns[0]
        ax = numeric[col].dropna().plot(kind="hist", bins=25, title=f"Distribution: {col}")
        plt.tight_layout()
        plt.show()
    cats = df.select_dtypes(include=["object", "category"]).columns
    for col in list(cats)[:3]:
        print("\nTop values:", col)
        display(df[col].value_counts(dropna=False).head(10))
    print("Use verified column names to analyze ratings, price, cuisine, votes and reviews.")
else:
    print("Add the task dataset, rerun this notebook, and document findings from its actual output.")
'''),
    ("markdown", "## Recommendations\n\nWrite five evidence-based recommendations after reviewing the real tables and charts. Do not claim business findings until the dataset has been analyzed.")
])

make_notebook("Task2_Loan_Approval_Prediction.ipynb", "Loan Approval Prediction", [
    ("markdown", "# Task 2 — Loan Approval Prediction\n\nPlace the official loan dataset in `data/raw/` with `loan` in its filename. Confirm the target column from the assignment before training."),
    ("code", common + "\ndf = load_data(['loan'])\noverview(df)"),
    ("markdown", "## Baseline classification model"),
    ("code", r'''
if df is not None:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    candidates = [c for c in df.columns if any(
        x in c.lower() for x in ["loan_status", "approved", "approval", "target"]
    )]
    print("Possible target columns:", candidates)
    if not candidates:
        print("TARGET NOT IDENTIFIED. Set target_col to the exact target column from the assignment.")
    else:
        target_col = candidates[0]
        work = df.dropna(subset=[target_col]).copy()
        X, y = work.drop(columns=[target_col]), work[target_col]
        if y.nunique() < 2:
            print("Target must contain at least two classes.")
        else:
            numeric_cols = X.select_dtypes(include="number").columns.tolist()
            cat_cols = X.select_dtypes(exclude="number").columns.tolist()
            pre = ColumnTransformer([
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric_cols),
                ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ])
            model = Pipeline([("preprocess", pre),
                              ("classifier", RandomForestClassifier(
                                  n_estimators=200, random_state=42,
                                  class_weight="balanced"))])
            strat = y if y.value_counts().min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=strat)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            print("Target:", target_col)
            print(classification_report(y_test, pred, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(y_test, pred))
            print("Suggested threshold must be selected using validation data, costs of false approvals/rejections, and calibration—not guessed.")
else:
    print("Dataset missing. No model metrics or threshold can be truthfully reported yet.")
'''),
    ("markdown", "## Threshold selection and limitations\n\nSelect the operating threshold using a validation set, business costs, fairness checks and calibration. Do not use the test set repeatedly to tune it. This model is a learning exercise, not a real lending decision system.")
])

make_notebook("Task3_Instagram_Engagement_Analysis.ipynb", "Instagram Engagement Analysis", [
    ("markdown", "# Task 3 — Instagram Engagement Analysis\n\nPlace the official Instagram/content-performance dataset in `data/raw/` with `instagram` or `engagement` in its filename."),
    ("code", common + "\ndf = load_data(['instagram', 'engagement', 'social'])\noverview(df)"),
    ("markdown", "## Engagement exploration"),
    ("code", r'''
if df is not None:
    cols = {c.lower().replace(" ", "_"): c for c in df.columns}
    print("Detected columns:", cols)
    for key in ["likes", "comments", "shares", "reach", "impressions", "engagement_rate"]:
        actual = cols.get(key)
        if actual:
            print(f"\nSummary for {actual}")
            display(df[actual].describe())
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in list(cat_cols)[:4]:
        print("\nBreakdown by", col)
        display(df.groupby(col, dropna=False).size().sort_values(ascending=False).head(10))
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric.hist(figsize=(10, 6), bins=20)
        plt.tight_layout()
        plt.show()
else:
    print("Dataset missing; do not invent engagement rates or best-performing post types.")
'''),
    ("markdown", "## Strategy and experiments\n\nAfter analyzing the actual results, recommend content formats, posting cadence and calls to action. Track reach, saves, shares, profile visits and link clicks. Use controlled experiments rather than promising guaranteed growth.")
])

make_notebook("Task4_Business_Sales_Analytics.ipynb", "Business Sales Analytics", [
    ("markdown", "# Task 4 — Open-Ended Business Analytics Solution\n\nProposed solution: business sales and customer analytics. Add the official Task 4 dataset to `data/raw/` with `sales`, `business` or `customer` in the filename. Confirm that this project matches the official Task 4 brief before submission."),
    ("code", common + "\ndf = load_data(['sales', 'business', 'customer'])\noverview(df)"),
    ("markdown", "## KPI and trend exploration"),
    ("code", r'''
if df is not None:
    numeric = df.select_dtypes(include="number")
    print("Numeric KPI candidates:", list(numeric.columns))
    for col in numeric.columns[:8]:
        print(f"{col}: sum={numeric[col].sum():,.2f}; mean={numeric[col].mean():,.2f}")
    date_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["date", "order_time", "timestamp"]
    )]
    if date_cols:
        print("Date candidates:", date_cols)
        parsed = pd.to_datetime(df[date_cols[0]], errors="coerce")
        if parsed.notna().any() and len(numeric.columns):
            tmp = df.assign(_date=parsed).dropna(subset=["_date"])
            metric = numeric.columns[0]
            trend = tmp.groupby(tmp["_date"].dt.to_period("M"))[metric].sum()
            display(trend.to_frame(metric))
            trend.plot(title=f"Monthly total: {metric}", figsize=(10, 4))
            plt.tight_layout()
            plt.show()
    print("Validate currency, units, duplicate orders, returns and missing values before conclusions.")
else:
    print("Dataset missing. Add the official data before writing findings.")
'''),
    ("markdown", "## Business recommendations\n\nUse measured KPIs to identify opportunities, quantify limitations and recommend a testable action. Include the dataset source, assumptions, evaluation method and reproducible steps.")
])

reports = {
"Task1_Zomato_Report.md": """# Task 1 — Zomato Restaurant & Review Analysis

## Dataset and method
Record the supplied dataset filename, source, row/column count, data-cleaning decisions and limitations after running the notebook.

## Key findings
Complete this section from verified notebook output. Do not invent restaurant rankings, rating trends or cuisine preferences.

## Five recommendations
1. Add a recommendation supported by a verified result.
2. Add a recommendation supported by a verified result.
3. Add a recommendation supported by a verified result.
4. Add a recommendation supported by a verified result.
5. Add a recommendation supported by a verified result.

## Limitations
Document missing values, selection bias, date coverage and any columns whose meanings are unclear.
""",
"Task2_Loan_Report.md": """# Task 2 — Loan Approval Prediction

## Dataset and target
Document the source, sample size, features and confirmed target column.

## Evaluation
Paste actual held-out classification metrics and confusion matrix from the notebook. Compare with a simple baseline.

## Threshold recommendation
Choose only after validation, calibration and review of the costs of false approvals and false rejections. Explain the trade-off. Do not state a threshold without measured evidence.

## Responsible use
Discuss data leakage, class imbalance, fairness, explainability and the fact that the model is not a substitute for responsible lending review.
""",
"Task3_Instagram_Strategy.md": """# Task 3 — Instagram Engagement Strategy

## Audience and baseline
Use the actual dataset to document the available audience, post types and performance measures.

## Content strategy
- Test educational/how-to content against other formats.
- Test short-form video, carousel and static posts where appropriate.
- Use clear, relevant calls to action.
- Review saves, shares, reach, profile visits and link clicks.
- Run controlled experiments and compare like-for-like periods.

## Content calendar
See `reports/Instagram_Content_Calendar.csv`. Adapt dates, formats and themes to the account and actual results.

## Measurement
Set a baseline first; compare median performance and engagement rates only when the required denominator is available. Do not promise guaranteed follower growth.
""",
"Task4_Business_Analytics_Report.md": """# Task 4 — Business Sales & Customer Analytics

## Problem statement
Proposed project: analyze sales and customer data to identify measurable business opportunities. Confirm this topic against the official Task 4 brief.

## Data and methodology
Record the source, schema, cleaning steps, KPIs and assumptions after running the notebook.

## Results
Insert verified KPI tables and charts from the notebook. No findings are claimed until the actual dataset has been analyzed.

## Recommendations and next steps
Link each recommendation to a measured result, state expected direction rather than unsupported financial promises, and propose a way to validate impact.
"""
}

for name, content in reports.items():
    (ROOT / "reports" / name).write_text(content, encoding="utf-8")

with (ROOT / "reports" / "Instagram_Content_Calendar.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Week", "Post idea", "Format", "Objective", "CTA", "Metric to track"])
    rows = [
        [1, "Answer a common audience question", "Carousel", "Saves and shares", "Save for later", "Saves, shares"],
        [1, "Behind-the-scenes process", "Short video", "Reach and trust", "Comment with a question", "Reach, comments"],
        [2, "Practical checklist", "Carousel", "Educational value", "Share with a friend", "Shares, saves"],
        [2, "Myth vs fact", "Short video", "Discovery", "Follow for more tips", "Reach, profile visits"],
        [3, "Frequently asked question", "Static or video", "Conversation", "Leave your question", "Comments"],
        [3, "Step-by-step tutorial", "Carousel", "Saves", "Save this guide", "Saves, shares"],
        [4, "Monthly lessons and recap", "Carousel", "Community feedback", "Vote or comment", "Engagement, comments"],
        [4, "Test the best-performing theme", "Best format from data", "Validate performance", "Relevant CTA", "Compare with baseline"]
    ]
    w.writerows(rows)

(ROOT / "requirements.txt").write_text(
    "pandas\nnumpy\nmatplotlib\nscikit-learn\nopenpyxl\njupyter\n",
    encoding="utf-8"
)

readme = """# Alfido Tech — Data Science Internship

Projects prepared for four provisional tasks:
1. Zomato restaurant and review analysis
2. Loan approval prediction
3. Instagram engagement analysis
4. Proposed business sales and customer analytics (confirm against official Task 4 brief)

## Repository structure
- `notebooks/`: four Jupyter notebooks
- `data/raw/`: place the original task datasets here
- `reports/`: report drafts and Instagram content calendar
- `outputs/`: reserved for generated charts and results

## Setup
```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Open and run each notebook from the repository root. Add each original dataset to `data/raw/`; filenames should include `zomato`, `loan`, `instagram`/`engagement`, or `sales`/`business`/`customer` respectively. Check the official column names and target variable before interpreting results.

## Submission quality
Reports currently contain evidence-required sections, not fabricated findings. Replace every instruction with verified outputs, cite data sources, run all notebook cells, inspect charts, and export final reports before submitting. Confirm all four tasks and deliverable formats against the official assignment instructions.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

print("\nCreated project files:")
for p in sorted(ROOT.rglob("*")):
    if p.is_file() and ".git" not in p.parts:
        print("-", p.relative_to(ROOT))
print("\nNEXT: Add the original datasets to data/raw/, install requirements, and run all four notebooks.")
print("No real findings were invented. Task 4 topic and task mapping must match the official brief.")
