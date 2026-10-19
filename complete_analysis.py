from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

ROOT = Path(".")
DATA = ROOT / "data/raw"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
NOTEBOOKS = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def find_csv(filename):
    p = DATA / filename
    if not p.exists():
        raise FileNotFoundError(f"Dataset missing: {p}")
    return pd.read_csv(p, low_memory=False)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close()

def get_col(df, candidates):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for c in df.columns:
        if any(x in str(c).lower() for x in candidates):
            return c
    return None

def add_section(notebook_name, title, code):
    path = NOTEBOOKS / notebook_name
    nb = nbformat.read(path, as_version=4)
    marker = "AUTO-GENERATED ANALYSIS EXTENSION"
    if any(marker in c.source for c in nb.cells):
        print(f"Extension already present: {notebook_name}")
        return
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f"# {title}\n\n{marker}\n\nThis section adds reproducible analysis using the dataset available in `data/raw/`."
    ))
    nb.cells.append(nbformat.v4.new_code_cell(code))
    nbformat.write(nb, path)
    print(f"Extended notebook: {notebook_name}")

# TASK 1: Zomato
z = find_csv("zomato.csv")
z.columns = [str(c).strip() for c in z.columns]
z = z.drop_duplicates()
rating = get_col(z, ["rate", "rating", "aggregate rating"])
city = get_col(z, ["city", "location"])
cuisine = get_col(z, ["cuisines", "cuisine"])
cost = get_col(z, ["average cost for two", "average_cost_for_two", "cost"])

if rating:
    z[rating] = pd.to_numeric(z[rating].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
z.to_csv(OUT / "task1_zomato_cleaned.csv", index=False)
z.describe(include="all").transpose().to_csv(OUT / "task1_zomato_descriptive_statistics.csv")

if city and rating:
    top = z.groupby(city)[rating].agg(["mean", "count"]).query("count >= 3").sort_values("mean", ascending=False).head(15)
    top.to_csv(OUT / "task1_city_rating_summary.csv")
    top["mean"].sort_values().plot(kind="barh", figsize=(9, 6), title="Highest Average Restaurant Ratings by City")
    save_fig("task1_city_average_rating.png")

if cuisine and rating:
    tmp = z.assign(_cuisine=z[cuisine].fillna("").astype(str).str.split(",")).explode("_cuisine")
    tmp["_cuisine"] = tmp["_cuisine"].str.strip()
    top = tmp.groupby("_cuisine")[rating].agg(["mean", "count"]).query("count >= 5").sort_values("count", ascending=False).head(15)
    top.to_csv(OUT / "task1_cuisine_summary.csv")
    top.sort_values("mean")["mean"].plot(kind="barh", figsize=(9, 6), title="Average Ratings by Common Cuisine")
    save_fig("task1_cuisine_average_rating.png")

if cost and rating:
    z[cost] = pd.to_numeric(z[cost].astype(str).str.replace(",", "", regex=False), errors="coerce")
    z[[cost, rating]].dropna().plot.scatter(x=cost, y=rating, alpha=.35, figsize=(8, 5), title="Cost vs Rating")
    save_fig("task1_cost_vs_rating.png")

# TASK 2: Loan prediction
loan = find_csv("loan_approval_dataset.csv")
loan.columns = [str(c).strip() for c in loan.columns]
target = get_col(loan, ["loan_status", "loan approval", "approved", "status"])
loan.to_csv(OUT / "task2_loan_cleaned.csv", index=False)
loan.describe(include="all").transpose().to_csv(OUT / "task2_loan_descriptive_statistics.csv")
model_summary = ["# Loan Approval Model Evaluation", ""]
if target:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X = loan.drop(columns=[target]).copy()
    X = X.drop(columns=[c for c in X.columns if str(c).strip().lower() in ("loan_id", "loan id", "id")], errors="ignore")
    X = X.replace(r"^\s*$", np.nan, regex=True)
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    y = loan[target].astype(str).str.strip()
    le = LabelEncoder()
    y = le.fit_transform(y)
    if len(np.unique(y)) >= 2 and len(y) >= 20:
        num = X.select_dtypes(include=np.number).columns.tolist()
        cat = [c for c in X.columns if c not in num]
        pre = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encode", OneHotEncoder(handle_unknown="ignore"))]), cat)
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=42, stratify=y
        )
        rows = []
        for name, clf in [
            ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("Decision Tree", DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42))
        ]:
            pipe = Pipeline([("preprocess", pre), ("model", clf)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            row = {"model": name, "accuracy": accuracy_score(y_test, pred)}
            report = classification_report(y_test, pred, target_names=le.classes_, zero_division=0)
            row["classification_report"] = report
            try:
                proba = pipe.predict_proba(X_test)
                row["roc_auc_ovr"] = roc_auc_score(y_test, proba, multi_class="ovr") if len(le.classes_) > 2 else roc_auc_score(y_test, proba[:, 1])
            except Exception:
                row["roc_auc_ovr"] = np.nan
            rows.append(row)
            (OUT / f"task2_{name.lower().replace(' ', '_')}_report.txt").write_text(
                f"{name}\nAccuracy: {row['accuracy']:.4f}\nROC-AUC: {row['roc_auc_ovr']}\n\n{report}\n",
                encoding="utf-8"
            )
            if name == "Logistic Regression":
                pd.DataFrame(confusion_matrix(y_test, pred), index=le.classes_, columns=le.classes_).to_csv(OUT / "task2_logistic_confusion_matrix.csv")
                plt.figure(figsize=(6, 5))
                from sklearn.metrics import ConfusionMatrixDisplay
                ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=le.classes_, xticks_rotation=30)
                save_fig("task2_logistic_confusion_matrix.png")
        pd.DataFrame([{k:v for k,v in r.items() if k != "classification_report"} for r in rows]).to_csv(OUT / "task2_model_comparison.csv", index=False)
        model_summary += [f"- Target column: `{target}`", "- Models: logistic regression and decision tree.", "- Metrics saved in `outputs/task2_model_comparison.csv`.", ""]
    else:
        model_summary.append("Target has insufficient classes or rows for classification.")
else:
    model_summary.append("Target column could not be detected automatically. Review dataset columns.")
REPORTS.joinpath("Task2_Model_Evaluation_Addendum.md").write_text("\n".join(model_summary), encoding="utf-8")

# TASK 3: Instagram
ig = find_csv("instagram_analytics.csv")
ig.columns = [str(c).strip() for c in ig.columns]
ig.to_csv(OUT / "task3_instagram_cleaned.csv", index=False)
ig.describe(include="all").transpose().to_csv(OUT / "task3_instagram_descriptive_statistics.csv")
likes = get_col(ig, ["likes", "like_count"])
comments = get_col(ig, ["comments", "comment_count"])
followers = get_col(ig, ["followers", "follower_count", "followers_count"])
media = get_col(ig, ["media_type", "post_type", "content_type", "media"])
category = get_col(ig, ["category", "content_category", "post_category"])
date = get_col(ig, ["date", "timestamp", "posted_at", "created_at", "upload_date"])
if likes and comments:
    ig[likes] = pd.to_numeric(ig[likes], errors="coerce")
    ig[comments] = pd.to_numeric(ig[comments], errors="coerce")
    ig["calculated_interactions"] = ig[likes].fillna(0) + ig[comments].fillna(0)
    if followers:
        ig[followers] = pd.to_numeric(ig[followers], errors="coerce")
        ig["engagement_per_follower_pct"] = 100 * ig["calculated_interactions"] / ig[followers].replace(0, np.nan)
if media and "calculated_interactions" in ig:
    summary = ig.groupby(media)["calculated_interactions"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    summary.to_csv(OUT / "task3_media_type_summary.csv")
    summary["mean"].sort_values().plot(kind="barh", figsize=(8, 5), title="Mean Interactions by Media Type")
    save_fig("task3_media_type_interactions.png")
if category and "calculated_interactions" in ig:
    summary = ig.groupby(category)["calculated_interactions"].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    summary.to_csv(OUT / "task3_category_summary.csv")
if date:
    ig["_parsed_date"] = pd.to_datetime(ig[date], errors="coerce")
    if ig["_parsed_date"].notna().any():
        ig["posting_day"] = ig["_parsed_date"].dt.day_name()
        ig["posting_hour"] = ig["_parsed_date"].dt.hour
        if "calculated_interactions" in ig:
            ig.groupby("posting_day")["calculated_interactions"].agg(["mean", "count"]).to_csv(OUT / "task3_engagement_by_day.csv")
            ig.groupby("posting_hour")["calculated_interactions"].agg(["mean", "count"]).to_csv(OUT / "task3_engagement_by_hour.csv")
ig.to_csv(OUT / "task3_instagram_enriched.csv", index=False)

# TASK 4: Business sales
sales = find_csv("business_sales_customers.csv")
sales.columns = [str(c).strip() for c in sales.columns]
sales.to_csv(OUT / "task4_business_sales_cleaned.csv", index=False)
sales.describe(include="all").transpose().to_csv(OUT / "task4_business_descriptive_statistics.csv")
revenue = get_col(sales, ["revenue", "sales", "sales_amount", "total_sales", "amount", "total_amount"])
profit = get_col(sales, ["profit", "net_profit"])
category = get_col(sales, ["category", "product_category", "product"])
date = get_col(sales, ["order_date", "date", "transaction_date", "purchase_date"])
if revenue:
    sales[revenue] = pd.to_numeric(sales[revenue].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if category:
        sales.groupby(category)[revenue].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False).to_csv(OUT / "task4_sales_by_category_summary.csv")
        s = sales.groupby(category)[revenue].sum().sort_values()
        s.plot(kind="barh", figsize=(9, 5), title="Revenue by Category")
        save_fig("task4_revenue_by_category.png")
if profit:
    sales[profit] = pd.to_numeric(sales[profit].astype(str).str.replace(",", "", regex=False), errors="coerce")
    if category:
        sales.groupby(category)[profit].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False).to_csv(OUT / "task4_profit_by_category_summary.csv")
if date and revenue:
    sales["_parsed_date"] = pd.to_datetime(sales[date], errors="coerce")
    if sales["_parsed_date"].notna().any():
        sales["_month"] = sales["_parsed_date"].dt.to_period("M").astype(str)
        monthly = sales.groupby("_month")[revenue].sum()
        monthly.to_csv(OUT / "task4_monthly_revenue.csv")
        monthly.plot(figsize=(10, 5), title="Monthly Revenue Trend")
        save_fig("task4_monthly_revenue.png")
sales.to_csv(OUT / "task4_business_sales_enriched.csv", index=False)

# Add runnable analysis code to each notebook.
extensions = {
"Task1_Zomato_Analysis.ipynb": r'''
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
df = pd.read_csv("data/raw/zomato.csv", low_memory=False)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
display(df.isna().sum().sort_values(ascending=False).head(15).to_frame("missing_values"))
display(df.describe(include="all").transpose().head(30))
''',
"Task2_Loan_Approval_Prediction.ipynb": r'''
import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/loan_approval_dataset.csv", low_memory=False)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
display(df.isna().sum().sort_values(ascending=False).to_frame("missing_values"))
print("Target candidates:", [c for c in df.columns if any(x in c.lower() for x in ["status","approv"])])
if Path("outputs/task2_model_comparison.csv").exists():
    display(pd.read_csv("outputs/task2_model_comparison.csv"))
''',
"Task3_Instagram_Engagement_Analysis.ipynb": r'''
import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/instagram_analytics.csv", low_memory=False)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
display(df.isna().sum().sort_values(ascending=False).head(20).to_frame("missing_values"))
if Path("outputs/task3_media_type_summary.csv").exists():
    display(pd.read_csv("outputs/task3_media_type_summary.csv"))
if Path("outputs/task3_engagement_by_day.csv").exists():
    display(pd.read_csv("outputs/task3_engagement_by_day.csv"))
''',
"Task4_Business_Sales_Analytics.ipynb": r'''
import pandas as pd
from pathlib import Path
df = pd.read_csv("data/raw/business_sales_customers.csv", low_memory=False)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head())
display(df.isna().sum().sort_values(ascending=False).head(20).to_frame("missing_values"))
if Path("outputs/task4_sales_by_category_summary.csv").exists():
    display(pd.read_csv("outputs/task4_sales_by_category_summary.csv"))
if Path("outputs/task4_monthly_revenue.csv").exists():
    display(pd.read_csv("outputs/task4_monthly_revenue.csv"))
'''
}
for filename, code in extensions.items():
    add_section(filename, "Additional Reproducible Analysis", code)

print("\nAnalysis output generation completed.")
print("Review reports/ and outputs/ for actual results.")
