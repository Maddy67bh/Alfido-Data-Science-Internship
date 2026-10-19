from pathlib import Path
import re
import subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(".")
DATA = ROOT / "data/raw"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
PDFS = ROOT / "deliverables"
for p in (OUT, REPORTS, PDFS):
    p.mkdir(exist_ok=True)

# -------------------------------------------------------
# 1. Correct Task 4 monthly revenue and category metrics
# -------------------------------------------------------
src = DATA / "business_sales_customers.csv"
df = pd.read_csv(src)
df.columns = [str(c).strip() for c in df.columns]

required = ["Year", "Month", "Order_Amount", "Profit_Amount", "Product_Category"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Required Task 4 columns not found: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

for col in ["Year", "Month", "Order_Amount", "Profit_Amount"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid = df.dropna(subset=["Year", "Month"]).copy()
valid["Year"] = valid["Year"].astype(int)
valid["Month"] = valid["Month"].astype(int)
valid = valid[valid["Month"].between(1, 12)].copy()
valid["_month"] = (
    valid["Year"].astype(str) + "-" +
    valid["Month"].astype(str).str.zfill(2)
)

monthly = (
    valid.dropna(subset=["Order_Amount"])
    .groupby("_month", as_index=False)["Order_Amount"]
    .sum()
    .rename(columns={"Order_Amount": "Revenue"})
    .sort_values("_month")
)
monthly.to_csv(OUT / "task4_monthly_revenue.csv", index=False)

monthly_sales = (
    valid.groupby("_month", as_index=False)
    .agg(Orders=("Order_ID", "nunique") if "Order_ID" in valid.columns
         else ("Order_Amount", "count"))
    .sort_values("_month")
)
monthly_sales.to_csv(OUT / "task4_monthly_sales.csv", index=False)

category = (
    df.groupby("Product_Category", dropna=False)
    .agg(
        sum=("Order_Amount", "sum"),
        mean=("Order_Amount", "mean"),
        count=("Order_Amount", "count")
    )
    .reset_index()
    .sort_values("sum", ascending=False)
)
category.to_csv(OUT / "task4_sales_by_category_summary.csv", index=False)
category.to_csv(OUT / "task4_profit_by_category_summary.csv", index=False)

profit_category = (
    df.groupby("Product_Category", dropna=False)
    .agg(
        sum=("Profit_Amount", "sum"),
        mean=("Profit_Amount", "mean"),
        count=("Profit_Amount", "count")
    )
    .reset_index()
    .sort_values("sum", ascending=False)
)
profit_category.to_csv(OUT / "task4_profit_by_category_summary.csv", index=False)

# Charts: explicitly use revenue, not discount amount.
if not monthly.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["_month"], monthly["Revenue"], marker="o")
    plt.title("Monthly Revenue")
    plt.xlabel("Month")
    plt.ylabel("Order Amount (source units; currency unverified)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "task4_monthly_revenue.png", dpi=160)
    plt.close()

if not category.empty:
    plt.figure(figsize=(9, 5))
    plt.bar(category["Product_Category"].astype(str), category["sum"])
    plt.title("Revenue by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Order Amount (source units; currency unverified)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "task4_revenue_by_category.png", dpi=160)
    plt.close()

plt.figure(figsize=(9, 5))
plt.bar(profit_category["Product_Category"].astype(str), profit_category["sum"])
plt.title("Profit by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Profit Amount (source units; currency unverified)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUT / "task4_profit_by_category.png", dpi=160)
plt.close()

# Replace Task 4 report with metrics recalculated from the source dataset.
total_revenue = df["Order_Amount"].sum()
total_profit = df["Profit_Amount"].sum()
avg_order = df["Order_Amount"].mean()
median_order = df["Order_Amount"].median()
dupes = int(df.duplicated().sum())
missing_cells = int(df.isna().sum().sum())
best_rev = category.iloc[0] if not category.empty else None
best_profit = profit_category.iloc[0] if not profit_category.empty else None
customer_col = next(
    (c for c in df.columns if c.lower() in ("customer_id", "customerid")),
    None
)
unique_customers = df[customer_col].nunique() if customer_col else None

report = f"""# Task 4: Business Sales and Customer Analytics

## Dataset and methodology

- Source: `data/raw/business_sales_customers.csv`.
- Records: {len(df):,}; columns: {len(df.columns)}.
- Duplicate full rows: {dupes:,}.
- Missing cells across the dataset: {missing_cells:,}.
- Revenue metric: sum of `Order_Amount`, grouped by the dataset's Year and Month fields.
- Profit metric: sum of `Profit_Amount`, grouped by `Product_Category`.
- Currency and units were not independently verified; monetary figures are reported in source units.

## Findings

- Total recorded order amount: {total_revenue:,.2f}.
- Average recorded order amount: {avg_order:,.2f}.
- Median recorded order amount: {median_order:,.2f}.
- Total recorded profit amount: {total_profit:,.2f}.
- Monthly revenue series contains {len(monthly):,} month groups with valid year/month fields.
"""
if unique_customers is not None:
    report += f"- Unique customers according to `{customer_col}`: {unique_customers:,}.\n"
if best_rev is not None:
    report += (
        f"- Highest recorded revenue category: {best_rev['Product_Category']} "
        f"({best_rev['sum']:,.2f}).\n"
    )
if best_profit is not None:
    report += (
        f"- Highest recorded total profit category: {best_profit['Product_Category']} "
        f"({best_profit['sum']:,.2f}).\n"
    )

report += """
## Business recommendations

- Monitor monthly revenue and profit separately; do not use discount amount as a proxy for revenue.
- Compare category revenue with category profit before allocating promotions.
- Validate currency, units, return handling, order statuses and duplicate order identifiers before operational decisions.
- Treat the results as descriptive analytics, not proof that a category or promotion caused better performance.
- Validate any forecast on a chronological holdout period before using it for planning.

## Outputs

- `outputs/task4_monthly_revenue.csv`
- `outputs/task4_monthly_revenue.png`
- `outputs/task4_sales_by_category_summary.csv`
- `outputs/task4_profit_by_category_summary.csv`
- `outputs/task4_revenue_by_category.png`
- `outputs/task4_profit_by_category.png`
"""
(REPORTS / "Task4_Business_Analytics_Report.md").write_text(report, encoding="utf-8")

# -------------------------------------------------------
# 2. Correct Task 2 report using the latest model CSV
# -------------------------------------------------------
models_path = OUT / "task2_model_comparison.csv"
models = pd.read_csv(models_path)
models["accuracy"] = pd.to_numeric(models["accuracy"], errors="coerce")
models["roc_auc_ovr"] = pd.to_numeric(models["roc_auc_ovr"], errors="coerce")

task2 = """# Task 2: Loan Approval Prediction

## Dataset and methodology

- Source: `data/raw/loan_approval_dataset.csv`.
- The latest saved experiment compares Logistic Regression and Decision Tree.
- Metrics below come from `outputs/task2_model_comparison.csv`.
- The results describe this dataset and evaluation split only.

## Latest model comparison

"""
for _, row in models.iterrows():
    task2 += (
        f"- **{row['model']}** — accuracy: {row['accuracy']:.4f} "
        f"({row['accuracy']*100:.2f}%); ROC-AUC (one-vs-rest): "
        f"{row['roc_auc_ovr']:.4f}.\n"
    )

if not models.empty:
    best = models.sort_values("roc_auc_ovr", ascending=False).iloc[0]
    task2 += (
        f"\n- Highest ROC-AUC in this comparison: {best['model']} "
        f"({best['roc_auc_ovr']:.4f}). This does not by itself establish "
        "that the model is suitable for real lending decisions.\n"
    )

task2 += """
## Interpretation and limitations

- Compare precision, recall, F1-score and confusion-matrix errors before choosing a model.
- Check for target leakage, preprocessing leakage, class imbalance and appropriate stratification.
- Assess calibration, subgroup fairness and the costs of false approvals and false rejections.
- These educational results are not a credit decision system and do not demonstrate real-world lending suitability.

## Outputs

- `outputs/task2_model_comparison.csv`
- `outputs/task2_logistic_regression_report.txt`
- `outputs/task2_decision_tree_report.txt`
- `outputs/task2_logistic_confusion_matrix.png`
"""
(REPORTS / "Task2_Loan_Report.md").write_text(task2, encoding="utf-8")

# -------------------------------------------------------
# 3. Document the limitation in Task 3's hourly analysis
# -------------------------------------------------------
task3_path = REPORTS / "Task3_Instagram_Strategy.md"
task3 = task3_path.read_text(encoding="utf-8")
hour_note = (
    "\n## Posting-time limitation\n\n"
    "The current hourly summary contains only one observed hour (9) across "
    "29,999 records. This does not provide a meaningful comparison of posting "
    "times. Verify the source timestamps and test multiple hours before "
    "recommending an optimal posting time.\n"
)
if "## Posting-time limitation" not in task3:
    task3 += hour_note
task3_path.write_text(task3, encoding="utf-8")

# -------------------------------------------------------
# 4. Point PDF generator to actual generated chart names
# -------------------------------------------------------
finalizer = ROOT / "finalize_deliverables.py"
if not finalizer.exists():
    raise FileNotFoundError("finalize_deliverables.py is missing")

code = finalizer.read_text(encoding="utf-8")
replacements = {
    '"task1_zomato_rating_distribution.png",\n     "task1_zomato_top_cities.png",\n     "task1_zomato_top_cuisines.png"':
    '"task1_city_average_rating.png",\n     "task1_cuisine_average_rating.png",\n     "task1_cost_vs_rating.png"',
    '"task2_loan_class_balance.png", "task2_loan_confusion_matrix.png"':
    '"task2_logistic_confusion_matrix.png"',
    '"task3_instagram_media_engagement.png",\n     "task3_instagram_category_engagement.png"':
    '"task3_media_type_interactions.png"',
    '"task4_monthly_sales.png", "task4_sales_by_category.png",\n     "task4_profit_by_category.png"':
    '"task4_monthly_revenue.png", "task4_revenue_by_category.png",\n     "task4_profit_by_category.png"',
}
for old, new in replacements.items():
    if old in code:
        code = code.replace(old, new)
    else:
        print("NOTICE: image list pattern not found; review finalizer:", old[:70])
finalizer.write_text(code, encoding="utf-8")

print("Corrected Task 4 monthly revenue and category metrics.")
print("Updated Task 2 model comparison and Task 3 posting-time limitation.")
