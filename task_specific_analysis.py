from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RAW = Path("data/raw")
OUT = Path("outputs")
REPORTS = Path("reports")
OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def load_csv(name):
    path = RAW / name
    if not path.exists():
        print(f"SKIP: {path} not found")
        return None
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            print(f"Loaded {name}: {df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", low_memory=False)

def find_col(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for c in df.columns:
        if any(x.lower() in str(c).lower() for x in candidates):
            return c
    return None

def save_bar(series, title, filename, limit=12):
    if series is None:
        return
    s = series.dropna()
    if s.empty:
        return
    s = s.head(limit)
    if s.empty:
        return
    plt.figure(figsize=(10, 5))
    s.sort_values().plot(kind="barh")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=150)
    plt.close()

def write_report(filename, title, lines):
    text = "# " + title + "\n\n"
    text += "## Dataset and methodology\n\n"
    text += "- Source: locally available CSV in `data/raw/`.\n"
    text += "- Findings describe this dataset only; they are not automatically generalizable.\n\n"
    text += "## Findings\n\n"
    text += "\n".join(f"- {line}" for line in lines)
    text += "\n\n## Outputs\n\n"
    text += "Charts and supporting CSV files are saved in `outputs/`.\n"
    (REPORTS / filename).write_text(text, encoding="utf-8")
    print(f"Report written: reports/{filename}")

# TASK 1 — ZOMATO
df = load_csv("zomato.csv")
if df is not None:
    lines = [
        f"Rows: {len(df):,}; columns: {len(df.columns)}.",
        f"Duplicate rows: {df.duplicated().sum():,}.",
        f"Cells missing values: {int(df.isna().sum().sum()):,}."
    ]
    rating = find_col(df, ["Aggregate rating", "aggregate_rating", "rating"])
    city = find_col(df, ["City"])
    restaurant = find_col(df, ["Restaurant Name", "restaurant_name"])
    cuisine = find_col(df, ["Cuisines", "cuisine"])
    cost = find_col(df, ["Average Cost for two", "average_cost_for_two", "cost"])
    votes = find_col(df, ["Votes", "votes"])
    if rating:
        r = pd.to_numeric(df[rating], errors="coerce")
        lines.append(f"Average restaurant rating: {r.mean():.2f}; median: {r.median():.2f}; valid ratings: {r.notna().sum():,}.")
        save_bar(r.value_counts().sort_index(), "Restaurant rating distribution", "task1_zomato_rating_distribution.png", 15)
    if city:
        top = df[city].value_counts().head(10)
        top.rename_axis("city").reset_index(name="restaurants").to_csv(OUT / "task1_top_cities.csv", index=False)
        lines.append(f"City with most listings in this dataset: {top.index[0]} ({top.iloc[0]:,} listings)." if len(top) else "City field has no usable values.")
        save_bar(top, "Top cities by restaurant listings", "task1_zomato_top_cities.png")
    if cuisine:
        cuisines = df[cuisine].dropna().astype(str).str.split(",").explode().str.strip()
        cuisines = cuisines[cuisines.ne("")]
        top_cuisines = cuisines.value_counts().head(15)
        top_cuisines.rename_axis("cuisine").reset_index(name="mentions").to_csv(OUT / "task1_top_cuisines.csv", index=False)
        if len(top_cuisines):
            lines.append(f"Most frequently listed cuisine: {top_cuisines.index[0]} ({top_cuisines.iloc[0]:,} mentions).")
            save_bar(top_cuisines, "Most frequently listed cuisines", "task1_zomato_top_cuisines.png")
    if cost:
        c = pd.to_numeric(df[cost].astype(str).str.replace(",", "", regex=False), errors="coerce")
        lines.append(f"{cost}: median {c.median():,.2f}; mean {c.mean():,.2f} (currency/units follow the source dataset).")
    if votes:
        v = pd.to_numeric(df[votes], errors="coerce")
        if rating:
            temp = pd.DataFrame({"rating": pd.to_numeric(df[rating], errors="coerce"), "votes": v})
            lines.append(f"Correlation between rating and votes: {temp.corr(numeric_only=True).iloc[0,1]:.3f} (descriptive, not causal).")
    df.head(1000).to_csv(OUT / "task1_zomato_sample.csv", index=False)
    write_report("Task1_Zomato_Report.md", "Task 1: Zomato Restaurant Analysis", lines)

# TASK 2 — LOAN APPROVAL
df = load_csv("loan_approval_dataset.csv")
if df is not None:
    lines = [
        f"Rows: {len(df):,}; columns: {len(df.columns)}.",
        f"Duplicate rows: {df.duplicated().sum():,}.",
        f"Cells missing values: {int(df.isna().sum().sum()):,}."
    ]
    target = find_col(df, ["loan_status", "loan approval status", "approval_status", "approved"])
    if target:
        labels = df[target].astype(str).str.strip().str.lower()
        counts = labels.value_counts()
        counts.rename_axis("class").reset_index(name="count").to_csv(OUT / "task2_loan_class_balance.csv", index=False)
        lines.append("Target class counts: " + "; ".join(f"{k}={v:,}" for k, v in counts.items()) + ".")
        save_bar(counts, "Loan target class balance", "task2_loan_class_balance.png")
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.metrics import ConfusionMatrixDisplay

            X = df.drop(columns=[target]).copy()
            y = labels
            # Drop likely row identifiers; keep genuine applicant attributes.
            id_cols = [c for c in X.columns if c.lower().strip() in ("loan_id", "id", "application_id")]
            X = X.drop(columns=id_cols, errors="ignore")
            # Normalize whitespace in text fields.
            for c in X.select_dtypes(include=["object", "string"]).columns:
                X[c] = X[c].astype("string").str.strip()
            numeric = X.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in X.columns if c not in numeric]
            pre = ColumnTransformer([
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)
            ])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            model = Pipeline([
                ("prep", pre),
                ("classifier", RandomForestClassifier(
                    n_estimators=250, random_state=42, class_weight="balanced",
                    min_samples_leaf=2, n_jobs=-1
                ))
            ])
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            rep = classification_report(y_test, pred, zero_division=0)
            cm = confusion_matrix(y_test, pred, labels=model.classes_)
            pd.DataFrame(cm, index=model.classes_, columns=model.classes_).to_csv(OUT / "task2_loan_confusion_matrix.csv")
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(xticks_rotation=30)
            plt.title("Loan approval: held-out test confusion matrix")
            plt.tight_layout()
            plt.savefig(OUT / "task2_loan_confusion_matrix.png", dpi=150)
            plt.close()
            (OUT / "task2_loan_classification_report.txt").write_text(rep, encoding="utf-8")
            lines.extend([
                f"Random Forest held-out test accuracy: {acc:.4f} ({acc:.1%}); test records: {len(y_test):,}.",
                "Classification report:\n\n" + "```\n" + rep + "\n```",
                "Confusion matrix saved to `outputs/task2_loan_confusion_matrix.csv` and PNG.",
                "This is a baseline on a public sample dataset. It does not establish real-world lending suitability, fairness, calibration, or causal validity."
            ])
        except Exception as e:
            lines.append(f"Model training did not complete: {type(e).__name__}: {e}")
    else:
        lines.append("No recognizable approval target column was found; model training was skipped.")
    write_report("Task2_Loan_Report.md", "Task 2: Loan Approval Prediction", lines)

# TASK 3 — INSTAGRAM ENGAGEMENT
df = load_csv("instagram_analytics.csv")
if df is not None:
    lines = [
        f"Rows: {len(df):,}; columns: {len(df.columns)}.",
        f"Duplicate rows: {df.duplicated().sum():,}.",
        f"Cells missing values: {int(df.isna().sum().sum()):,}."
    ]
    engagement = find_col(df, ["engagement_rate"])
    category = find_col(df, ["content_category", "content type", "category"])
    media = find_col(df, ["media_type", "media type"])
    for col in [engagement, "likes", "comments", "shares", "saves", "reach", "impressions", "followers_gained"]:
        if col and col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"{col}: mean {vals.mean():,.2f}; median {vals.median():,.2f}; valid rows {vals.notna().sum():,}.")
    for col, label, prefix in [
        (category, "content category", "task3_instagram_category"),
        (media, "media type", "task3_instagram_media")
    ]:
        if col:
            agg = df.groupby(col, dropna=False).agg(
                posts=(col, "size")
            )
            if engagement:
                agg["avg_engagement_rate"] = pd.to_numeric(df[engagement], errors="coerce").groupby(df[col]).mean()
            if "likes" in df.columns:
                agg["avg_likes"] = pd.to_numeric(df["likes"], errors="coerce").groupby(df[col]).mean()
            agg.sort_values("posts", ascending=False).to_csv(OUT / f"{prefix}_performance.csv")
            if engagement and "avg_engagement_rate" in agg:
                ranked = agg.sort_values("avg_engagement_rate", ascending=False)
                lines.append(f"Highest average engagement rate by {label}: {ranked.index[0]} ({ranked.iloc[0]['avg_engagement_rate']:.3f}).")
                save_bar(ranked["avg_engagement_rate"], f"Average engagement rate by {label}", f"{prefix}_engagement.png")
    if engagement and media:
        plt.figure(figsize=(9, 5))
        temp = df[[media, engagement]].copy()
        temp[engagement] = pd.to_numeric(temp[engagement], errors="coerce")
        temp.boxplot(column=engagement, by=media, rot=25)
        plt.title("Engagement rate by media type")
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(OUT / "task3_instagram_engagement_by_media_boxplot.png", dpi=150)
        plt.close()
    write_report("Task3_Instagram_Strategy.md", "Task 3: Instagram Engagement Analysis", lines)

# TASK 4 — BUSINESS SALES AND CUSTOMERS
df = load_csv("business_sales_customers.csv")
if df is not None:
    lines = [
        f"Rows: {len(df):,}; columns: {len(df.columns)}.",
        f"Duplicate rows: {df.duplicated().sum():,}.",
        f"Cells missing values: {int(df.isna().sum().sum()):,}."
    ]
    date_col = find_col(df, ["Order_Date", "order date", "date"])
    amount = find_col(df, ["Order_Amount", "order amount", "sales", "revenue"])
    profit = find_col(df, ["Profit_Amount", "profit amount", "profit"])
    product = find_col(df, ["Product_Category", "product category", "category"])
    customer = find_col(df, ["Customer_ID", "customer id"])
    returned = find_col(df, ["Returned", "return status", "returned"])
    status = find_col(df, ["Order_Status", "order status"])
    for col, label in [(amount, "order amount"), (profit, "profit amount")]:
        if col:
            x = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"{label.title()}: total {x.sum():,.2f}; average {x.mean():,.2f}; median {x.median():,.2f}. Units/currency are not independently verified.")
    if customer:
        lines.append(f"Unique customers: {df[customer].nunique(dropna=True):,}.")
    if product and amount:
        df[amount] = pd.to_numeric(df[amount], errors="coerce")
        agg = df.groupby(product, dropna=False).agg(
            orders=(product, "size"),
            total_order_amount=(amount, "sum"),
            average_order_amount=(amount, "mean")
        ).sort_values("total_order_amount", ascending=False)
        agg.to_csv(OUT / "task4_sales_by_category.csv")
        if len(agg):
            lines.append(f"Category with highest recorded order amount: {agg.index[0]} ({agg.iloc[0]['total_order_amount']:,.2f}).")
            save_bar(agg["total_order_amount"], "Total order amount by product category", "task4_sales_by_category.png")
        if profit:
            df[profit] = pd.to_numeric(df[profit], errors="coerce")
            prof = df.groupby(product)[profit].sum().sort_values(ascending=False)
            prof.rename_axis("product_category").reset_index(name="total_profit").to_csv(OUT / "task4_profit_by_category.csv", index=False)
            save_bar(prof, "Total profit by product category", "task4_profit_by_category.png")
    if date_col and amount:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        trend = df.dropna(subset=[date_col]).groupby(df[date_col].dt.to_period("M"))[amount].sum()
        if len(trend):
            trend.index = trend.index.astype(str)
            trend.rename_axis("month").reset_index(name="total_order_amount").to_csv(OUT / "task4_monthly_sales.csv", index=False)
            plt.figure(figsize=(11, 5))
            trend.plot()
            plt.title("Monthly order amount")
            plt.xlabel("Month")
            plt.ylabel("Order amount (source units)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUT / "task4_monthly_sales.png", dpi=150)
            plt.close()
            lines.append(f"Monthly trend covers {len(trend)} months after date parsing.")
    if returned:
        lines.append("Return field value counts: " + "; ".join(
            f"{k}={v:,}" for k, v in df[returned].astype(str).value_counts().head(10).items()
        ) + ".")
    if status:
        lines.append("Order status counts: " + "; ".join(
            f"{k}={v:,}" for k, v in df[status].astype(str).value_counts().head(10).items()
        ) + ".")
    write_report("Task4_Business_Analytics_Report.md", "Task 4: Business Sales and Customer Analytics", lines)

print("\nAnalysis run finished. Check reports/ and outputs/ for generated files.")
