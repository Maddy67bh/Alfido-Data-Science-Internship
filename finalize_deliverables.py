from pathlib import Path
import re
import pandas as pd
import nbformat
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
)
from xml.sax.saxutils import escape

ROOT = Path(".")
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
DELIVERABLES = ROOT / "deliverables"
DATA = ROOT / "data" / "raw"
DELIVERABLES.mkdir(exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=19, leading=23, spaceAfter=16
))
styles.add(ParagraphStyle(
    name="SmallBody", parent=styles["BodyText"],
    fontSize=9, leading=13, spaceAfter=7
))

def safe_text(s):
    return escape(str(s)).replace("\n", "<br/>")

def markdown_to_pdf(title, source_file, output_file, extras, images=()):
    story = [Paragraph(safe_text(title), styles["ReportTitle"])]
    story.append(Paragraph(
        "Alfido Tech Data Science Internship | Project deliverable",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    if source_file.exists():
        lines = source_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("|") or line.startswith("```"):
                continue
            if line.startswith("#"):
                heading = re.sub(r"^#+\s*", "", line)
                story.append(Paragraph(safe_text(heading), styles["Heading2"]))
            elif line.startswith(("- ", "* ")):
                story.append(Paragraph(
                    "• " + safe_text(line[2:]), styles["SmallBody"]
                ))
            else:
                line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
                line = re.sub(r"`([^`]*)`", r"\1", line)
                story.append(Paragraph(safe_text(line), styles["SmallBody"]))
    else:
        story.append(Paragraph(
            "A source Markdown report was not found. See the notebook and "
            "generated analysis outputs in this repository.",
            styles["SmallBody"]
        ))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Recommendations and next steps", styles["Heading1"]))
    for item in extras:
        story.append(Paragraph("• " + safe_text(item), styles["SmallBody"]))

    for image_path in images:
        path = OUTPUTS / image_path
        if path.exists():
            story.append(Spacer(1, 10))
            story.append(Paragraph(safe_text(path.stem.replace("_", " ").title()),
                                   styles["Heading2"]))
            img = Image(str(path))
            img._restrictSize(6.2 * inch, 3.5 * inch)
            story.append(img)

    doc = SimpleDocTemplate(
        str(output_file), pagesize=A4,
        rightMargin=45, leftMargin=45, topMargin=42, bottomMargin=42
    )
    doc.build(story)
    print("Created:", output_file)

# Validate the four expected datasets and report their shapes.
dataset_files = {
    "Zomato": DATA / "zomato.csv",
    "Loan approval": DATA / "loan_approval_dataset.csv",
    "Instagram": DATA / "instagram_analytics.csv",
    "Business sales": DATA / "business_sales_customers.csv",
}
inventory = ["# Dataset inventory", ""]
for name, path in dataset_files.items():
    if path.exists():
        try:
            df = pd.read_csv(path, nrows=5, encoding="utf-8")
            count = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
            inventory.append(
                f"- **{name}:** `{path.as_posix()}` — approximately "
                f"{count:,} data rows, {len(df.columns)} columns."
            )
        except Exception as e:
            inventory.append(f"- **{name}:** file exists; read check needs review ({e}).")
    else:
        inventory.append(f"- **{name}:** MISSING — add the required dataset.")
(REPORTS / "Dataset_Inventory.md").write_text(
    "\n".join(inventory) + "\n", encoding="utf-8"
)

markdown_to_pdf(
    "Task 1 — Zomato Restaurant Analysis",
    REPORTS / "Task1_Zomato_Report.md",
    DELIVERABLES / "Task1_Zomato_Analysis_Report.pdf",
    [
        "Compare restaurant ratings across cities and cuisines before choosing partnership targets.",
        "Review price range and average cost alongside ratings; do not assume price alone causes better ratings.",
        "Prioritize locations with a strong restaurant supply and useful customer demand indicators.",
        "Use cuisine popularity and review volume to guide restaurant discovery content.",
        "Validate missing values, duplicate records, currency units and dataset source before business decisions."
    ],
    ["task1_zomato_rating_distribution.png",
     "task1_zomato_top_cities.png",
     "task1_zomato_top_cuisines.png"]
)

markdown_to_pdf(
    "Task 2 — Loan Approval Prediction",
    REPORTS / "Task2_Loan_Report.md",
    DELIVERABLES / "Task2_Loan_Model_Report.pdf",
    [
        "Compare logistic regression with a tree-based model using the same train/test split.",
        "Report precision, recall, F1-score and ROC-AUC, not accuracy alone.",
        "Review class imbalance and confusion-matrix errors, especially false approvals and false rejections.",
        "Choose any deployment threshold only after validation, calibration and cost analysis.",
        "Treat results as educational dataset performance, not as a real lending decision or guarantee."
    ],
    ["task2_loan_class_balance.png", "task2_loan_confusion_matrix.png"]
)

markdown_to_pdf(
    "Task 3 — Instagram Engagement Analysis",
    REPORTS / "Task3_Instagram_Strategy.md",
    DELIVERABLES / "Task3_Instagram_Strategy.pdf",
    [
        "Compare engagement rate and reach by media type and content category.",
        "Test posting days and times using upload timestamps; do not infer an optimal time without evidence.",
        "Measure likes, comments, shares and saves separately to understand different engagement signals.",
        "Run controlled hashtag and caption experiments, changing one major factor at a time.",
        "Review results weekly and adapt the content calendar to the best-performing verified patterns."
    ],
    ["task3_instagram_media_engagement.png",
     "task3_instagram_category_engagement.png"]
)

markdown_to_pdf(
    "Task 4 — Business Sales Analytics (Optional Capstone)",
    REPORTS / "Task4_Business_Analytics_Report.md",
    DELIVERABLES / "Task4_Business_Sales_Capstone.pdf",
    [
        "Track revenue, profit, order volume and average order value over time.",
        "Compare category-level sales and profit before reallocating promotions.",
        "Check returns, discounts, missing values, currency and duplicate order IDs.",
        "Segment customers only after confirming the meaning and reliability of customer identifiers.",
        "Present findings as descriptive analytics; forecast only after a validated time-based evaluation."
    ],
    ["task4_monthly_sales.png", "task4_sales_by_category.png",
     "task4_profit_by_category.png"]
)

# Produce a clean repository README without deleting existing content.
readme = ROOT / "README.md"
existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Alfido Data Science Internship\n"
marker = "<!-- FINAL DELIVERABLES SECTION -->"
if marker not in existing:
    section = f"""

{marker}
## Project deliverables

This repository contains notebooks, analysis reports, generated visualizations,
and supplementary PDF deliverables for the projects documented here.

### Tasks
1. **Zomato restaurant analysis** — `notebooks/Task1_Zomato_Analysis.ipynb`
2. **Loan approval prediction** — `notebooks/Task2_Loan_Approval_Prediction.ipynb`
3. **Instagram engagement analysis** — `notebooks/Task3_Instagram_Engagement_Analysis.ipynb`
4. **Business sales capstone (optional)** — `notebooks/Task4_Business_Sales_Analytics.ipynb`

### PDF deliverables
- `deliverables/Task1_Zomato_Analysis_Report.pdf`
- `deliverables/Task2_Loan_Model_Report.pdf`
- `deliverables/Task3_Instagram_Strategy.pdf`
- `deliverables/Task4_Business_Sales_Capstone.pdf`

### Run locally
```bash
pip install pandas matplotlib scikit-learn jupyter nbformat nbclient reportlab
jupyter notebook
```

### Reproducibility and limitations
- Input files are located in `data/raw/`.
- Review each dataset's original source and usage terms before redistribution.
- Model metrics describe performance on the available dataset and split only.
- Thresholds, correlations and descriptive trends do not establish causation.
- Confirm task requirements against the official Alfido assignment before submission.
"""
    readme.write_text(existing.rstrip() + "\n" + section, encoding="utf-8")

print("\nDeliverable generation finished.")
print("Review the PDFs and README before submitting.")
