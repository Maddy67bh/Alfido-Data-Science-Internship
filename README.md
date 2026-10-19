# Alfido Tech — Data Science Internship

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


<!-- FINAL DELIVERABLES SECTION -->
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
