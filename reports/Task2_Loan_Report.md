# Task 2: Loan Approval Prediction

## Dataset and methodology

- Source: locally available CSV in `data/raw/`.
- Findings describe this dataset only; they are not automatically generalizable.

## Findings

- Rows: 4,269; columns: 13.
- Duplicate rows: 0.
- Cells missing values: 0.
- Target class counts: approved=2,656; rejected=1,613.
- Random Forest held-out test accuracy: 0.9754 (97.5%); test records: 854.
- Classification report:

```
              precision    recall  f1-score   support

    approved       0.98      0.98      0.98       531
    rejected       0.97      0.97      0.97       323

    accuracy                           0.98       854
   macro avg       0.97      0.97      0.97       854
weighted avg       0.98      0.98      0.98       854

```
- Confusion matrix saved to `outputs/task2_loan_confusion_matrix.csv` and PNG.
- This is a baseline on a public sample dataset. It does not establish real-world lending suitability, fairness, calibration, or causal validity.

## Outputs

Charts and supporting CSV files are saved in `outputs/`.
