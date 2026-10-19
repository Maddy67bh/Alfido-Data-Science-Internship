# Loan Approval Prediction

Dataset: `loan_approval_dataset.csv`

Rows used: 4269

Target column: ` loan_status`

Holdout accuracy: 0.979

Classification report:

```
              precision    recall  f1-score   support

    Approved       0.98      0.98      0.98       531
    Rejected       0.98      0.97      0.97       323

    accuracy                           0.98       854
   macro avg       0.98      0.98      0.98       854
weighted avg       0.98      0.98      0.98       854

```

This is a baseline model, not a lending decision system. Review fairness, leakage, and class balance before use.
