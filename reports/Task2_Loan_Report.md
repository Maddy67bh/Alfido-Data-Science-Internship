# Task 2: Loan Approval Prediction

## Dataset and methodology

- Source: `data/raw/loan_approval_dataset.csv`.
- The latest saved experiment compares Logistic Regression and Decision Tree.
- Metrics below come from `outputs/task2_model_comparison.csv`.
- The results describe this dataset and evaluation split only.

## Latest model comparison

- **Logistic Regression** — accuracy: 0.9356 (93.56%); ROC-AUC (one-vs-rest): 0.9748.
- **Decision Tree** — accuracy: 0.9778 (97.78%); ROC-AUC (one-vs-rest): 0.9938.

- Highest ROC-AUC in this comparison: Decision Tree (0.9938). This does not by itself establish that the model is suitable for real lending decisions.

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
