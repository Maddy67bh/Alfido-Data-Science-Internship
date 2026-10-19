from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW=Path("data/raw"); OUT=Path("outputs"); REP=Path("reports")
OUT.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
files=list(RAW.glob("*.csv"))+list(RAW.glob("*.xlsx"))+list(RAW.glob("*.xls"))
def read(p):
    return pd.read_excel(p) if p.suffix.lower() in (".xlsx",".xls") else pd.read_csv(p, low_memory=False)
def save_report(name, title, body):
    (REP/name).write_text("# "+title+"\n\n"+body+"\n", encoding="utf-8")
summary=[]
for p in files:
    try:
        df=read(p)
        print(f"\n{p.name}: {df.shape}")
        print(df.head(3).to_string(index=False))
        df.describe(include="all").transpose().to_csv(OUT/(p.stem+"_summary.csv"))
        df.head(1000).to_csv(OUT/(p.stem+"_preview.csv"),index=False)
        summary.append(f"- **{p.name}**: {df.shape[0]} rows, {df.shape[1]} columns.")
        # Generate general exploratory charts only; do not assume a target variable.
        numeric=df.select_dtypes(include="number")
        if not numeric.empty:
            ax=numeric.hist(figsize=(12,8), bins=20)
            plt.tight_layout(); plt.savefig(OUT/(p.stem+"_numeric_distributions.png")); plt.close("all")
        # Loan model only when an approval-like target column is identifiable.
        target=next((c for c in df.columns if any(k in c.lower().replace(" ","_") for k in ["loan_status","loan_approved","loanapproval","approved"])),None)
        if target and df[target].nunique(dropna=True)==2:
            data=df.dropna(subset=[target]).copy()
            X=data.drop(columns=[target]); y=data[target].astype(str)
            if len(data)>=30:
                cat=X.select_dtypes(exclude="number").columns.tolist()
                num=X.select_dtypes(include="number").columns.tolist()
                pre=ColumnTransformer([
                    ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                    ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
                ])
                Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
                model=Pipeline([("preprocess",pre),("model",RandomForestClassifier(n_estimators=200,random_state=42,class_weight="balanced"))])
                model.fit(Xtr,ytr); pred=model.predict(Xte)
                report=f"Dataset: `{p.name}`\n\nRows used: {len(data)}\n\nTarget column: `{target}`\n\nHoldout accuracy: {accuracy_score(yte,pred):.3f}\n\nClassification report:\n\n```\n{classification_report(yte,pred,zero_division=0)}\n```\n\nThis is a baseline model, not a lending decision system. Review fairness, leakage, and class balance before use."
                save_report("Task2_Loan_Report.md","Loan Approval Prediction",report)
    except Exception as e:
        print("Skipped",p.name,":",e)
if not files:
    save_report("DATA_REQUIRED.md","Datasets still required","No dataset files were found in `data/raw`. Upload the official Alfido datasets to run evidence-based analyses.")
else:
    (OUT/"dataset_inventory.md").write_text("# Dataset inventory\n\n"+"\n".join(summary)+"\n",encoding="utf-8")
print("\nDone. Inspect outputs/ and reports/; confirm each file matches its assigned task before submission.")
