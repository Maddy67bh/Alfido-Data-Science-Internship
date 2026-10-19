from pathlib import Path
import requests, subprocess, sys

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Public practice datasets; these are NOT confirmed official Alfido datasets.
sources = {
    "zomato.csv": [
        ("sivalatha/Zomato-Dataset-EDA", "zomato.csv"),
    ],
    "loan_approval_dataset.csv": [
        ("saiteja18/loan-approval-prediction", "loan_approval_dataset.csv"),
        ("sabin74/Loan-Approval-Prediction", "loan_approval_dataset.csv"),
    ],
    "instagram_analytics.csv": [
        ("CodeMaverick-143/Instagram_Analytics", "Instagram_Analytics.csv"),
        ("aishsingh91/Instagram-analytics-", "instagram_data.csv"),
    ],
    "business_sales_customers.csv": [
        ("CoreLogicLabs/E-Commerce-Sales-Customer-Analytics",
         "ecommerce_orders_dataset.csv"),
    ],
}

session = requests.Session()
session.headers["User-Agent"] = "Alfido-Internship-Student-Project"

def download_repo_csv(repo, expected_name):
    try:
        meta = session.get(
            f"https://api.github.com/repos/{repo}", timeout=20
        )
        if not meta.ok:
            return None
        branch = meta.json().get("default_branch", "main")
        tree = session.get(
            f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1",
            timeout=30
        )
        if not tree.ok:
            return None
        entries = tree.json().get("tree", [])
        candidates = [
            x for x in entries
            if x.get("type") == "blob"
            and x["path"].lower().endswith(".csv")
            and Path(x["path"]).name.lower() == expected_name.lower()
        ]
        if not candidates:
            candidates = [
                x for x in entries
                if x.get("type") == "blob"
                and x["path"].lower().endswith(".csv")
                and any(k in Path(x["path"]).name.lower()
                        for k in ("zomato", "loan_approval", "instagram_analytics",
                                  "ecommerce_orders"))
            ]
        for item in candidates:
            url = (
                f"https://raw.githubusercontent.com/{repo}/"
                f"{branch}/{item['path']}"
            )
            r = session.get(url, timeout=45)
            if r.ok and len(r.content) > 1000:
                return r.content
    except Exception as e:
        print(f"  Could not check {repo}: {e}")
    return None

for target, candidates in sources.items():
    destination = RAW / target
    if destination.exists() and destination.stat().st_size > 1000:
        print(f"Already present: {target}")
        continue
    found = False
    for repo, expected in candidates:
        print(f"Looking for {expected} in {repo} ...")
        content = download_repo_csv(repo, expected)
        if content:
            destination.write_bytes(content)
            print(f"Downloaded {target} ({len(content):,} bytes)")
            found = True
            break
    if not found:
        print(f"NOT FOUND: {target}")

print("\nDataset files:", [p.name for p in RAW.glob("*") if p.is_file()])

# Run the generic analysis script if present.
analysis = ROOT / "complete_available.py"
if analysis.exists():
    result = subprocess.run([sys.executable, str(analysis)])
    if result.returncode:
        print("Analysis script reported an error.")

# Execute notebooks and preserve output.
runner = ROOT / "run_all_alfido.sh"
if runner.exists():
    subprocess.run(["bash", str(runner)])

print("\nCheck every output and report before submitting.")
print("Public practice datasets may not match Alfido's official task data.")
