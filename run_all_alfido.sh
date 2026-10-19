#!/usr/bin/env bash
set -u

ROOT="/workspaces/Alfido-Data-Science-Internship"
cd "$ROOT" || exit 1

echo "========================================"
echo " ALFIDO INTERNSHIP - ALL TASKS CHECK"
echo "========================================"

mkdir -p incoming data/raw outputs reports notebooks

echo
echo "1. Checking for uploaded ZIP files..."
find incoming . -maxdepth 2 -type f \
  \( -iname '*.zip' -o -iname '*.csv' -o -iname '*.xlsx' -o -iname '*.json' \) \
  -not -path './.git/*' -print 2>/dev/null | sort -u

# Extract a ZIP only if it has been uploaded into incoming.
for z in incoming/*.zip; do
  [ -f "$z" ] || continue
  echo
  echo "Inspecting ZIP: $z"
  unzip -l "$z" | head -40
  echo "Extracting into incoming/unpacked..."
  mkdir -p incoming/unpacked
  unzip -oq "$z" -d incoming/unpacked
done

echo
echo "2. Searching for datasets..."
find incoming data/raw -type f \
  \( -iname '*.csv' -o -iname '*.xlsx' -o -iname '*.xls' -o -iname '*.json' \) \
  -print 2>/dev/null

echo
echo "3. Checking Python dependencies..."
python - <<'PY'
import importlib.util
for package in ["pandas", "numpy", "matplotlib", "sklearn", "openpyxl", "nbformat", "nbclient"]:
    print(f"{package}: {'available' if importlib.util.find_spec(package) else 'MISSING'}")
PY

python -m pip install -q nbformat nbclient ipykernel

echo
echo "4. Checking notebook files..."
find notebooks -maxdepth 1 -name '*.ipynb' -print | sort

echo
echo "5. Running notebooks (missing datasets will be reported)..."
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
for path in sorted((root / "notebooks").glob("*.ipynb")):
    print(f"\n--- {path.name} ---")
    try:
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb, timeout=300, kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
            allow_errors=True
        )
        client.execute()
        errors = []
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(output.get("ename", "Error") + ": " +
                                  output.get("evalue", ""))
        nbformat.write(nb, path)
        if errors:
            print("Completed with issues:")
            for err in errors:
                print(" -", err)
        else:
            print("Notebook executed; inspect outputs for missing-data messages.")
    except Exception as e:
        print("Could not execute:", str(e)[:500])

print("\nExecution attempt finished. Verify all notebook outputs before submission.")
PY

echo
echo "6. Final project status..."
echo "Datasets currently in data/raw:"
find data/raw -type f -print 2>/dev/null
echo
git status --short
echo
echo "NOTE: No dataset means no genuine analysis results."
echo "Task 4 topic and official requirements still need verification."
