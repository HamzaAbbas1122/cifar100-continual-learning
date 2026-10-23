

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_scripts.download_dataset import main as run_download
from data_scripts.clean_validate import main as run_clean_validate
from data_scripts.preprocess_eda import main as run_preprocess_eda

def main():
    print("=" * 65)
    print("STEP 1: Downloading & Verifying CIFAR-100 Dataset")
    print("=" * 65)
    run_download()

    print("\n" + "=" * 65)
    print("STEP 2: Cleaning & Validating Dataset Integrity")
    print("=" * 65)
    run_clean_validate()

    print("\n" + "=" * 65)
    print("STEP 3: Task Splitting, Preprocessing, and EDA Visualizations")
    print("=" * 65)
    run_preprocess_eda()

    print("\n" + "=" * 65)
    print("All steps finished successfully!")
    print("Artifacts generated:")
    print(" - data/raw/collection_log.json")
    print(" - data/processed/cleaning_validation_report.json")
    print(" - data/processed/task_mapping.json")
    print(" - reports/figures/ (4 plots)")
    print("=" * 65)

if __name__ == "__main__":
    main()
