"""
One-time download of the two public datasets behind the project's first design.

    Banking77   about 13,000 real customer-service questions with 77 intent labels, saved as
                data/raw/banking77_train.csv and data/raw/banking77_test.csv
                (fetched from the mteb/banking77 mirror on Hugging Face: the original PolyAI/banking77 repository
                uses a legacy loading script that current versions of the datasets library refuse to run)
    FinQA       financial questions with tables and numeric answers, git-cloned into ./FinQA
                (turn it into CSV files with scripts/convert_finqa_to_csv.py)

You do NOT need either dataset to run the pipeline, train the models or use the dashboard: the live system is tested
on the synthetic Bitext-style questions in data/synthetic/. They are only used by the ground-truth helpers in
src/quality.py. Fetch them if you want to repeat the project's earliest experiments.

Run once, from the project root, inside the venv (needs the datasets package, and Git for FinQA):
    python scripts/fetch_data.py
Safe to re-run: anything already downloaded is skipped, and the script says so.
"""
import os
import subprocess
from datasets import load_dataset

DATA_DIR = "data/raw"


def fetch_banking77():
    """Download Banking77 (train and test) to data/raw as CSV from the mteb mirror, skipping it if both
    files exist.
    """
    train_path = f"{DATA_DIR}/banking77_train.csv"
    test_path = f"{DATA_DIR}/banking77_test.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Banking77 already downloaded - skipping (found {train_path})")
        return

    print("Downloading Banking77 from Hugging Face...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # The original PolyAI/banking77 repo is currently in a broken state on
    # Hugging Face's own infrastructure: its legacy loading script blocks
    # the modern datasets library, and that same broken status also blocks
    # HF's own Parquet-resolution API for this specific repo. Rather than
    # route around that (via a library downgrade or hand-built URLs),
    # mteb/banking77 is the same underlying data -- same 3,080-example
    # test split, same 77 labels -- natively stored in modern Parquet
    # format as part of the actively maintained MTEB benchmark suite.
    # No script, no special flags, no version pin required.
    ds = load_dataset("mteb/banking77")

    print(f"Columns found: {ds['test'].column_names}")  # sanity check -- see note below if this differs from ['text', 'label']

    ds["train"].to_pandas().to_csv(train_path, index=False)
    ds["test"].to_pandas().to_csv(test_path, index=False)
    print(f"Saved {train_path} and {test_path}")


def fetch_finqa():
    """git clone FinQA into ./FinQA, skipping it if the folder exists and saying so if Git is not
    installed.
    """
    finqa_dir = "FinQA"

    if os.path.exists(finqa_dir):
        print(f"FinQA already cloned - skipping (found {finqa_dir}/)")
        return

    print("Cloning FinQA from GitHub...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/czyssrs/FinQA.git"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("git clone failed. Is Git installed? (git-scm.com/downloads)")
        print(result.stderr)
    else:
        print(f"Cloned into {finqa_dir}/")


if __name__ == "__main__":
    fetch_banking77()
    fetch_finqa()
    print("\nData acquisition complete.")
