"""
One-time data acquisition script: downloads Banking77 (via Hugging Face
datasets) and clones FinQA (via git), saving everything into data/raw/.

Run once, from the project root:
    python scripts/fetch_data.py
"""
import os
import subprocess
from datasets import load_dataset

DATA_DIR = "data/raw"


def fetch_banking77():
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
