"""
Fetches the Bitext retail-banking chatbot dataset and splits it into
payments/ and retail_bank/ training files by category -- saved in both
JSONL (for fine-tuning tools that expect it, e.g. Unsloth/TRL) and CSV
(for quick inspection with pandas/Excel).

This dataset already contains 25,545 real instruction/response pairs
(~1,000 per intent) with full, detailed, good-quality responses -- no
synthetic data generation needed.

License: CDLA-Sharing 1.0 -- free to use, requires attribution, and any
derivative data you share must stay under the same license. Cite:
Bitext Innovations, "Bitext-retail-banking-llm-chatbot-training-dataset", 2024.

Run once, from the project root:
    python scripts/fetch_bitext_banking.py
"""
import os
import json
import pandas as pd
from datasets import load_dataset

OUT_DIR = "data/raw"

# Category -> business unit mapping, derived directly from the dataset's
# own category taxonomy -- no synthetic labels invented.
PAYMENTS_CATEGORIES = {"CARD", "TRANSFER", "ATM", "FEES"}
RETAIL_BANK_CATEGORIES = {"ACCOUNT", "LOAN", "PASSWORD", "CONTACT", "FIND"}


def write_jsonl(rows: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(rows: list, path: str):
    pd.DataFrame(rows).to_csv(path, index=False)


def fetch_and_split():
    payments_jsonl = f"{OUT_DIR}/bitext_payments.jsonl"
    payments_csv = f"{OUT_DIR}/bitext_payments.csv"
    retail_jsonl = f"{OUT_DIR}/bitext_retail_bank.jsonl"
    retail_csv = f"{OUT_DIR}/bitext_retail_bank.csv"

    if all(os.path.exists(p) for p in [payments_jsonl, payments_csv, retail_jsonl, retail_csv]):
        print("Bitext data already split - skipping (found existing files)")
        return

    print("Downloading Bitext retail-banking dataset...")
    ds = load_dataset("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", split="train")
    print(f"Loaded {len(ds)} total records")

    os.makedirs(OUT_DIR, exist_ok=True)
    payments_rows = []
    retail_rows = []
    unmatched_categories = set()

    for row in ds:
        category = row["category"]
        record = {
            "instruction": row["instruction"],
            "category": category,
            "intent": row["intent"],
            "response": row["response"],
        }
        if category in PAYMENTS_CATEGORIES:
            payments_rows.append(record)
        elif category in RETAIL_BANK_CATEGORIES:
            retail_rows.append(record)
        else:
            unmatched_categories.add(category)

    write_jsonl(payments_rows, payments_jsonl)
    write_csv(payments_rows, payments_csv)
    write_jsonl(retail_rows, retail_jsonl)
    write_csv(retail_rows, retail_csv)

    print(f"Saved {payments_jsonl} and {payments_csv} ({len(payments_rows)} rows)")
    print(f"Saved {retail_jsonl} and {retail_csv} ({len(retail_rows)} rows)")
    if unmatched_categories:
        print(f"WARNING: found categories not in either mapping: {unmatched_categories}")


if __name__ == "__main__":
    fetch_and_split()
    print("\nBitext dataset fetch and split complete.")
