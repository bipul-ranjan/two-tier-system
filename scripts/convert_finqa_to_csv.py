"""
Converts the FinQA JSON files (train.json, dev.json, test.json) into flat CSV files with one row per question.

FinQA's raw files are deeply nested (pre_text, post_text, table, and a qa sub-object). This flattens each record
into one row so it can be loaded and filtered like every other file in data/raw/: finqa_train.csv, finqa_dev.csv
and finqa_test.csv. The table and gold_inds columns keep their structure as JSON strings, so json.loads() restores
them. Like fetch_data.py, this is only needed to repeat the project's earliest experiments, not to run the system.

Run once, after scripts/fetch_data.py has cloned FinQA, from the project root:
    python scripts/convert_finqa_to_csv.py

No "-m" is needed: the script imports only the standard library and pandas, never a sibling file. Safe to re-run:
a CSV that already exists is skipped.
"""
import os
import json
import pandas as pd

FINQA_DIR = "FinQA/dataset"
OUT_DIR = "data/raw"

SPLITS = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
}


def convert_split(split_name: str, filename: str):
    """Flatten one FinQA JSON split into data/raw/finqa_<split>.csv. Skips it if the CSV exists, and
    says so if the JSON has not been downloaded yet.
    """
    out_path = f"{OUT_DIR}/finqa_{split_name}.csv"

    if os.path.exists(out_path):
        print(f"finqa_{split_name}.csv already exists - skipping (found {out_path})")
        return

    in_path = f"{FINQA_DIR}/{filename}"
    if not os.path.exists(in_path):
        print(f"Cannot find {in_path} -- run scripts/fetch_data.py first to clone FinQA.")
        return

    print(f"Converting {in_path}...")
    with open(in_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for rec in records:
        qa = rec.get("qa", {})
        rows.append({
            "id": rec.get("id"),
            "pre_text": " ".join(rec.get("pre_text", [])),
            "table": json.dumps(rec.get("table", [])),
            "post_text": " ".join(rec.get("post_text", [])),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "exe_ans": qa.get("exe_ans"),
            "program": qa.get("program"),
            "gold_inds": json.dumps(qa.get("gold_inds", {})),
        })

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    for split_name, filename in SPLITS.items():
        convert_split(split_name, filename)
    print("\nFinQA conversion complete.")
