"""
Converts the FinQA JSON files (train.json, dev.json, test.json) into flat
CSV files, one row per question, suitable for feeding into the Tier 2
escalation-testing pipeline.

Run once, after scripts/fetch_data.py has cloned FinQA:
    python scripts/convert_finqa_to_csv.py

Note: no "-m" needed here, same reasoning as fetch_data.py -- this
script only imports the standard library (json, os, csv via pandas),
never a sibling file, so it runs directly.

Safe to re-run -- skips any CSV that already exists.
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
