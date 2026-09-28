"""
One-off: add a run_id column to the results logs that already exist (from before
run ids were introduced), and seed results/logs/results_history.csv with them, so
your current results are kept as the first entry in the run history.

Run it ONCE, before the next pipeline run:
    python scripts/backfill_run_id.py                         # uses "run-default"
    python scripts/backfill_run_id.py "run-28-Sep-26-06:29 PM"   # or any id you choose

Safe to re-run: files that already have a run_id are left alone, and the history
file is only created if it does not exist yet.
"""
import os
import sys
import pandas as pd

LOGS_DIR = "results/logs"
FILES = ["results_log_combined.csv", "results_log_payments.csv", "results_log_retail_bank.csv"]
HISTORY = f"{LOGS_DIR}/results_history.csv"
DEFAULT_RUN_ID = "run-default"


def main():
    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_ID
    print(f"Run id for existing records: {run_id}\n")

    for name in FILES:
        path = f"{LOGS_DIR}/{name}"
        if not os.path.exists(path):
            print(f"{name}: not found - skipped")
            continue
        df = pd.read_csv(path)
        if "run_id" in df.columns:
            print(f"{name}: already has run_id - left alone")
            continue
        df.insert(0, "run_id", run_id)
        df.to_csv(path, index=False)
        print(f"{name}: run_id added to {len(df)} rows")

    combined = f"{LOGS_DIR}/results_log_combined.csv"
    if os.path.exists(HISTORY):
        print(f"\n{HISTORY} already exists - left alone")
    elif os.path.exists(combined):
        pd.read_csv(combined).to_csv(HISTORY, index=False)
        print(f"\n{HISTORY} created from the existing results")
    else:
        print("\nNo existing results found, so there is nothing to seed the history with.")


if __name__ == "__main__":
    main()
