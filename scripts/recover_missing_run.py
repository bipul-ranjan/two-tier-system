"""
One-off: recover a run that completed and wrote results_log_combined.csv (and the
per-unit logs) but never made it into results_history.csv -- typically because
results_history.csv was open in Excel or another program at the moment the pipeline
tried to save it (a Windows file-lock PermissionError), which happens at the very
last step, after every query has already run.

Run it once, after closing whatever program had the history file open:
    python scripts/recover_missing_run.py

Safe to re-run: if the run in results_log_combined.csv is already present in
results_history.csv (by run_id), nothing is changed.
"""
import os
import pandas as pd

LOGS_DIR = "results/logs"
COMBINED = f"{LOGS_DIR}/results_log_combined.csv"
HISTORY = f"{LOGS_DIR}/results_history.csv"


def main():
    """Append the run in results_log_combined.csv to results_history.csv if its run_id is missing
    there, which happens when the history file was open elsewhere as the run finished.
    """
    if not os.path.exists(COMBINED):
        print(f"{COMBINED} not found -- nothing to recover.")
        return

    latest = pd.read_csv(COMBINED)
    if "run_id" not in latest.columns or latest.empty:
        print(f"{COMBINED} has no run_id column or is empty -- nothing to recover.")
        return
    run_id = latest["run_id"].iloc[0]

    if os.path.exists(HISTORY):
        history = pd.read_csv(HISTORY)
        if "run_id" in history.columns and run_id in set(history["run_id"]):
            print(f"'{run_id}' is already in {HISTORY} -- nothing to do.")
            return
        merged = pd.concat([history, latest], ignore_index=True)
    else:
        merged = latest

    try:
        merged.to_csv(HISTORY, index=False)
    except PermissionError as e:
        print(f"Still cannot write to {HISTORY} -- is it still open somewhere? ({e})")
        return

    print(f"Recovered '{run_id}' ({len(latest)} rows) into {HISTORY}.")
    print(f"{HISTORY} now has {merged['run_id'].nunique()} run(s).")


if __name__ == "__main__":
    main()
