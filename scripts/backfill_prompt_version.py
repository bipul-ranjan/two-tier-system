"""
One-off: tags existing escalated rows in results_history.csv with which Tier 2 system prompt
was actually in effect, for runs that predate tier2_prompt_version being logged automatically
(src/tier2_escalate.py now logs it on every new escalation -- this just backfills the history
that came before that).

CUTOFF_RUN_ID is the first run_id that used the new persona prompt (see PROMPT_VERSION in
src/tier2_escalate.py) -- every escalated row in a run at or after this one is tagged
"v2-persona", everything earlier is tagged "v1-generic". Edit CUTOFF_RUN_ID if you ever need
to re-run this for a different prompt change.

Only touches rows where tier2_prompt_version is still blank, so safe to re-run.

Run from the project root:
    python scripts/backfill_prompt_version.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import HistoryLock, guard_writable, LOGS_DIR, HISTORY_PATH
from src.tier2_escalate import PROMPT_VERSION

CUTOFF_RUN_ID = "run-01-Oct-26-08:36 AM"  # first run with the new persona prompt
OLD_VERSION = "v1-generic"


def tag(history: pd.DataFrame) -> pd.DataFrame:
    """Fill tier2_prompt_version for escalated rows that lack it: v1-generic for runs before
    CUTOFF_RUN_ID, v2-persona from it onwards. Rows that are already tagged are left alone.
    """
    if "tier2_prompt_version" not in history.columns:
        history["tier2_prompt_version"] = None

    # Chronological order within results_history.csv is first-appearance order (how
    # run_pipeline always appends), so a run's position in that order tells us before/at/after
    # the cutoff even though run_id itself is just a timestamp string, not directly sortable.
    run_order = list(dict.fromkeys(history["run_id"]))
    if CUTOFF_RUN_ID not in run_order:
        raise ValueError(f"CUTOFF_RUN_ID '{CUTOFF_RUN_ID}' not found in {HISTORY_PATH}. "
                          f"Available run_ids: {run_order}")
    cutoff_index = run_order.index(CUTOFF_RUN_ID)
    runs_before = set(run_order[:cutoff_index])
    runs_at_or_after = set(run_order[cutoff_index:])

    blank = history["tier2_prompt_version"].isna() & (history["decision"] == "ESCALATE")
    before_mask = blank & history["run_id"].isin(runs_before)
    after_mask = blank & history["run_id"].isin(runs_at_or_after)

    history.loc[before_mask, "tier2_prompt_version"] = OLD_VERSION
    history.loc[after_mask, "tier2_prompt_version"] = PROMPT_VERSION

    print(f"Tagged {before_mask.sum()} escalated rows as '{OLD_VERSION}' (before {CUTOFF_RUN_ID}).")
    print(f"Tagged {after_mask.sum()} escalated rows as '{PROMPT_VERSION}' (at/after {CUTOFF_RUN_ID}).")
    return history


def main():
    """Tag the history under its lock, and write it back only if something changed."""
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"{HISTORY_PATH} not found -- nothing to backfill")

    combined_path = f"{LOGS_DIR}/results_log_combined.csv"
    guard_writable([p for p in [combined_path] if os.path.exists(p)])

    with HistoryLock(HISTORY_PATH) as lock:
        print(f"Locked {HISTORY_PATH}.")
        history = lock.read()
        n_before = history["tier2_prompt_version"].notna().sum() if "tier2_prompt_version" in history.columns else 0
        history = tag(history)
        n_after = history["tier2_prompt_version"].notna().sum()
        if n_after == n_before:
            print("Nothing to tag (already tagged, or no escalated rows).")
        else:
            lock.write(history)
            print(f"Updated {HISTORY_PATH}.")

    # Propagate to results_log_combined.csv and the per-unit files, same positional-alignment
    # safety pattern as backfill_quality_scores.py -- skip (with a warning) any run whose row
    # count doesn't match, rather than risk writing a tag onto the wrong row.
    if os.path.exists(combined_path):
        combined = pd.read_csv(combined_path)
        if "tier2_prompt_version" not in combined.columns:
            combined["tier2_prompt_version"] = None
        if "run_id" in combined.columns:
            changed = False
            for run_id in combined["run_id"].unique():
                hist_rows = history[history["run_id"] == run_id].reset_index(drop=True)
                comb_mask = combined["run_id"] == run_id
                comb_rows = combined[comb_mask]
                if len(hist_rows) != len(comb_rows):
                    print(f"  WARNING: row count for '{run_id}' differs between history ({len(hist_rows)}) "
                          f"and {combined_path} ({len(comb_rows)}) -- skipping this run here.")
                    continue
                combined.loc[comb_mask, "tier2_prompt_version"] = hist_rows["tier2_prompt_version"].values
                changed = True
            if changed:
                combined.to_csv(combined_path, index=False)
                print(f"Also updated {combined_path}.")
                for unit, group in combined.groupby("business_unit"):
                    unit_path = f"{LOGS_DIR}/results_log_{unit}.csv"
                    if os.path.exists(unit_path):
                        group.to_csv(unit_path, index=False)
                        print(f"Also updated {unit_path}.")


if __name__ == "__main__":
    main()
