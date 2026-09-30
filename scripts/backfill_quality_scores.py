"""
Retroactively scores answer quality for rows that predate the --score-quality
pipeline flag, so your existing results_history.csv (and, for whichever run is
still the current "latest," results_log_combined.csv and the per-unit files
too) end up with the same quality_* columns as any new run scored going
forward -- one consistent set of columns across all 4 result files, old rows
included.

Only scores rows that don't already have a quality_overall value, so this is
safe to re-run (e.g. after a partial run, or after adding more history) --
already-scored rows are left untouched, not re-scored and not re-billed.

Run from the project root, with ANTHROPIC_API_KEY set:
    python scripts/backfill_quality_scores.py
    python scripts/backfill_quality_scores.py --unit retail_bank --run run-29-Sep-26-09:44
    python scripts/backfill_quality_scores.py --limit 200   # cap how many rows to score this pass
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality import judge_answer_quality, QUALITY_DIMS
from src.pipeline import HistoryLock, guard_writable, LOGS_DIR, HISTORY_PATH
from src.tier2_escalate import client as anthropic_client

QUALITY_COLS = [f"quality_{d}" for d in QUALITY_DIMS] + ["quality_overall", "quality_note"]


def ensure_quality_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in QUALITY_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def score_missing(df: pd.DataFrame, limit: int | None) -> tuple[pd.DataFrame, int]:
    """Score every row with a missing quality_overall, in place. Returns (df, n_scored)."""
    df = ensure_quality_cols(df)
    to_score = df[df["quality_overall"].isna()]
    if limit is not None:
        to_score = to_score.head(limit)
    if to_score.empty:
        return df, 0

    print(f"Scoring {len(to_score)} previously-unscored rows...")
    for i, (idx, row) in enumerate(to_score.iterrows(), 1):
        answer = row.get("tier1_answer")
        if pd.isna(answer):
            continue  # nothing to score if the answer itself was never logged
        result = judge_answer_quality(anthropic_client, row["query"], answer)
        if result:
            for dim in QUALITY_DIMS:
                df.at[idx, f"quality_{dim}"] = result[dim]
            df.at[idx, "quality_overall"] = round(sum(result[d] for d in QUALITY_DIMS) / len(QUALITY_DIMS), 2)
            df.at[idx, "quality_note"] = result["note"]
        if i % 10 == 0 or i == len(to_score):
            print(f"  {i}/{len(to_score)} scored")
    return df, len(to_score)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", default=None, help="only score this business_unit (default: all)")
    ap.add_argument("--run", default=None, help="only score this run_id (default: all runs in history)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of rows scored this pass")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"{HISTORY_PATH} not found -- nothing to backfill")

    combined_path = f"{LOGS_DIR}/results_log_combined.csv"
    guard_writable([p for p in [combined_path] if os.path.exists(p)])

    # results_history.csv is the source of truth for "past runs" -- results_log_combined.csv
    # and the per-unit files only ever hold the LATEST run, so backfilling those only makes
    # sense for whichever run_id is still current in them.
    with HistoryLock(HISTORY_PATH) as lock:
        print(f"Locked {HISTORY_PATH} for the backfill.")
        history = lock.read()
        if args.unit:
            history = history  # filter applied at the mask level below, not by dropping rows
        mask = pd.Series(True, index=history.index)
        if args.unit:
            mask &= history["business_unit"] == args.unit
        if args.run:
            mask &= history["run_id"] == args.run

        subset = history[mask]
        rest = history[~mask]
        scored_subset, n_scored = score_missing(subset, args.limit)
        history = pd.concat([rest, scored_subset]).sort_index()

        if n_scored == 0:
            print("Nothing to score (all matching rows already have quality scores, or none matched the filters).")
        else:
            lock.write(history)
            print(f"\nScored {n_scored} rows. Updated {HISTORY_PATH}.")

    # Also update results_log_combined.csv and the per-unit files, for whichever run_id(s)
    # they currently hold, so all 4 files stay consistent for the latest run. Aligned
    # POSITIONALLY within each run_id (not merged on query text), since two sampled queries
    # could in principle be identical text -- a text-based join could silently duplicate
    # rows in that case. Both files were originally written from the same per-run dataframe,
    # so row order within a run_id is expected to match; row-count is checked before trusting
    # that alignment, and a run is skipped here (with a clear warning) rather than risk
    # writing misaligned quality scores onto the wrong rows.
    if n_scored > 0 and os.path.exists(combined_path):
        combined = pd.read_csv(combined_path)
        combined = ensure_quality_cols(combined)
        if "run_id" in combined.columns:
            for run_id in combined["run_id"].unique():
                hist_rows = history[history["run_id"] == run_id].reset_index(drop=True)
                comb_mask = combined["run_id"] == run_id
                comb_rows = combined[comb_mask]
                if len(hist_rows) != len(comb_rows):
                    print(f"  WARNING: row count for '{run_id}' differs between history ({len(hist_rows)}) "
                          f"and {combined_path} ({len(comb_rows)}) -- skipping this run here to avoid misaligned scores.")
                    continue
                for col in QUALITY_COLS:
                    combined.loc[comb_mask, col] = hist_rows[col].values
            combined.to_csv(combined_path, index=False)
            print(f"Also updated {combined_path}.")
            for unit, group in combined.groupby("business_unit"):
                unit_path = f"{LOGS_DIR}/results_log_{unit}.csv"
                if os.path.exists(unit_path):
                    group.to_csv(unit_path, index=False)
                    print(f"Also updated {unit_path}.")


if __name__ == "__main__":
    main()
