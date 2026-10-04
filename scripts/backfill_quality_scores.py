"""
Retroactively scores answer quality for rows that predate quality scoring, so your existing
results_history.csv (and, for whichever run is still the current "latest," results_log_combined.csv
and the per-unit files too) end up with the same quality columns as any new run scored going
forward -- one consistent set across all 4 result files, old rows included.

Scores with Claude by default (the validated judge -- "the numbers you'll report"). Pass
--judge-local <model> to score with a local Ollama model instead (the cheap, less-validated
first pass) -- this writes the separate quality_local_* columns, never the quality_* ones, so
the two judges are never confused with each other. Run the script twice, once each way, if you
want both sets of columns on the same historical data.

By default this scores final_answer -- what the customer actually received (Claude's answer
for an escalated row, the local model's answer otherwise). Pass --draft to instead score
tier1_answer -- the local model's DISCARDED draft, which only differs from final_answer on
escalated rows. This is how you get a genuine "what would the local model have said, judged
the same way as what Claude actually said" comparison, using data you already have (no new
generation needed) -- writes separate quality_draft_* columns, and is automatically restricted
to escalated rows (on a local row, the draft IS the final answer, so scoring it again would
just duplicate quality_overall for no reason).

Only scores rows missing a value for whichever judge/column-set you're running, so this is
safe to re-run (e.g. after a partial run, or after adding more history) -- already-scored rows
are left untouched, not re-scored and not re-billed.

Run from the project root:
    python scripts/backfill_quality_scores.py                       # Claude, needs ANTHROPIC_API_KEY
    python scripts/backfill_quality_scores.py --judge-local llama3.2:3b   # local, needs Ollama running
    python scripts/backfill_quality_scores.py --draft                # Claude judges the local draft (escalated rows only)
    python scripts/backfill_quality_scores.py --unit retail_bank --run run-29-Sep-26-09:44
    python scripts/backfill_quality_scores.py --limit 200            # cap how many rows this pass
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quality import judge_answer_quality, judge_answer_quality_local, QUALITY_DIMS
from src.pipeline import HistoryLock, guard_writable, LOGS_DIR, HISTORY_PATH
from src.tier2_escalate import client as anthropic_client


def quality_cols(prefix: str) -> list:
    return [f"{prefix}{d}" for d in QUALITY_DIMS] + [f"{prefix}overall", f"{prefix}note"]


def ensure_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            df[col] = None
    return df


def score_missing(df: pd.DataFrame, prefix: str, judge_label: str, judge_one, limit, use_draft: bool) -> tuple:
    """Score every row missing a <prefix>overall value, in place, using judge_one(query, answer)
    -> result dict or None. Returns (df, n_scored)."""
    cols = quality_cols(prefix)
    df = ensure_cols(df, cols)
    to_score = df[df[f"{prefix}overall"].isna()]
    if limit is not None:
        to_score = to_score.head(limit)
    if to_score.empty:
        return df, 0

    print(f"Scoring {len(to_score)} previously-unscored rows ({judge_label} judge, "
          f"{'local draft' if use_draft else 'final answer'})...")
    for i, (idx, row) in enumerate(to_score.iterrows(), 1):
        if use_draft:
            # the discarded local-model draft -- only meaningful on escalated rows (callers
            # restrict to those), since on a local row this is identical to final_answer.
            answer = row.get("tier1_answer")
        else:
            # final_answer, not tier1_answer: for an escalated row, tier1_answer is the
            # discarded low-confidence draft, and final_answer is what Tier 2 actually
            # produced and the customer actually received -- scoring tier1_answer there
            # would score the wrong text. Falls back to tier1_answer only for any row that
            # predates final_answer being logged.
            answer = row.get("final_answer")
            if pd.isna(answer):
                answer = row.get("tier1_answer")
        if pd.isna(answer):
            continue  # nothing to score if no answer was ever logged for this row
        result = judge_one(row["query"], answer)
        if result:
            for dim in QUALITY_DIMS:
                df.at[idx, f"{prefix}{dim}"] = result[dim]
            df.at[idx, f"{prefix}overall"] = round(sum(result[d] for d in QUALITY_DIMS) / len(QUALITY_DIMS), 2)
            df.at[idx, f"{prefix}note"] = result["note"]
        if i % 10 == 0 or i == len(to_score):
            print(f"  {i}/{len(to_score)} scored")
    return df, len(to_score)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", default=None, help="only score this business_unit (default: all)")
    ap.add_argument("--run", default=None, help="only score this run_id (default: all runs in history)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of rows scored this pass")
    ap.add_argument("--judge-local", default=None, metavar="MODEL",
                     help="score with this local Ollama model instead of Claude, writing quality_local_* columns")
    ap.add_argument("--draft", action="store_true",
                     help="score the local model's discarded draft (tier1_answer) instead of final_answer -- "
                          "writes quality_draft_* columns, auto-restricted to escalated rows")
    args = ap.parse_args()

    if args.judge_local:
        judge_label = f"local/{args.judge_local}"
        judge_one = lambda q, a: judge_answer_quality_local(args.judge_local, q, a)
        prefix = "quality_local_draft_" if args.draft else "quality_local_"
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        judge_label = "claude"
        judge_one = lambda q, a: judge_answer_quality(anthropic_client, q, a)
        prefix = "quality_draft_" if args.draft else "quality_"

    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"{HISTORY_PATH} not found -- nothing to backfill")

    cols = quality_cols(prefix)
    combined_path = f"{LOGS_DIR}/results_log_combined.csv"
    guard_writable([p for p in [combined_path] if os.path.exists(p)])

    with HistoryLock(HISTORY_PATH) as lock:
        print(f"Locked {HISTORY_PATH} for the backfill.")
        history = lock.read()
        mask = pd.Series(True, index=history.index)
        if args.unit:
            mask &= history["business_unit"] == args.unit
        if args.run:
            mask &= history["run_id"] == args.run
        if args.draft:
            # on a local row, tier1_answer IS final_answer -- scoring it again would just
            # duplicate quality_overall for no reason, so --draft only ever applies here.
            n_before = mask.sum()
            mask &= history["decision"] == "ESCALATE"
            skipped = n_before - mask.sum()
            if skipped:
                print(f"--draft: restricting to escalated rows only ({skipped} local rows excluded -- "
                      f"their draft and final answer are identical, already scored).")

        subset = history[mask]
        rest = history[~mask]
        scored_subset, n_scored = score_missing(subset, prefix, judge_label, judge_one, args.limit, args.draft)
        history = pd.concat([rest, scored_subset]).sort_index()

        if n_scored == 0:
            print("Nothing to score (all matching rows already have quality scores, or none matched the filters).")
        else:
            lock.write(history)
            print(f"\nScored {n_scored} rows. Updated {HISTORY_PATH}.")

    if n_scored > 0 and os.path.exists(combined_path):
        combined = pd.read_csv(combined_path)
        combined = ensure_cols(combined, cols)
        if "run_id" in combined.columns:
            for run_id in combined["run_id"].unique():
                hist_rows = history[history["run_id"] == run_id].reset_index(drop=True)
                comb_mask = combined["run_id"] == run_id
                comb_rows = combined[comb_mask]
                if len(hist_rows) != len(comb_rows):
                    print(f"  WARNING: row count for '{run_id}' differs between history ({len(hist_rows)}) "
                          f"and {combined_path} ({len(comb_rows)}) -- skipping this run here to avoid misaligned scores.")
                    continue
                for col in cols:
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
