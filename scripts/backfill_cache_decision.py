"""
One-off: relabel rows already in your logs that were served from the semantic cache.

The pipeline now logs decision = CACHE for a cache hit (every row is exactly one of LOCAL,
CACHE or ESCALATE). Cache hits logged before that change say ESCALATE with cache_hit = True;
this changes those to CACHE in results_history.csv and the combined / per-unit logs, so the
log reads the same everywhere. The dashboard already treats the old form as CACHE, so nothing
breaks if you skip this -- it just keeps the stored column consistent.

The rule only looks at each row's own columns (ESCALATE + cache_hit True -> CACHE), so it is
applied to each file independently and is safe to re-run: rows already labelled CACHE, and
every real ESCALATE row, are left alone.

Run from the project root:
    python scripts/backfill_cache_decision.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import HistoryLock, guard_writable, LOGS_DIR, HISTORY_PATH


def relabel(df: pd.DataFrame) -> int:
    """ESCALATE + cache_hit True -> CACHE, in place. Returns how many rows changed."""
    if "cache_hit" not in df.columns or "decision" not in df.columns:
        return 0
    m = (df["decision"] == "ESCALATE") & df["cache_hit"].astype(str).str.lower().eq("true")
    df.loc[m, "decision"] = "CACHE"
    return int(m.sum())


def main():
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"{HISTORY_PATH} not found -- nothing to relabel")

    other_logs = [p for p in sorted(glob.glob(f"{LOGS_DIR}/results_log_*.csv")) if os.path.getsize(p) > 0]
    guard_writable(other_logs)

    with HistoryLock(HISTORY_PATH) as lock:
        history = lock.read()
        n = relabel(history)
        if n:
            lock.write(history)
        print(f"{HISTORY_PATH}: {n} cache rows relabelled ESCALATE -> CACHE" if n else f"{HISTORY_PATH}: nothing to change")

    for path in other_logs:
        df = pd.read_csv(path)
        n = relabel(df)
        if n:
            df.to_csv(path, index=False)
        print(f"{path}: {n} rows relabelled" if n else f"{path}: nothing to change")


if __name__ == "__main__":
    main()
