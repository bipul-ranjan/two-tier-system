"""
Removes personal identifiers from the text already stored in your result logs, using the same filter the pipeline
now applies to every new row (src/privacy.py).

Why: rows logged before the filter existed hold the question and answers exactly as typed. If those files are
copied, shared, or sent to Claude for marking, the identifiers go with them. This script replaces them with
placeholders such as [CARD_NUMBER] in the text columns, and fills the audit columns pii_filter, pii_count and
pii_types so every row says whether it was filtered.

By default it only REPORTS what it would change (a dry run). Add --apply to write the changes. Before writing,
it copies every file it will change into results/logs/backup-redact-<time>/ so nothing is lost. Do not run it
while a pipeline run is in progress.

Files covered: results_history.csv, results_log_combined.csv, results_log_payments.csv and
results_log_retail_bank.csv, in the logs folder. Columns redacted: query, tier1_answer, final_answer,
cache_matched_query, and the quality notes (they can quote the text).

Note that redaction changes the text the router learns from. After applying it, retrain the router
(python scripts/train_text_router.py) so it reads the same kind of text it will see in production.

Usage:
    python scripts/redact_history.py              # dry run: counts only
    python scripts/redact_history.py --apply      # redact in place, with a backup
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.privacy import redact, FILTER_VERSION, ANSWER_TYPES
from src.pipeline import HistoryLock, guard_writable, LOGS_DIR, HISTORY_PATH

TEXT_COLS = ["query", "tier1_answer", "final_answer", "cache_matched_query", "quality_note", "quality_draft_note",
             "quality_local_note", "quality_local_draft_note"]
FILES = ["results_log_combined.csv", "results_log_payments.csv", "results_log_retail_bank.csv"]


QUESTION_COLS = ["query", "cache_matched_query"]


def redact_frame(df: pd.DataFrame):
    """Return a redacted copy of a log DataFrame, plus a dictionary of how many identifiers of each type were found.
    Questions are filtered in full; answers and notes only for customer-identifier types and for values the same
    row's question contained (see src/privacy.py)."""
    out = df.copy()
    totals: dict = {}
    n = len(out)
    per_row_count, per_row_types = [0] * n, [set() for _ in range(n)]
    present = [c for c in TEXT_COLS if c in out.columns]
    new = {c: list(out[c].tolist()) for c in present}
    for i in range(n):
        q_known = []
        if "query" in out.columns and isinstance(out["query"].iat[i], str):
            qr = redact(out["query"].iat[i])
            q_known = qr.originals
        for c in present:
            v = new[c][i]
            if not isinstance(v, str):
                continue
            r = redact(v) if c in QUESTION_COLS else redact(v, only=ANSWER_TYPES, known=q_known)
            for k, cnt in r.findings.items():
                totals[k] = totals.get(k, 0) + cnt
                per_row_types[i].add(k)
            per_row_count[i] += r.count
            new[c][i] = r.text
    for c in present:
        out[c] = new[c]
    # Rows the pipeline already filtered keep their own audit values; older rows get them now.
    done = out["pii_filter"].notna() if "pii_filter" in out.columns else pd.Series(False, index=out.index)
    filt = out["pii_filter"] if "pii_filter" in out.columns else pd.Series(None, index=out.index, dtype=object)
    cnt = out["pii_count"] if "pii_count" in out.columns else pd.Series(None, index=out.index, dtype=object)
    typ = out["pii_types"] if "pii_types" in out.columns else pd.Series(None, index=out.index, dtype=object)
    out["pii_filter"] = [f if d else f"{FILTER_VERSION}-retro" for f, d in zip(filt, done)]
    out["pii_count"] = [c if d else x for c, d, x in zip(cnt, done, per_row_count)]
    out["pii_types"] = [t if d else ",".join(sorted(x)) for t, d, x in zip(typ, done, per_row_types)]
    return out, totals


def main() -> None:
    """Report (or apply) the redaction of every result log file."""
    ap = argparse.ArgumentParser(description="Remove personal identifiers from stored result logs.")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run, report only)")
    args = ap.parse_args()

    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(f"{HISTORY_PATH} not found -- nothing to redact")
    others = [os.path.join(LOGS_DIR, f) for f in FILES if os.path.exists(os.path.join(LOGS_DIR, f))]
    guard_writable(others)

    with HistoryLock(HISTORY_PATH) as lock:
        hist = lock.read()
        new_hist, totals = redact_frame(hist)
        rows_changed = int(sum(1 for a, b in zip(hist.get("query", []), new_hist.get("query", [])) if a != b))
        print(f"{HISTORY_PATH}: {len(hist)} rows; {sum(totals.values())} identifiers found "
              f"({', '.join(f'{k}={v}' for k, v in sorted(totals.items())) or 'none'}); {rows_changed} questions would change.")
        if not args.apply:
            print("Dry run only. Re-run with --apply to write the changes (a backup is made first).")
            return
        backup = os.path.join(LOGS_DIR, "backup-redact-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(backup, exist_ok=True)
        for path in [HISTORY_PATH] + others:
            shutil.copy2(path, backup)
        lock.write(new_hist)
        for path in others:
            df = pd.read_csv(path)
            red, _ = redact_frame(df)
            red.to_csv(path, index=False)
        print(f"Applied. Originals copied to {backup}. Retrain the router next: python scripts/train_text_router.py")


if __name__ == "__main__":
    main()
