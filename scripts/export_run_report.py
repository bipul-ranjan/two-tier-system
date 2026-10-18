"""
Shows one pipeline run as a plain table, with a summary and a list of problems, and saves it as a CSV and an Excel file.

Why you may want this: the pipeline prints a long block of text for every question, which is hard to read, share or check.
The same information is already saved in results/logs/results_history.csv; this script lays it out as one row per question.

Two sources:
  * the saved log (default). The complete record of every row that was saved, with the full question and answers, the
    router's prediction, the decision, the timings and the quality marks.
  * pasted console output (--console FILE). It is only a printout, but it is the one place a row that FAILED still appears.
    A row that fails (for example when the Claude connection drops) is skipped and never saved, so the log quietly leaves it
    out. Parsing the printout recovers those rows and counts them, so the run is not understated.

To use console output, copy the text the pipeline printed into a file (for example run_console.txt) and pass it with --console.
Or capture it while running:   python -m src.pipeline 100 | Tee-Object -FilePath run_console.txt   (Windows PowerShell)
                               python -m src.pipeline 100 | tee run_console.txt                    (Mac / Linux)

Outputs go to results/reports/ (not tracked by Git, because they hold question text): <run id>.csv and, if openpyxl is
installed (pip install openpyxl), <run id>.xlsx with a Run sheet (colour-coded decisions) and a Summary sheet.

Options:
    --run RUN_ID      a run in the history (default: the newest)
    --list            list the most recent runs and exit
    --console FILE    read pasted console output instead of the saved log
    --history PATH    a different history file (default results/logs/results_history.csv)
    --out-dir DIR     where to write the files (default results/reports)
    --no-files        print only; write nothing
    --rows N          print only the first N rows of the table (the files always hold all rows)

Usage:
    python scripts/export_run_report.py
    python scripts/export_run_report.py --run "run-18-Oct-26-09:05 AM"
    python scripts/export_run_report.py --console run_console.txt
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run_report import parse_console, tidy_from_history, summarize, problems, format_table, write_outputs

HISTORY_PATH = "results/logs/results_history.csv"


def print_summary(t: pd.DataFrame, source: str) -> None:
    """Print the counts, shares, medians and marks for the run."""
    s = summarize(t)
    n = s["rows"]
    print(f"\nRows: {n}" + (f" ({s['failed']} failed and not saved)" if s["failed"] else ""))
    print("Decisions: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in sorted(s["decisions"].items(), key=lambda kv: -kv[1])))
    print(f"Sent up by the router: {s['sent_up_by_router']} of {n} ({s['sent_up_share']:.0%}), "
          f"{s['decisions'].get('CACHE', 0)} answered from the cache, {s['decisions'].get('ESCALATE', 0)} by Claude"
          + (f", {s['failed']} failed" if s["failed"] else ""))
    for unit, d in s["by_unit"].items():
        total = sum(d.values())
        up = sum(v for k, v in d.items() if k != "LOCAL")
        print(f"  {unit}: {total} rows, {up} sent up ({up / total:.0%})")
    if s["median_total_ms"]:
        print("Median time (ms): " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(s["median_total_ms"].items())))
    if s["mean_confidence"] is not None:
        extra = f", mean predicted quality {s['mean_predicted_quality']:.2f}" if s["mean_predicted_quality"] is not None else ""
        print(f"Mean confidence {s['mean_confidence']:.3f}{extra}")
    if "mean_quality" in s:
        print(f"Mean quality mark {s['mean_quality']:.2f} over {n - s['failed'] - s['unscored']} marked rows")


def main() -> None:
    """Build the report for the chosen run and print it."""
    ap = argparse.ArgumentParser(description="Show a pipeline run as a plain table.")
    ap.add_argument("--run", default=None, help="run id (default: the newest run in the history)")
    ap.add_argument("--list", action="store_true", help="list the most recent runs and exit")
    ap.add_argument("--console", default=None, metavar="FILE", help="parse pasted console output instead of the saved log")
    ap.add_argument("--history", default=HISTORY_PATH, help=f"history file (default {HISTORY_PATH})")
    ap.add_argument("--out-dir", default="results/reports", help="where to write the files (default results/reports)")
    ap.add_argument("--no-files", action="store_true", help="print only; write no files")
    ap.add_argument("--rows", type=int, default=None, help="print only the first N rows")
    args = ap.parse_args()

    if args.console:
        with open(args.console, encoding="utf-8", errors="replace") as f:
            t = parse_console(f.read())
        if t.empty:
            sys.exit(f"No questions found in {args.console}. Paste the text the pipeline printed, starting at a 'Start Time' line.")
        stem, source, title = os.path.splitext(os.path.basename(args.console))[0] + "_report", "console", f"console output {args.console}"
    else:
        if not os.path.exists(args.history):
            sys.exit(f"{args.history} not found. Run the pipeline first, or point --history at your history file.")
        hist = pd.read_csv(args.history)
        if args.list:
            runs = hist.groupby("run_id", sort=False).size().tail(15)
            print("Most recent runs (newest last):")
            for rid, n in runs.items():
                print(f"  {rid}   {n} rows")
            return
        t, rid = tidy_from_history(hist, args.run)
        stem, source, title = rid, "history", f"saved run {rid}"

    print(f"Report for {title}\n")
    print(format_table(t if args.rows is None else t.head(args.rows)))
    print_summary(t, source)
    msgs = problems(t, source)
    if msgs:
        print("\nPROBLEMS")
        for m in msgs:
            print(f"  - {m}")
    else:
        print("\nNo problems found.")
    if not args.no_files:
        csv_path, xlsx_path = write_outputs(t, args.out_dir, stem)
        print(f"\nWrote {csv_path}" + (f" and {xlsx_path}" if xlsx_path else "  (install openpyxl for an Excel file: pip install openpyxl)"))


if __name__ == "__main__":
    main()
