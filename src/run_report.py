"""
Turns one pipeline run into a plain table, a short summary, and a list of problems, so a run can be read, shared and
checked without scrolling through console output.

Two sources, because they hold different things:

1. The saved log (results/logs/results_history.csv). This is the normal, complete record of every row that was SAVED:
   full question, full answers, confidence, the router's prediction, the decision, timings, quality marks. Use it by default.
2. Pasted console output (the text the pipeline prints while it runs). It is only a printout, but it is the only place a
   row that FAILED still appears. A row that fails (for example because the Claude connection dropped) is skipped and never
   saved, so the CSV quietly omits it. Parsing the console text recovers those rows and lets the summary count them.

Why the failed rows matter: the rows that fail at the Claude step are exactly the ones the router sent up and the cache could
not answer. Leaving them out makes a run look as if it escalated less and cost less than it did.

parse_console(text)           console text -> table (one row per question, including failed ones)
tidy_from_history(df, run_id) a run's rows from the history table -> the same layout of table
summarize(table)              the counts, shares and medians that describe the run
problems(table)               plain-language warnings, each with the command that fixes it
format_table(table)           a compact table for the screen
write_outputs(table, ...)     writes a CSV, and an Excel workbook if openpyxl is installed

Used by scripts/export_run_report.py. Needs only pandas; openpyxl is optional (for the Excel file).
"""
import os
import re

import pandas as pd

_LABEL = re.compile(r"^(Start Time|Business Unit|Scenario|Query|Tier 1 Answer|Confidence AVG|Confidence MIN|Router|"
                    r"Threshold|Decision|Cache hit|Latency)\s*:\s?(.*)$")
_FAIL = re.compile(r"WARNING: row failed and will be skipped \((.*)\)\s*$")
_LATENCY = re.compile(r"Tier1 (\d+) ms(?: \(model load (\d+) ms\))? \| Tier2 (?:(\d+) ms(?: \(Claude confidence ([\d.]+)\))?|-) \| Total (\d+) ms")
_ROUTER = re.compile(r"(\S+) \| predicted quality ([\d.]+) \(cut-point ([\d.]+)\)")
_SEPARATOR = re.compile(r"^-{20,}\s*$")

COLUMNS = ["n", "start_time", "business_unit", "category", "intent", "is_exception", "query", "tier1_confidence_avg",
           "predicted_quality", "cut_point", "router", "router_decision", "final_decision", "cache_similarity",
           "tier1_ms", "tier2_ms", "total_ms", "status", "quality_overall", "quality_draft_overall", "tier1_answer", "final_answer"]


def _block_to_row(b: dict) -> dict:
    """Turn one parsed console block (a dict of label -> text) into one table row."""
    unit_m = re.match(r"(\S+)", b.get("Business Unit", ""))
    scen_m = re.match(r"(\S+)\s*/\s*(\S+)(\s*\[EXCEPTION\])?", b.get("Scenario", ""))
    router_m = _ROUTER.search(b.get("Router", ""))
    lat_m = _LATENCY.search(b.get("Latency", ""))
    cache_m = re.search(r"similarity ([\d.]+)", b.get("Cache hit", ""))
    failed = b.get("_failed")
    decision = b.get("Decision", "").strip()
    if failed:
        final = "FAILED"
    elif cache_m:
        final = "CACHE"
    else:
        final = decision
    def num(key):
        try:
            return float(b.get(key, "").strip())
        except ValueError:
            return None
    cut = router_m.group(3) if router_m else b.get("Threshold", "").strip() or None
    return {
        "start_time": b.get("Start Time", "").strip(),
        "business_unit": unit_m.group(1) if unit_m else None,
        "category": scen_m.group(1) if scen_m else None,
        "intent": scen_m.group(2) if scen_m else None,
        "is_exception": bool(scen_m and scen_m.group(3)),
        "query": b.get("Query", "").strip(),
        "tier1_confidence_avg": num("Confidence AVG"),
        "predicted_quality": float(router_m.group(2)) if router_m else None,
        "cut_point": float(cut) if cut else None,
        "router": router_m.group(1) if router_m else "threshold",
        "router_decision": decision,
        "final_decision": final,
        "cache_similarity": float(cache_m.group(1)) if cache_m else None,
        "tier1_ms": int(lat_m.group(1)) if lat_m else None,
        "tier2_ms": int(lat_m.group(3)) if lat_m and lat_m.group(3) else None,
        "total_ms": int(lat_m.group(5)) if lat_m else None,
        "status": f"FAILED: {failed}" if failed else "OK",
        "tier1_answer": " ".join(b.get("Tier 1 Answer", "").split()),
    }


def parse_console(text: str) -> pd.DataFrame:
    """Parse the pipeline's console output into a table with one row per question, in order.
    Rows that failed and were skipped are included, with final_decision FAILED and the reason in status.
    Text before the first question and the run summary after the last one are ignored."""
    blocks, cur, last = [], None, None
    for raw in text.splitlines():
        line = raw.rstrip()
        m = _LABEL.match(line)
        if m and m.group(1) == "Start Time":
            cur = {"Start Time": m.group(2)}
            blocks.append(cur)
            last = "Start Time"
            continue
        if cur is None:
            continue
        f = _FAIL.search(line)
        if f:
            cur["_failed"] = f.group(1)
            last = None
            continue
        if _SEPARATOR.match(line):
            last = None
            continue
        if m:
            cur[m.group(1)] = m.group(2)
            last = m.group(1)
        elif last in ("Query", "Tier 1 Answer") and line.strip():
            cur[last] = cur[last] + " " + line.strip()  # a long answer printed over several lines
        elif not line.strip():
            continue
        else:
            last = None  # the run summary after the last question
    rows = [_block_to_row(b) for b in blocks if "Scenario" in b]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df.insert(0, "n", range(1, len(df) + 1))
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[COLUMNS]


def tidy_from_history(history: pd.DataFrame, run_id: str = None):
    """Pick one run out of the history table (the newest by default) and lay it out like the console table.
    Returns (table, run_id). Rows are in the order they were logged."""
    if "run_id" not in history.columns:
        raise ValueError("the history has no run_id column: run python scripts/backfill_run_id.py once")
    rid = run_id or history["run_id"].iloc[-1]
    h = history[history["run_id"] == rid].reset_index(drop=True)
    if h.empty:
        raise ValueError(f"run {rid!r} not found. Runs in the history: {', '.join(map(str, history['run_id'].drop_duplicates().tail(8)))}")
    g = lambda c: h[c] if c in h.columns else pd.Series([None] * len(h))
    # Older logs recorded a cache hit as decision ESCALATE with cache_hit true; the dashboard reads those as CACHE, and so do we.
    hit = g("cache_hit").astype(str).str.lower().eq("true")
    final = g("decision").where(~hit, "CACHE")
    df = pd.DataFrame({
        "n": range(1, len(h) + 1), "start_time": g("request_start_time"), "business_unit": g("business_unit"),
        "category": g("category"), "intent": g("intent"), "is_exception": g("is_exception"), "query": g("query"),
        "tier1_confidence_avg": g("tier1_confidence_avg"), "predicted_quality": g("router_predicted_quality"),
        "cut_point": g("router_cutoff").fillna(g("threshold_used")), "router": g("router").fillna("threshold"),
        "router_decision": final.replace({"CACHE": "ESCALATE"}), "final_decision": final,
        "cache_similarity": g("cache_similarity"), "tier1_ms": g("tier1_latency_ms"), "tier2_ms": g("tier2_latency_ms"),
        "total_ms": g("total_latency_ms"), "status": "OK", "quality_overall": g("quality_overall"),
        "quality_draft_overall": g("quality_draft_overall"), "tier1_answer": g("tier1_answer"), "final_answer": g("final_answer")})
    return df[COLUMNS], rid


def summarize(t: pd.DataFrame) -> dict:
    """Describe a run: row counts, the share of each decision, how many the router sent up, medians and marks."""
    n = len(t)
    counts = t["final_decision"].value_counts().to_dict()
    sent_up = sum(counts.get(k, 0) for k in ("CACHE", "ESCALATE", "FAILED"))
    out = {"rows": n, "decisions": counts, "sent_up_by_router": sent_up,
           "sent_up_share": sent_up / n if n else 0.0, "failed": counts.get("FAILED", 0)}
    out["by_unit"] = t.groupby("business_unit")["final_decision"].value_counts().unstack(fill_value=0).to_dict("index") if n else {}
    out["median_total_ms"] = {k: float(v) for k, v in t[t["total_ms"].notna()].groupby("final_decision")["total_ms"].median().items()}
    out["mean_confidence"] = float(t["tier1_confidence_avg"].mean()) if t["tier1_confidence_avg"].notna().any() else None
    out["mean_predicted_quality"] = float(t["predicted_quality"].mean()) if t["predicted_quality"].notna().any() else None
    ok = t[t["status"] == "OK"]
    if ok["quality_overall"].notna().any():  # only the saved log carries quality marks; console output does not
        out["unscored"] = int(ok["quality_overall"].isna().sum())
        out["mean_quality"] = float(ok["quality_overall"].mean())
    return out


def problems(t: pd.DataFrame, source: str = "history") -> list:
    """Plain-language warnings about a run, each with what to do about it. An empty list means nothing was found.
    source is "history" (the saved log) or "console" (pasted console output, which has no quality marks to check)."""
    msgs = []
    failed = t[t["status"].str.startswith("FAILED")]
    if len(failed):
        reasons = ", ".join(sorted({s_.split(": ", 1)[1] for s_ in failed["status"]}))
        n = len(t)
        msgs.append(
            f"{len(failed)} of {n} rows FAILED ({reasons}) and were never saved to the log. They are the questions the router sent up that the cache "
            f"could not answer, so the saved run understates how many went to Claude: {len(failed)} of the {n} needed a fresh Claude answer. "
            f"A connection error usually means the internet, a VPN or proxy, or the API was unreachable. Check it, then re-run: "
            f"python -m src.pipeline {n}")
    if source == "history":
        ok = t[t["status"] == "OK"]
        blank = int(ok["quality_overall"].isna().sum())
        if blank:
            who = "saved row has" if blank == 1 else "saved rows have"
            msgs.append(
                f"{blank} {who} no quality mark, so the quality columns are incomplete for this run. Fill them in (it only judges the blank rows): "
                f"python scripts/backfill_quality_scores.py   and then   python scripts/backfill_quality_scores.py --draft")
    return msgs


def format_table(t: pd.DataFrame, width: int = 140) -> str:
    """A compact table for the screen: the columns a person reads first, with the question shortened."""
    v = t[["n", "business_unit", "intent", "final_decision", "tier1_confidence_avg", "predicted_quality", "total_ms", "query"]].copy()
    v["business_unit"] = v["business_unit"].str.replace("retail_bank", "retail").str.replace("payments", "pay")
    v["query"] = v["query"].fillna("").str.slice(0, 48)
    v["total_ms"] = pd.to_numeric(v["total_ms"], errors="coerce").map(lambda x: "-" if pd.isna(x) else f"{x:,.0f}")
    for c in ("tier1_confidence_avg", "predicted_quality"):
        v[c] = pd.to_numeric(v[c], errors="coerce").map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    v.columns = ["#", "unit", "intent", "decision", "conf", "pred.q", "ms", "question"]
    return v.to_string(index=False, max_colwidth=48, line_width=width, na_rep="-", justify="left")


def write_outputs(t: pd.DataFrame, out_dir: str, stem: str):
    """Write the table as CSV and, if openpyxl is installed, as a formatted Excel workbook.
    Returns (csv_path, xlsx_path or None)."""
    os.makedirs(out_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    csv_path = os.path.join(out_dir, safe + ".csv")
    t.to_csv(csv_path, index=False)
    try:
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        return csv_path, None
    xlsx_path = os.path.join(out_dir, safe + ".xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        t.to_excel(xw, sheet_name="Run", index=False)
        ws = xw.sheets["Run"]
        ws.freeze_panes = "A2"
        fills = {"LOCAL": "DCEBFA", "CACHE": "FDE9E0", "ESCALATE": "DFF3EA", "FAILED": "F8D7DA"}
        dcol = list(t.columns).index("final_decision") + 1
        for c in ws[1]:
            c.font = Font(bold=True)
        for row in ws.iter_rows(min_row=2):
            fill = fills.get(row[dcol - 1].value)
            if fill:
                row[dcol - 1].fill = PatternFill("solid", fgColor=fill)
        widths = {"query": 60, "tier1_answer": 60, "final_answer": 60, "status": 34, "intent": 30, "start_time": 20}
        for i, col in enumerate(t.columns, start=1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(col, 14)
        for col in ("query", "tier1_answer", "final_answer"):
            if col in t.columns:
                for cell in ws[ws.cell(1, list(t.columns).index(col) + 1).column_letter][1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
        s = summarize(t)
        rows = [("Rows", s["rows"]), ("Sent up by the router", f"{s['sent_up_by_router']} ({s['sent_up_share']:.0%})"), ("Failed", s["failed"])]
        rows += [(f"Decision {k}", v) for k, v in s["decisions"].items()]
        pd.DataFrame(rows, columns=["Measure", "Value"]).to_excel(xw, sheet_name="Summary", index=False)
        xw.sheets["Summary"].column_dimensions["A"].width = 28
        xw.sheets["Summary"].column_dimensions["B"].width = 18
    return csv_path, xlsx_path
