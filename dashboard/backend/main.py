"""
Two-tier system dashboard - backend.

Reads the pipeline's own logs from the project's results/logs folder and serves
them as JSON, plus the prebuilt frontend:

    results/logs/results_history.csv      every run (preferred)
    results/logs/results_log_combined.csv latest run only (used if there is no history yet)

Nothing else is needed - evaluate.py does not have to be run for the dashboard.

Run from the project root:
    python -m uvicorn dashboard.backend.main:app --port 8000
then open http://localhost:8000

Set TWO_TIER_LOGS_DIR to read logs from somewhere else.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
LOGS_DIR = Path(os.environ.get("TWO_TIER_LOGS_DIR", PROJECT_ROOT / "results" / "logs"))
DIST_DIR = HERE.parent / "frontend" / "dist"

HISTORY_FILE = "results_history.csv"
LATEST_FILE = "results_log_combined.csv"
MAX_POINTS = 2000        # cap on dots sent for the strip chart
COLD_START_MS = 1000     # a Tier 1 call that spent longer than this loading a model is a cold start
RECENT_ROWS = 20

app = FastAPI(title="Two-tier system dashboard")


# --------------------------------------------------------------------------- helpers
def clean(o):
    """Make anything JSON-safe: numpy types to Python, NaN/inf to None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if o is pd.NaT or o is pd.NA:
        return None
    return o


def col(df, name):
    """A column as a Series, or all-NaN if the log does not have it (older runs)."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def num(df, name):
    return pd.to_numeric(col(df, name), errors="coerce")


def normalize(df):
    """Map whatever schema the log has onto one set of working columns, so runs
    recorded by older versions of the pipeline still show up."""
    out = pd.DataFrame(index=df.index)
    out["run_id"] = df["run_id"].fillna("(no run id)").astype(str)
    out["unit"] = col(df, "business_unit").fillna("unknown").astype(str)
    out["decision"] = col(df, "decision").astype(str).str.upper().str.strip()
    conf = num(df, "tier1_confidence_avg")
    out["conf"] = conf.fillna(num(df, "tier1_confidence"))          # older logs
    out["conf_min"] = num(df, "tier1_confidence_min")
    out["t1_ms"] = num(df, "tier1_latency_ms")
    out["t2_ms"] = num(df, "tier2_latency_ms")
    out["total_ms"] = num(df, "total_latency_ms").fillna(out["t1_ms"] + out["t2_ms"].fillna(0))
    out["load_ms"] = num(df, "tier1_load_ms")
    out["cost"] = num(df, "estimated_cost_usd").fillna(0.0)
    out["claude_conf"] = num(df, "tier2_confidence")   # Claude's own self-reported confidence (escalated rows only)
    out["claude_prompt_version"] = col(df, "tier2_prompt_version")   # which Tier 2 system prompt was in effect (escalated rows only)
    # cache_hit, logged as the string "True"/"False" like is_exception, not an actual bool column
    out["cache_hit"] = col(df, "cache_hit").map(lambda v: str(v).strip().lower() == "true")
    out["cache_similarity"] = num(df, "cache_similarity")
    out["quality_overall"] = num(df, "quality_overall")  # Claude-as-judge answer quality (1-5), where scored
    out["quality_local_overall"] = num(df, "quality_local_overall")  # local-SLM-as-judge (cheap first pass, separate judge)
    for dim in ("correctness", "completeness", "tone", "safety", "clarity"):
        out[f"quality_{dim}"] = num(df, f"quality_{dim}")
        out[f"quality_local_{dim}"] = num(df, f"quality_local_{dim}")
    out["is_exc"] = col(df, "is_exception").map(lambda v: str(v).strip().lower() == "true")
    out["threshold"] = num(df, "threshold_used")
    out["model"] = col(df, "tier1_model")
    out["query"] = col(df, "query").fillna("").astype(str)
    out["category"] = col(df, "category")
    out["intent"] = col(df, "intent")
    out["start"] = pd.to_datetime(col(df, "request_start_time"), errors="coerce")
    out["end"] = pd.to_datetime(col(df, "response_end_time"), errors="coerce")
    return out


_cache = {"sig": None, "df": None, "source": "none"}


def load():
    """(dataframe, source) - re-read only when the file changes."""
    history, latest = LOGS_DIR / HISTORY_FILE, LOGS_DIR / LATEST_FILE
    path = history if history.exists() else latest if latest.exists() else None
    if path is None:
        return None, "none"
    stat = path.stat()
    sig = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cache["sig"] != sig:
        try:
            raw = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return _cache["df"], _cache["source"]        # file caught mid-write: keep the last good copy
        if raw.empty:
            return None, "none"
        if "run_id" not in raw.columns:
            raw.insert(0, "run_id", "(no run id)")
        _cache.update(sig=sig, df=normalize(raw), source="history" if path == history else "latest_only")
    return _cache["df"], _cache["source"]


def metrics(g):
    n = len(g)
    local = g["decision"].eq("LOCAL")
    esc = g["decision"].eq("ESCALATE")
    return {
        "rows": n,
        "local": int(local.sum()),
        "escalated": int(esc.sum()),
        "local_pct": 100.0 * local.sum() / n if n else None,
        "escalated_pct": 100.0 * esc.sum() / n if n else None,
        # decision stays "ESCALATE" for a cache hit (Tier 1 still wasn't confident enough on
        # its own) -- escalated_pct above is therefore "needed escalation", not "called
        # Claude". These three split that correctly: claude_pct is the corrected real-call
        # number, cache_pct is what the cache actually saved, and cache_hit_rate is specifically
        # "of the queries that needed escalation, what fraction were served from cache" -- the
        # number that answers "is the cache pulling its weight," independent of how the
        # confidence threshold itself is performing.
        "cache_hits": int((esc & g["cache_hit"]).sum()),
        "claude_calls": int((esc & ~g["cache_hit"]).sum()),
        "cache_pct": 100.0 * (esc & g["cache_hit"]).sum() / n if n else None,
        "claude_pct": 100.0 * (esc & ~g["cache_hit"]).sum() / n if n else None,
        "cache_hit_rate": 100.0 * (esc & g["cache_hit"]).sum() / esc.sum() if esc.sum() else None,
        "avg_cache_similarity": g.loc[esc & g["cache_hit"], "cache_similarity"].mean() if (esc & g["cache_hit"]).any() else None,
        "avg_conf": g["conf"].mean(),
        "avg_conf_min": g["conf_min"].mean(),
        # Splitting average confidence by what happened to the query is more useful than one
        # blended number: routing itself is decided by confidence vs. threshold, so the overall
        # average mixes two different populations. These two let you compare "how confident were
        # the answers we kept locally" against "how confident were the ones sent to Claude" -
        # and pick a new threshold using the gap between them.
        "avg_local_conf": g.loc[local, "conf"].mean() if local.any() else None,
        "avg_escalated_conf": g.loc[esc, "conf"].mean() if esc.any() else None,
        # Claude's own self-reported confidence on the queries it actually answered -- a
        # genuinely different signal from avg_escalated_conf above (which is still Tier 1's
        # confidence, just filtered to the escalated rows). May be blank on rows logged before
        # this was added, or on the rare row where Claude did not follow the confidence format.
        "avg_claude_conf": g["claude_conf"].mean() if g["claude_conf"].notna().any() else None,
        "claude_conf_known": int(g["claude_conf"].notna().sum()),
        # Answer quality (1-5, Claude-as-judge) -- a separate measurement from confidence,
        # scored offline after the fact (see src/quality.py). Only present on rows scored
        # with --score-quality or the backfill script, so quality_known is usually much
        # smaller than rows -- don't read avg_quality as representative of the whole run
        # unless quality_known is close to rows.
        "avg_quality": g["quality_overall"].mean() if g["quality_overall"].notna().any() else None,
        "quality_known": int(g["quality_overall"].notna().sum()),
        "quality_by_dim": {
            dim: g[f"quality_{dim}"].mean() if g[f"quality_{dim}"].notna().any() else None
            for dim in ("correctness", "completeness", "tone", "safety", "clarity")
        },
        # Local-SLM-judge quality: a SEPARATE, less validated signal (see src/quality.py) --
        # kept in its own fields throughout, never blended with the Claude-judged quality_*
        # fields above, so the dashboard never implies they're the same measurement.
        "avg_quality_local": g["quality_local_overall"].mean() if g["quality_local_overall"].notna().any() else None,
        "quality_local_known": int(g["quality_local_overall"].notna().sum()),
        "quality_local_by_dim": {
            dim: g[f"quality_local_{dim}"].mean() if g[f"quality_local_{dim}"].notna().any() else None
            for dim in ("correctness", "completeness", "tone", "safety", "clarity")
        },
        "median_total_ms": g["total_ms"].median(),
        "median_local_ms": g.loc[local, "total_ms"].median(),
        "median_escalated_ms": g.loc[esc, "total_ms"].median(),
        "tier2_cost_usd": g["cost"].sum(),
        "cold_starts": int((g["load_ms"] > COLD_START_MS).sum()),
        "load_known": int(g["load_ms"].notna().sum()),      # rows that recorded a model-load time at all
        # The threshold(s) actually in effect for this group. Usually one value; a list lets a
        # mixed group (e.g. a run where the threshold changed mid-way) show that honestly rather
        # than picking one arbitrarily.
        "threshold": sorted({round(float(t), 3) for t in g["threshold"].dropna()}),
    }


def models_of(g):
    """{business unit: model used} for a run."""
    out = {}
    for unit, sub in g.groupby("unit"):
        names = sub["model"].dropna()
        out[unit] = names.mode().iloc[0] if not names.empty else None
    return out


def claude_prompt_version_of(g):
    """The Tier 2 system prompt version in effect for this run's escalations, if any --
    shared across both business units (Claude is one model either way), unlike models_of()
    which is necessarily per-unit."""
    versions = g["claude_prompt_version"].dropna()
    return versions.mode().iloc[0] if not versions.empty else None


def quality_three_way(g):
    """Claude-judged quality (quality_overall), split three ways: the Payments SLM's own
    answers, the Retail Bank SLM's own answers, and Claude's answers -- each only for the
    rows that model actually produced (LOCAL for an SLM, ESCALATE for Claude, which handles
    escalations from both units combined, since it's the same model either way).

    This is NOT a fair "which model is better" comparison -- ESCALATE is specifically the
    subset each SLM found hard (low confidence), while LOCAL is the subset it found easy, so
    the three groups are answering different-difficulty queries by construction. Read this as
    "is the quality acceptable on each path," not as a capability ranking between the SLMs and
    Claude. For a same-query, controlled comparison, see the quality_draft_* columns instead
    (scripts/backfill_quality_scores.py --draft).
    """
    def q(sub):
        vals = sub["quality_overall"].dropna()
        return {"avg": vals.mean() if len(vals) else None, "known": int(len(vals)), "rows": int(len(sub))}

    payments_local = g[(g["unit"] == "payments") & (g["decision"] == "LOCAL")]
    retail_local = g[(g["unit"] == "retail_bank") & (g["decision"] == "LOCAL")]
    claude = g[g["decision"] == "ESCALATE"]
    return {"payments_slm": q(payments_local), "retail_slm": q(retail_local), "claude": q(claude)}


# --------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    df, source = load()
    return {"status": "ok", "logs_dir": str(LOGS_DIR), "source": source,
            "rows": 0 if df is None else len(df),
            "runs": 0 if df is None else int(df["run_id"].nunique())}


@app.get("/api/runs")
def runs():
    """Every run in the order it happened, with headline metrics - feeds the run ledger and trend chart."""
    df, source = load()
    if df is None:
        return {"source": "none", "runs": []}
    out = []
    for run_id in dict.fromkeys(df["run_id"]):            # first appearance = chronological
        g = df[df["run_id"] == run_id]
        out.append({
            "run_id": run_id,
            "rows": len(g),
            "models": models_of(g),
            "claude_prompt_version": claude_prompt_version_of(g),
            "overall": metrics(g),
            "units": {unit: metrics(sub) for unit, sub in g.groupby("unit")},
            "quality_three_way": quality_three_way(g),
        })
    return clean({"source": source, "runs": out})


@app.get("/api/run")
def run_detail(run_id: str = Query(...)):
    df, source = load()
    if df is None or run_id not in set(df["run_id"]):
        raise HTTPException(status_code=404, detail=f"No run called '{run_id}'")
    g = df[df["run_id"] == run_id]
    local = g["decision"].eq("LOCAL")
    esc = g["decision"].eq("ESCALATE")

    # where each answer landed against the threshold
    pts = g.dropna(subset=["conf"])
    if len(pts) > MAX_POINTS:
        pts = pts.sample(MAX_POINTS, random_state=0)
    points = [{"c": r.conf, "u": r.unit, "x": bool(r.is_exc), "d": r.decision,
               "q": r.query[:110], "i": r.intent if isinstance(r.intent, str) else ""}
              for r in pts.itertuples()]

    scenarios = []
    for is_exc, label in ((False, "Normal"), (True, "Exception")):
        sub = g[g["is_exc"] == is_exc]
        if len(sub):
            scenarios.append({"type": label, **metrics(sub)})

    recent = g.tail(RECENT_ROWS).iloc[::-1]
    return clean({
        "run_id": run_id,
        "source": source,
        "models": models_of(g),
        "claude_prompt_version": claude_prompt_version_of(g),
        "thresholds": sorted({round(float(t), 3) for t in g["threshold"].dropna()}),
        "started": g["start"].min().isoformat() if g["start"].notna().any() else None,
        "ended": g["end"].max().isoformat() if g["end"].notna().any() else None,
        "kpis": metrics(g),
        "quality_three_way": quality_three_way(g),
        "by_unit": [{"unit": u, "model": models_of(sub).get(u), **metrics(sub)} for u, sub in g.groupby("unit")],
        "scenarios": scenarios,
        "latency": {
            "local": {"tier1": g.loc[local, "t1_ms"].median(), "tier2": 0},
            "escalated": {"tier1": g.loc[esc, "t1_ms"].median(), "tier2": g.loc[esc, "t2_ms"].median()},
        },
        "points": points,
        "recent": [{"start": r.start.isoformat() if pd.notna(r.start) else None, "unit": r.unit,
                    "category": r.category, "intent": r.intent, "exception": bool(r.is_exc),
                    "query": r.query, "conf": r.conf, "decision": r.decision, "total_ms": r.total_ms}
                   for r in recent.itertuples()],
    })


# --------------------------------------------------------------------------- frontend
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="ui")
else:
    @app.get("/")
    def no_frontend():
        return {"message": "The frontend is not built. Build it with: cd dashboard/frontend && npm install && npm run build"}
