"""
Main pipeline: pick random rows from the synthetic Bitext-format CSVs
(data/synthetic/synthetic_bitext_*.csv), run each through the two-tier
cascade, and log every decision for later evaluation.

- Business unit comes from the row's category (CARD/TRANSFER/ATM/FEES/
  PAYMENT_EXCEPTION -> payments; ACCOUNT/LOAN/PASSWORD/CONTACT/FIND/
  RETAIL_EXCEPTION -> retail_bank), so each query goes to the right
  assistant instead of an arbitrary split.
- Threshold is set per business unit (and per normal/exception scenario within
  it) via the THRESHOLDS dict below, since different Tier 1 models can have very
  different confidence distributions -- a smaller fine-tuned model naturally
  produces less confident answers even when its content is fine.
- Both confidence methods (avg and min) are logged for every query; routing
  uses confidence_score (currently = the avg method) via router.route().
- Latency is logged three ways per query, so it can be evaluated properly:
    request_start_time / response_end_time  wall-clock timestamps
    tier1_latency_ms, tier2_latency_ms      time spent in each tier
    total_latency_ms                        end-to-end (tier 1 + routing + tier 2)
  plus tier1_load_ms (Ollama's model-load time inside the Tier 1 call --
  large on a cold start, ~0 when the model is already in memory) and
  tier1_output_tokens (answer length, which drives generation time).
- Models are preloaded before the first query, kept loaded for the whole run
  (KEEP_ALIVE), and unloaded when the run ends -- including on an error or
  Ctrl+C -- so model-load time never lands inside a query's latency.
- Every run gets a run_id, run-<dd-mmm-yy>-<hh:mm AM/PM> (e.g. run-28-Sep-26-06:29 PM),
  as the first column of every log. results_log_*.csv hold the LATEST run only (which is
  what evaluate.py reads); every run is also appended to results/logs/results_history.csv,
  so runs -- e.g. before and after retraining -- can be compared.

Run from the project root:
    python -m src.pipeline          # default sample size
    python -m src.pipeline 50       # 50 random rows
"""
import glob
import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests

from .config import OLLAMA_URL, get_model_for_unit
from .tier1 import ask_tier1
from .router import route
from .tier2_escalate import ask_tier2, estimate_cost, client as anthropic_client
from .quality import judge_answer_quality, judge_answer_quality_local, QUALITY_DIMS
from .semantic_cache import SemanticCacheIndex, try_semantic_cache

# Whole-file exclusive locking so results_history.csv cannot be opened elsewhere
# (Excel, a text editor) for the whole duration of a run -- not just checked once
# at the start. msvcrt on Windows, flock on Mac/Linux.
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

SYNTHETIC_DIR = "data/synthetic"
LOGS_DIR = "results/logs"

N_SAMPLES = 100
SEED = None  # None = different random rows every run; set an int to reproduce a run

# Escalate anything scoring below its threshold. Threshold is set per business unit,
# and can be set separately for normal vs. exception rows within a unit -- edit the
# values below to whatever your data supports. A unit not listed here, or a run whose
# rows have no business_unit match, falls back to DEFAULT_THRESHOLD.
THRESHOLDS = {
    "payments": {"normal": 0.71, "exception": 0.71},
    # 0.55 is a suggested starting point, not a measured optimum: it sits just above
    # retail_bank's own mean/median confidence (0.529 / 0.527 in the latest run), the
    # same relationship 0.7 already has to payments' mean (0.700). Re-tune from the
    # Overview page once you have more runs at this setting.
    "retail_bank": {"normal": 0.6, "exception": 0.6},
}
DEFAULT_THRESHOLD = 0.7


def get_threshold(business_unit: str, is_exception: bool) -> float:
    scenario = "exception" if is_exception else "normal"
    return THRESHOLDS.get(business_unit, {}).get(scenario, DEFAULT_THRESHOLD)

# How long Ollama keeps a model loaded after each request during a run. Finite on
# purpose: if the script is killed hard (so it can't unload), the models still free
# themselves after this long.
KEEP_ALIVE = "30m"

HISTORY_PATH = f"{LOGS_DIR}/results_history.csv"

PAYMENTS_CATEGORIES = {"CARD", "TRANSFER", "ATM", "FEES", "PAYMENT_EXCEPTION"}
RETAIL_CATEGORIES = {"ACCOUNT", "LOAN", "PASSWORD", "CONTACT", "FIND", "RETAIL_EXCEPTION"}
EXCEPTION_CATEGORIES = {"PAYMENT_EXCEPTION", "RETAIL_EXCEPTION"}


def unit_for_category(category: str) -> str:
    if category in PAYMENTS_CATEGORIES:
        return "payments"
    if category in RETAIL_CATEGORIES:
        return "retail_bank"
    raise ValueError(f"Unknown category '{category}' -- add it to PAYMENTS_CATEGORIES or RETAIL_CATEGORIES")


def load_random_sample(n: int = N_SAMPLES, seed=SEED) -> pd.DataFrame:
    """Pool every synthetic CSV and draw n random rows from the whole pool."""
    files = sorted(glob.glob(f"{SYNTHETIC_DIR}/synthetic_bitext_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files matching {SYNTHETIC_DIR}/synthetic_bitext_*.csv -- "
            f"run scripts/generate_synthetic_bitext.py (or unzip the CSVs there) first."
        )
    pool = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sample = pool.sample(n=min(n, len(pool)), random_state=seed).reset_index(drop=True)
    sample["business_unit"] = sample["category"].map(unit_for_category)
    sample["is_exception"] = sample["category"].isin(EXCEPTION_CATEGORIES)
    return sample


def make_run_id(existing_ids=(), now=None) -> str:
    """run-<dd-mmm-yy>-<hh:mm AM/PM>. If that id is already taken (two runs inside the
    same minute), a -2, -3 ... suffix keeps ids unique."""
    base = (now or datetime.now()).strftime("run-%d-%b-%y-%I:%M %p")
    if base not in existing_ids:
        return base
    n = 2
    while f"{base}-{n}" in existing_ids:
        n += 1
    return f"{base}-{n}"


def existing_run_ids() -> set:
    if not os.path.exists(HISTORY_PATH):
        return set()
    try:
        return set(pd.read_csv(HISTORY_PATH, usecols=["run_id"])["run_id"].dropna())
    except ValueError:
        return set()


def guard_legacy_log() -> None:
    """Refuse to overwrite an older results log that has no run_id, so its data is not lost."""
    combined = f"{LOGS_DIR}/results_log_combined.csv"
    if os.path.exists(combined) and "run_id" not in pd.read_csv(combined, nrows=0).columns:
        raise RuntimeError(
            f"{combined} has no run_id column and this run would overwrite it. "
            f"Run `python scripts/backfill_run_id.py` first, so the existing results are kept."
        )


def guard_writable(paths) -> None:
    """Fail fast, before spending any time or Claude API cost, if a log file is open
    elsewhere (e.g. Excel on Windows) and would refuse to be written to at the end
    of the run. Only checks files that already exist -- a new file is always writable."""
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "a"):
                pass
        except PermissionError as e:
            raise RuntimeError(
                f"Cannot write to {path} -- it looks like it's open in another program "
                f"(Excel, a text editor, etc.). Close it there first, then re-run. ({e})"
            ) from e


# Sentinel byte-count used for msvcrt.locking(). Windows lets you lock a region larger
# than the file's current size (it just has to be consistent between lock and unlock),
# so one large fixed value safely covers the file at any size it might grow to.
_WIN_LOCK_BYTES = 0x7FFFFFFF - 1


class HistoryLock:
    """Holds an exclusive OS-level lock on results_history.csv for an entire pipeline
    run, so nothing else (Excel, a text editor, a second run of the pipeline) can open
    it from the moment the run starts until it finishes and the file is updated --
    not just at the final save step. Use as a context manager:

        with HistoryLock(HISTORY_PATH) as lock:
            ...run the whole pipeline...
            lock.write(final_history_dataframe)

    The lock is always released on the way out, including if the run raises partway
    through, so a crash never leaves the file stuck locked.
    """

    def __init__(self, path: str):
        self.path = path
        self.file = None

    def __enter__(self):
        if not os.path.exists(self.path):
            open(self.path, "a", encoding="utf-8").close()  # create it so it can be reopened in r+ mode
        try:
            # encoding="utf-8" matches pandas' own default for read_csv/to_csv when given a
            # path directly. Without it, Python falls back to the OS's local codepage (cp1252
            # on Windows), which cannot decode the UTF-8 bytes pandas itself writes -- that
            # mismatch, not the file's actual content, is what caused the UnicodeDecodeError.
            self.file = open(self.path, "r+", newline="", encoding="utf-8")
        except PermissionError as e:
            raise RuntimeError(
                f"Cannot open {self.path} -- it looks like it's open in another program "
                f"(Excel, a text editor, etc.). Close it there first, then re-run. ({e})"
            ) from e
        try:
            self._lock()
        except OSError as e:
            self.file.close()
            self.file = None
            raise RuntimeError(
                f"{self.path} is locked by another program (Excel, a text editor, or "
                f"another run of the pipeline). Close it there first, then re-run. ({e})"
            ) from e
        return self

    def _lock(self) -> None:
        self.file.seek(0)
        if sys.platform == "win32":
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, _WIN_LOCK_BYTES)
        else:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(self) -> None:
        self.file.seek(0)
        if sys.platform == "win32":
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, _WIN_LOCK_BYTES)
        else:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def read(self) -> pd.DataFrame:
        """Read the locked file's current contents through the same open handle."""
        self.file.seek(0)
        try:
            return pd.read_csv(self.file)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def write(self, df: pd.DataFrame) -> None:
        """Overwrite the locked file's contents with df, through the same open handle --
        never opens a second handle to the path, so it can't collide with the lock."""
        self.file.seek(0)
        self.file.truncate()
        df.to_csv(self.file, index=False)
        self.file.flush()
        os.fsync(self.file.fileno())

    def __exit__(self, exc_type, exc, tb):
        if self.file is not None:
            try:
                self._unlock()
            finally:
                self.file.close()
                self.file = None
        return False


def load_models(models: list) -> None:
    """Load each model into memory now (a generate request with no prompt just loads it)."""
    for m in models:
        t = time.perf_counter()
        try:
            r = requests.post(OLLAMA_URL, json={"model": m, "keep_alive": KEEP_ALIVE}, timeout=300)
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Could not load model '{m}'. Is Ollama running, and has `ollama create` been run for it? ({e})"
            ) from e
        print(f"Loaded {m} in {time.perf_counter() - t:.1f} s")


def warn_if_not_resident(models: list) -> None:
    """Ollama unloads an idle model if a new one does not fit alongside it. Check that
    every model is actually still in memory, and warn loudly if not."""
    try:
        base = OLLAMA_URL.rsplit("/api/", 1)[0]
        running = requests.get(f"{base}/api/ps", timeout=10).json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return
    norm = lambda n: n if ":" in n else n + ":latest"
    resident = {norm(m.get("name", "")) for m in running}
    missing = [m for m in models if norm(m) not in resident]
    if missing:
        print(f"WARNING: {missing} not resident after preloading -- the models do not fit in memory "
              f"together, so Ollama will swap them and tier1_load_ms will stay high.")


def unload_models(models: list) -> None:
    """Unload each model (keep_alive 0). Never raises -- this runs in a finally block."""
    for m in models:
        try:
            requests.post(OLLAMA_URL, json={"model": m, "keep_alive": 0}, timeout=60).raise_for_status()
            print(f"Unloaded {m}")
        except requests.RequestException as e:
            print(f"Warning: could not unload {m}: {e}")


def process_query(query: str, business_unit: str, category: str, intent: str, is_exception: bool, run_id: str,
                   cache_index: SemanticCacheIndex = None) -> dict:
    threshold = get_threshold(business_unit, is_exception)

    start_dt = datetime.now()
    t_query_start = time.perf_counter()

    t1_result = ask_tier1(query, business_unit=business_unit, keep_alive=KEEP_ALIVE)
    tier1_latency_ms = (time.perf_counter() - t_query_start) * 1000

    decision = route(t1_result["confidence_score"], threshold)

    print(f"Start Time     : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Business Unit  : {t1_result['business_unit']} ({t1_result['assistant_name']})")
    print(f"Scenario       : {category} / {intent}{'  [EXCEPTION]' if is_exception else ''}")
    print(f"Query          : {query}")
    print(f"Tier 1 Answer  : {t1_result['answer'][:100]}{'...' if len(t1_result['answer']) > 100 else ''}")
    print(f"Confidence AVG : {t1_result['confidence_score_avg']:.4f}")
    print(f"Confidence MIN : {t1_result['confidence_score_min']:.4f}")
    print(f"Threshold      : {threshold}")
    print(f"Decision       : {decision}")

    row = {
        "run_id": run_id,
        "request_start_time": start_dt.isoformat(timespec="milliseconds"),
        "response_end_time": None,
        "business_unit": business_unit,
        "assistant_name": t1_result["assistant_name"],
        "tier1_model": t1_result["model"],
        "category": category,
        "intent": intent,
        "is_exception": is_exception,
        "query": query,
        "tier1_answer": t1_result["answer"],
        "tier1_confidence_avg": t1_result["confidence_score_avg"],
        "tier1_confidence_min": t1_result["confidence_score_min"],
        "threshold_used": threshold,
        "decision": decision,
        "tier1_latency_ms": round(tier1_latency_ms, 1),
        "tier2_latency_ms": None,
        "total_latency_ms": None,
        "tier1_load_ms": t1_result.get("ollama_load_ms"),
        "tier1_output_tokens": t1_result.get("ollama_output_tokens"),
        "tier2_input_tokens": None,
        "tier2_output_tokens": None,
        "tier2_confidence": None,
        "tier2_prompt_version": None,
        "estimated_cost_usd": 0.0,
        "final_answer": t1_result["answer"],
        "cache_hit": False,
        "cache_similarity": None,
        "cache_path": None,  # "direct" when cache_hit is True (always "direct" now -- the gray-zone
                             # verification path was removed; see src/semantic_cache.py)
        "cache_matched_query": None,
    }

    tier2_latency_ms = None
    if decision == "ESCALATE":
        cache_result = None
        if cache_index is not None:
            try:
                cache_result = try_semantic_cache(cache_index, query, category=category, intent=intent)
            except Exception as e:
                # The cache is an optimisation, not the pipeline's job: if a lookup fails for
                # any reason, treat it as a miss and escalate normally rather than losing the row.
                print(f"    WARNING: cache lookup failed ({type(e).__name__}: {e}) -- treating as a miss")
        if cache_result is not None:
            # Served from the semantic cache -- no Tier 2 call, no cost, and (for a "direct"
            # hit) no extra latency beyond the cache lookup itself. decision stays "ESCALATE"
            # (Tier 1 still wasn't confident enough to answer alone) so existing LOCAL/ESCALATE
            # analysis keeps working unchanged; cache_hit/cache_path are the new, additive
            # fields that distinguish "served from cache" from "actually called Claude".
            row["cache_hit"] = True
            row["cache_similarity"] = round(cache_result["similarity"], 4)
            row["cache_path"] = cache_result["path"]
            row["cache_matched_query"] = cache_result["matched_query"]
            row["final_answer"] = cache_result["answer"]
        else:
            t2_start = time.perf_counter()
            t2_result = ask_tier2(query, business_unit=business_unit)
            tier2_latency_ms = (time.perf_counter() - t2_start) * 1000
            row["tier2_latency_ms"] = round(tier2_latency_ms, 1)
            row["tier2_input_tokens"] = t2_result["input_tokens"]
            row["tier2_output_tokens"] = t2_result["output_tokens"]
            row["tier2_confidence"] = t2_result["confidence"]
            row["tier2_prompt_version"] = t2_result["prompt_version"]
            row["estimated_cost_usd"] = estimate_cost(t2_result["input_tokens"], t2_result["output_tokens"])
            row["final_answer"] = t2_result["answer"]

    total_latency_ms = (time.perf_counter() - t_query_start) * 1000
    row["total_latency_ms"] = round(total_latency_ms, 1)
    row["response_end_time"] = datetime.now().isoformat(timespec="milliseconds")

    tier2_text = f"{tier2_latency_ms:.0f} ms" if tier2_latency_ms is not None else "-"
    tier2_conf_text = f" (Claude confidence {row['tier2_confidence']:.2f})" if row["tier2_confidence"] is not None else ""
    load_ms = row["tier1_load_ms"]
    load_text = f" (model load {load_ms:.0f} ms)" if load_ms else ""
    print(f"Latency        : Tier1 {tier1_latency_ms:.0f} ms{load_text} | Tier2 {tier2_text}{tier2_conf_text} | Total {total_latency_ms:.0f} ms")
    print("-" * 60)
    return row


def score_quality_pass(rows: list) -> None:
    """Score every row's Tier 1 answer for quality (correctness/completeness/tone/safety/
    clarity, 1-5 each, via Claude-as-judge -- see src/quality.py), in place, adding
    quality_<dimension>, quality_overall and quality_note keys to each row dict. Runs AFTER
    all Tier 1/Tier 2 processing is done, so it never affects routing or latency numbers --
    it's a separate, offline measurement of whether the answers are actually good, not just
    confidently produced. This is one extra Claude API call per row: for a large sample this
    adds real time and cost on top of the pipeline run itself, which is why it can be skipped
    with --noquality.
    """
    print(f"\nScoring answer quality for {len(rows)} rows (Claude-as-judge, {QUALITY_DIMS})...")
    for i, row in enumerate(rows, 1):
        # final_answer, not tier1_answer: for an escalated row, tier1_answer is the discarded
        # low-confidence draft, and final_answer is what Tier 2 actually produced and the
        # customer actually received. Scoring tier1_answer for an escalated row would measure
        # the wrong text entirely.
        result = judge_answer_quality(anthropic_client, row["query"], row["final_answer"])
        if result:
            for dim in QUALITY_DIMS:
                row[f"quality_{dim}"] = result[dim]
            row["quality_overall"] = round(sum(result[d] for d in QUALITY_DIMS) / len(QUALITY_DIMS), 2)
            row["quality_note"] = result["note"]
        else:
            for dim in QUALITY_DIMS:
                row[f"quality_{dim}"] = None
            row["quality_overall"] = None
            row["quality_note"] = None
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} scored")


def score_quality_pass_local(rows: list, judge_model: str) -> None:
    """Cheap, broad first-pass quality scoring via a local Ollama model instead of Claude --
    fast and free enough to run over every row without the cost concern that makes the
    Claude pass opt-in. Writes quality_local_<dimension> columns, kept separate from
    quality_<dimension> (the Claude-judged columns) so it's never mistaken for the numbers
    you'd actually report. Treat this as a triage signal, not a citable result on its own --
    see scripts/compare_quality_judges.py to check how well it agrees with Claude before
    leaning on it for anything beyond that.
    """
    print(f"\nScoring answer quality locally via {judge_model} for {len(rows)} rows (cheap first pass, not for reporting)...")
    for i, row in enumerate(rows, 1):
        result = judge_answer_quality_local(judge_model, row["query"], row["final_answer"])  # see comment above
        if result:
            for dim in QUALITY_DIMS:
                row[f"quality_local_{dim}"] = result[dim]
            row["quality_local_overall"] = round(sum(result[d] for d in QUALITY_DIMS) / len(QUALITY_DIMS), 2)
            row["quality_local_note"] = result["note"]
        else:
            for dim in QUALITY_DIMS:
                row[f"quality_local_{dim}"] = None
            row["quality_local_overall"] = None
            row["quality_local_note"] = None
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} scored")


def run_pipeline(sample: pd.DataFrame, log_path: str = None, score_quality: bool = False, score_quality_local: str = None,
                  use_semantic_cache: bool = False) -> pd.DataFrame:
    """Run every sampled row through the cascade. Writes one combined log
    (read by evaluate.py) plus one log per business unit.

    use_semantic_cache: opt-in (default off) since, unlike score_quality/score_quality_local,
    this changes live ROUTING behaviour, not just logging -- an escalated query may now be
    served from a cached prior Claude answer instead of making a fresh Tier 2 call. Builds
    the cache index once, from results_history.csv as it stands at the start of this run (so
    cache entries only ever come from genuinely earlier runs' Tier 2 answers, never from rows
    being written during this same run).
    """
    if log_path is None:
        log_path = f"{LOGS_DIR}/results_log_combined.csv"
    os.makedirs(LOGS_DIR, exist_ok=True)
    guard_legacy_log()
    unit_log_paths = [f"{LOGS_DIR}/results_log_{u}.csv" for u in sample["business_unit"].unique()]
    guard_writable([log_path, *unit_log_paths])
    run_id = make_run_id(existing_run_ids())
    print(f"Run ID: {run_id}")

    models = sorted({get_model_for_unit(u) for u in sample["business_unit"].unique()})

    # results_history.csv is locked for the whole run, from here until the run finishes
    # and the file is updated -- not just checked once at the start. Nothing else (Excel,
    # a text editor, a second run of the pipeline) can open it in the meantime.
    with HistoryLock(HISTORY_PATH) as lock:
        print(f"Locked {HISTORY_PATH} -- it can't be opened elsewhere until this run finishes.")
        cache_index = None
        if use_semantic_cache:
            print("Building semantic cache index from existing Tier 2 answers...")
            try:
                cache_index = SemanticCacheIndex()
                cache_index.build(lock.read())
                print(f"  {len(cache_index.queries)} cached Tier 2 answers available.")
            except Exception as e:
                # The cache is on by default, so a missing optional dependency (e.g.
                # sentence-transformers not installed) or a failed model download must not take
                # down the whole run -- but it must not be silent either, or a 0% hit rate would
                # look like "no matches" when the cache never actually ran.
                cache_index = None
                print(f"  WARNING: semantic cache unavailable ({type(e).__name__}: {e}).")
                print("  Continuing this run WITHOUT the cache -- every escalation will call Claude.")
                print("  To fix: pip install sentence-transformers   (or pass --nocache to silence this)")
        try:
            print("Preloading models...")
            load_models(models)
            warn_if_not_resident(models)
            print("-" * 60)
            # One row's irrecoverable failure (e.g. Ollama hanging well past its retries in
            # tier1.py) no longer takes the whole batch down -- it's logged and skipped, and
            # every other row still completes and gets saved. Before this, a single bad row
            # late in a multi-hundred-row run meant losing every row that had already
            # succeeded, since nothing is written to disk until the whole loop finishes.
            rows = []
            failed = []
            for r in sample.itertuples(index=False):
                try:
                    rows.append(process_query(r.instruction, r.business_unit, r.category, r.intent, bool(r.is_exception), run_id,
                                               cache_index=cache_index))
                except Exception as e:
                    failed.append({"query": r.instruction, "business_unit": r.business_unit, "error": str(e)})
                    print(f"  WARNING: row failed and will be skipped ({type(e).__name__}: {e})")
            if failed:
                print(f"\n{len(failed)} of {len(sample)} rows failed and were skipped (see warnings above). "
                      f"The other {len(rows)} rows completed normally and will still be saved.")
        finally:
            print("Unloading models...")
            unload_models(models)

        # Both scoring passes are slow (one judge call per row) and now run by default, so a
        # failure or Ctrl+C partway through must never throw away the whole finished run --
        # rows scored so far keep their scores, the rest stay blank, and the run is saved either
        # way. scripts/backfill_quality_scores.py is idempotent and finishes whatever is blank.
        for scoring_pass in ([lambda: score_quality_pass_local(rows, score_quality_local)] if score_quality_local else []) + \
                            ([lambda: score_quality_pass(rows)] if score_quality else []):
            try:
                scoring_pass()
            except (Exception, KeyboardInterrupt) as e:
                print(f"\n  WARNING: quality scoring stopped early ({type(e).__name__}: {e}).")
                print("  The run itself is being saved anyway. To score whatever is still blank, run:")
                print("      python scripts/backfill_quality_scores.py")
                break

        df = pd.DataFrame(rows)
        df.to_csv(log_path, index=False)
        for unit, group in df.groupby("business_unit"):
            group.to_csv(f"{LOGS_DIR}/results_log_{unit}.csv", index=False)

        existing_history = lock.read()
        history = pd.concat([existing_history, df], ignore_index=True) if not existing_history.empty else df
        lock.write(history)
        print(f"\nRun {run_id} appended to {HISTORY_PATH} ({history['run_id'].nunique()} run(s) in history)")
    print(f"Released lock on {HISTORY_PATH}.")

    print(f"\nSaved {len(df)} results to {log_path} (plus one log per business unit)")
    print("\nDecisions by scenario type:")
    print(df.groupby("is_exception")["decision"].value_counts().unstack(fill_value=0).to_string())
    print("\nMedian total latency (ms) by decision:")
    print(df.groupby("decision")["total_latency_ms"].median().round(0).to_string())
    return df


def parse_cli_args(args: list) -> tuple:
    """Simple positional-or-flag parsing, kept deliberately lightweight rather than pulling
    in argparse for a few options. Returns (n, score_quality, local_judge_model, use_semantic_cache).

    Both the semantic cache and Claude quality evaluation are ON by default; each has an
    explicit opt-out flag:
        python -m src.pipeline 100                  # cache ON, quality evaluation ON
        python -m src.pipeline 100 --nocache        # skip the semantic cache
        python -m src.pipeline 100 --noquality      # skip the post-run Claude quality evaluation
        python -m src.pipeline 100 --nocache --noquality
        python -m src.pipeline 100 --score-quality-local llama3.2:3b   # extra, opt-in local judge
    """
    flags = {"--nocache", "--noquality", "--score-quality-local"}
    unknown = [a for a in args if a.startswith("--") and a not in flags]
    if unknown:
        raise SystemExit(f"Unknown option(s): {' '.join(unknown)}. "
                         f"Valid options: --nocache, --noquality, --score-quality-local <model>")
    local_judge_model = None
    if "--score-quality-local" in args:
        idx = args.index("--score-quality-local")
        if idx + 1 >= len(args):
            raise SystemExit("--score-quality-local needs a model name, e.g. --score-quality-local llama3.2:3b")
        local_judge_model = args[idx + 1]
    positional = [a for a in args if a not in flags and a != local_judge_model]
    n = int(positional[0]) if positional else N_SAMPLES
    return n, "--noquality" not in args, local_judge_model, "--nocache" not in args


def main(argv=None) -> None:
    n, score_quality, local_judge_model, use_semantic_cache = parse_cli_args(sys.argv[1:] if argv is None else argv)
    if use_semantic_cache:
        print("Semantic cache: ON (default). An escalation first checks for a close-enough past Claude answer "
              "before calling Claude again. Pass --nocache to turn it off.")
    else:
        print("Semantic cache: OFF (--nocache). Every escalation will call Claude.")
    if score_quality:
        print(f"Quality evaluation: ON (default). After processing, every one of the {n} rows' final answers -- whether "
              f"answered by Tier 1, served from the cache, or answered by Tier 2 -- gets one Claude judge call. Pass --noquality to skip.")
    else:
        print("Quality evaluation: OFF (--noquality). Rows are saved with blank quality columns; "
              "run scripts/backfill_quality_scores.py to fill them in later.")
    if local_judge_model:
        print(f"Local quality first-pass is ON via {local_judge_model}: cheap/free, but not the numbers to report -- see scripts/compare_quality_judges.py.")
    run_pipeline(load_random_sample(n), score_quality=score_quality, score_quality_local=local_judge_model,
                 use_semantic_cache=use_semantic_cache)


if __name__ == "__main__":
    main()
