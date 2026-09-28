"""
Main pipeline: pick random rows from the synthetic Bitext-format CSVs
(data/synthetic/synthetic_bitext_*.csv), run each through the two-tier
cascade, and log every decision for later evaluation.

- Business unit comes from the row's category (CARD/TRANSFER/ATM/FEES/
  PAYMENT_EXCEPTION -> payments; ACCOUNT/LOAN/PASSWORD/CONTACT/FIND/
  RETAIL_EXCEPTION -> retail_bank), so each query goes to the right
  assistant instead of an arbitrary split.
- Every row currently uses one confidence threshold, 0.7: below 0.7 escalates
  to Tier 2. Normal and exception rows have separate constants (NORMAL_THRESHOLD,
  EXCEPTION_THRESHOLD) so they can be split again later by changing one number.
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
from .tier2_escalate import ask_tier2, estimate_cost

SYNTHETIC_DIR = "data/synthetic"
LOGS_DIR = "results/logs"

N_SAMPLES = 100
SEED = None  # None = different random rows every run; set an int to reproduce a run

# Escalate anything scoring below this. Two constants, so normal and exception
# rows can be given different thresholds again later.
NORMAL_THRESHOLD = 0.7
EXCEPTION_THRESHOLD = 0.7

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


def process_query(query: str, business_unit: str, category: str, intent: str, is_exception: bool, run_id: str) -> dict:
    threshold = EXCEPTION_THRESHOLD if is_exception else NORMAL_THRESHOLD

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
        "estimated_cost_usd": 0.0,
        "final_answer": t1_result["answer"],
    }

    tier2_latency_ms = None
    if decision == "ESCALATE":
        t2_start = time.perf_counter()
        t2_result = ask_tier2(query)
        tier2_latency_ms = (time.perf_counter() - t2_start) * 1000
        row["tier2_latency_ms"] = round(tier2_latency_ms, 1)
        row["tier2_input_tokens"] = t2_result["input_tokens"]
        row["tier2_output_tokens"] = t2_result["output_tokens"]
        row["estimated_cost_usd"] = estimate_cost(t2_result["input_tokens"], t2_result["output_tokens"])
        row["final_answer"] = t2_result["answer"]

    total_latency_ms = (time.perf_counter() - t_query_start) * 1000
    row["total_latency_ms"] = round(total_latency_ms, 1)
    row["response_end_time"] = datetime.now().isoformat(timespec="milliseconds")

    tier2_text = f"{tier2_latency_ms:.0f} ms" if tier2_latency_ms is not None else "-"
    load_ms = row["tier1_load_ms"]
    load_text = f" (model load {load_ms:.0f} ms)" if load_ms else ""
    print(f"Latency        : Tier1 {tier1_latency_ms:.0f} ms{load_text} | Tier2 {tier2_text} | Total {total_latency_ms:.0f} ms")
    print("-" * 60)
    return row


def run_pipeline(sample: pd.DataFrame, log_path: str = None) -> pd.DataFrame:
    """Run every sampled row through the cascade. Writes one combined log
    (read by evaluate.py) plus one log per business unit.
    """
    if log_path is None:
        log_path = f"{LOGS_DIR}/results_log_combined.csv"
    os.makedirs(LOGS_DIR, exist_ok=True)
    guard_legacy_log()
    run_id = make_run_id(existing_run_ids())
    print(f"Run ID: {run_id}")

    models = sorted({get_model_for_unit(u) for u in sample["business_unit"].unique()})
    try:
        print("Preloading models...")
        load_models(models)
        warn_if_not_resident(models)
        print("-" * 60)
        rows = [
            process_query(r.instruction, r.business_unit, r.category, r.intent, bool(r.is_exception), run_id)
            for r in sample.itertuples(index=False)
        ]
    finally:
        print("Unloading models...")
        unload_models(models)

    df = pd.DataFrame(rows)
    df.to_csv(log_path, index=False)
    for unit, group in df.groupby("business_unit"):
        group.to_csv(f"{LOGS_DIR}/results_log_{unit}.csv", index=False)

    history = pd.concat([pd.read_csv(HISTORY_PATH), df], ignore_index=True) if os.path.exists(HISTORY_PATH) else df
    history.to_csv(HISTORY_PATH, index=False)
    print(f"\nRun {run_id} appended to {HISTORY_PATH} ({history['run_id'].nunique()} run(s) in history)")

    print(f"\nSaved {len(df)} results to {log_path} (plus one log per business unit)")
    print("\nDecisions by scenario type:")
    print(df.groupby("is_exception")["decision"].value_counts().unstack(fill_value=0).to_string())
    print("\nMedian total latency (ms) by decision:")
    print(df.groupby("decision")["total_latency_ms"].median().round(0).to_string())
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_SAMPLES
    run_pipeline(load_random_sample(n))
