"""
Main pipeline: run the full two-tier cascade over a batch of queries,
logging every decision for later evaluation.
"""
import time
from datetime import datetime
import pandas as pd

from .tier1 import ask_tier1
from .router import route
from .tier2_escalate import ask_tier2, estimate_cost

DATA_DIR = "data/raw"
LOGS_DIR = "results/logs"


def run_pipeline(queries: list, business_unit: str = "payments", log_path: str = None) -> pd.DataFrame:
    if log_path is None:
        log_path = f"{LOGS_DIR}/results_log_{business_unit}.csv"

    rows = []
    for query in queries:
        start_time = datetime.now()
        start = time.time()
        t1_result = ask_tier1(query, business_unit=business_unit)
        tier1_latency_ms = (time.time() - start) * 1000

        decision = route(t1_result["confidence_score"])

        print(f"Start Time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Business Unit  : {t1_result['business_unit']} ({t1_result['assistant_name']})")
        print(f"Query          : {query}")
        print(f"Result         : {t1_result['intent']}")
        print(f"Confidence     : {t1_result['confidence_score']} ({t1_result['confidence_label']})")
        print(f"Decision       : {decision}")
        print("-" * 60)

        row = {
            "business_unit": business_unit,
            "assistant_name": t1_result["assistant_name"],
            "tier1_model": t1_result["model"],
            "query": query,
            "tier1_intent": t1_result["intent"],
            "tier1_confidence": t1_result["confidence_score"],
            "decision": decision,
            "tier1_latency_ms": round(tier1_latency_ms, 1),
            "tier2_latency_ms": None,
            "tier2_input_tokens": None,
            "tier2_output_tokens": None,
            "estimated_cost_usd": 0.0,
            "final_answer": t1_result["intent"],
        }

        if decision == "ESCALATE":
            start2 = time.time()
            t2_result = ask_tier2(query)
            tier2_latency_ms = (time.time() - start2) * 1000
            row["tier2_latency_ms"] = round(tier2_latency_ms, 1)
            row["tier2_input_tokens"] = t2_result["input_tokens"]
            row["tier2_output_tokens"] = t2_result["output_tokens"]
            row["estimated_cost_usd"] = estimate_cost(t2_result["input_tokens"], t2_result["output_tokens"])
            row["final_answer"] = t2_result["answer"]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(log_path, index=False)
    print(f"\nSaved {len(df)} results to {log_path}")
    return df


def run_all_business_units(queries_by_unit: dict, log_path: str = None) -> pd.DataFrame:
    """Run the pipeline separately for each business unit and combine into
    one log file -- e.g. queries_by_unit = {"payments": [...], "retail_bank": [...]}
    """
    if log_path is None:
        log_path = f"{LOGS_DIR}/results_log_combined.csv"

    all_rows = []
    for unit, queries in queries_by_unit.items():
        df = run_pipeline(queries, business_unit=unit)
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(log_path, index=False)
    print(f"\nCombined {len(combined)} results from {len(queries_by_unit)} business units into {log_path}")
    return combined


if __name__ == "__main__":
    all_queries = pd.read_csv(f"{DATA_DIR}/banking77_train.csv").sample(100, random_state=42)["text"].tolist()
    half = len(all_queries) // 2
    queries_by_unit = {
        "payments": all_queries[:half],
        "retail_bank": all_queries[half:],
    }
    run_all_business_units(queries_by_unit)
