"""
Compute the headline metrics from a results log: resolution rate, latency
comparison, cost vs. an all-LLM baseline, and a cost-accuracy trade-off
curve across multiple thresholds.

Part of the src/ package — run from the project root with:
    python -m src.evaluate
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def summarize(log_path: str = "results/logs/results_log_combined.csv", out_dir: str = "results/tables") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(log_path)

    resolution_rate = (df["decision"] == "LOCAL").mean()
    avg_latency_local = df.loc[df["decision"] == "LOCAL", "tier1_latency_ms"].mean()

    escalated = df[df["decision"] == "ESCALATE"]
    avg_latency_escalated = (escalated["tier1_latency_ms"] + escalated["tier2_latency_ms"]).mean()

    total_cost = df["estimated_cost_usd"].sum()
    avg_escalation_cost = escalated["estimated_cost_usd"].mean() if len(escalated) else 0.0
    all_llm_baseline_cost = len(df) * avg_escalation_cost

    summary = {
        "resolution_rate_tier1": round(resolution_rate, 3),
        "avg_latency_local_ms": round(avg_latency_local, 1) if pd.notna(avg_latency_local) else None,
        "avg_latency_escalated_ms": round(avg_latency_escalated, 1) if pd.notna(avg_latency_escalated) else None,
        "total_cost_usd": round(total_cost, 4),
        "estimated_all_llm_cost_usd": round(all_llm_baseline_cost, 4),
        "cost_reduction_pct": round((1 - total_cost / all_llm_baseline_cost) * 100, 1) if all_llm_baseline_cost else None,
    }

    pd.DataFrame([summary]).to_csv(f"{out_dir}/summary_metrics.csv", index=False)
    print(summary)
    return summary


def summarize_by_business_unit(log_path: str = "results/logs/results_log_combined.csv", out_dir: str = "results/tables") -> pd.DataFrame:
    """Break out the same headline metrics per business unit, so you can
    compare e.g. payments (Phi-3-mini) against retail_bank (Qwen2.5-1.5B)
    directly, rather than only seeing an aggregate number.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(log_path)

    rows = []
    for unit, group in df.groupby("business_unit"):
        resolution_rate = (group["decision"] == "LOCAL").mean()
        avg_latency_local = group.loc[group["decision"] == "LOCAL", "tier1_latency_ms"].mean()
        escalated = group[group["decision"] == "ESCALATE"]
        avg_latency_escalated = (
            (escalated["tier1_latency_ms"] + escalated["tier2_latency_ms"]).mean()
            if len(escalated) else None
        )
        total_cost = group["estimated_cost_usd"].sum()
        model_used = group["tier1_model"].iloc[0] if "tier1_model" in group.columns else "unknown"

        rows.append({
            "business_unit": unit,
            "tier1_model": model_used,
            "query_count": len(group),
            "resolution_rate_tier1": round(resolution_rate, 3),
            "avg_latency_local_ms": round(avg_latency_local, 1) if pd.notna(avg_latency_local) else None,
            "avg_latency_escalated_ms": round(avg_latency_escalated, 1) if avg_latency_escalated and pd.notna(avg_latency_escalated) else None,
            "total_cost_usd": round(total_cost, 4),
        })

    result = pd.DataFrame(rows)
    result.to_csv(f"{out_dir}/business_unit_metrics.csv", index=False)
    print(result)
    return result


def plot_tradeoff(threshold_results: dict, out_path: str = "results/plots/tradeoff_curve.png"):
    """threshold_results: {threshold_value: (accuracy, cost_per_1000)}"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    thresholds = sorted(threshold_results.keys())
    accuracies = [threshold_results[t][0] for t in thresholds]
    costs = [threshold_results[t][1] for t in thresholds]

    plt.figure(figsize=(6, 4))
    plt.plot(costs, accuracies, marker="o")
    for tau, c, a in zip(thresholds, costs, accuracies):
        plt.annotate(f"\u03c4={tau}", (c, a))
    plt.xlabel("Cost per 1,000 queries (USD)")
    plt.ylabel("Accuracy")
    plt.title("Cost\u2013Accuracy Trade-off")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved trade-off plot to {out_path}")


if __name__ == "__main__":
    LOG = "results/logs/results_log_combined.csv"
    summarize(LOG)
    summarize_by_business_unit(LOG)
