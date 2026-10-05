# results/tables/

Summary tables for the **latest run**, written by `python -m src.evaluate` from `results/logs/results_log_combined.csv`. Not tracked by Git.

| File | Contains |
|---|---|
| `summary_metrics.csv` | One row: `resolution_rate_tier1` (the share decided `LOCAL`), `avg_latency_local_ms`, `avg_latency_escalated_ms` (Tier 1 plus Claude time, for `ESCALATE` rows), `total_cost_usd`, `estimated_all_llm_cost_usd`, `cost_reduction_pct` |
| `business_unit_metrics.csv` | One row per business unit: `tier1_model`, `query_count`, `resolution_rate_tier1`, the two latencies, `total_cost_usd` |

How to read them: the all-Claude baseline is the number of rows times the average cost of a row that really called Claude. Rows served
from the cache (`CACHE`) cost nothing and are not counted as escalated here, so they lower the cost without appearing in the escalated
latency. The dashboard shows the three-way split directly and covers every run, not just the latest.
