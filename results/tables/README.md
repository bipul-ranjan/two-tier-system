# results/tables/

Aggregated metrics, produced by `python -m src.evaluate` from the raw
logs in `results/logs/`. These are the numbers that actually go into
your dissertation write-up and the live dashboard — the logs are
evidence, these tables are the findings.

## Files

| File | Produced by | Contents |
|---|---|---|
| `summary_metrics.csv` | `evaluate.summarize()` | One row: overall resolution rate, average latency (local vs. escalated), total cost, cost reduction vs. an all-LLM baseline |
| `business_unit_metrics.csv` | `evaluate.summarize_by_business_unit()` | One row per business unit: same metrics, broken out so you can compare `payments` (phi3:mini) against `retail_bank` (qwen2.5:1.5b) directly |

## This is what the live dashboard reads

If you're running the separate `live-dashboard` project alongside this
one, point its backend at this exact folder:

```
export TOPIC1_RESULTS_DIR=/path/to/two-tier-system/results/tables
```

The dashboard shows a green "LIVE DATA" badge once it finds real files
here, and orange "PLACEHOLDER" data until then — so an empty version of
this folder isn't an error, it's just what things look like before
you've run `evaluate.py` for the first time.

## Not tracked by Git

Same reasoning as `results/logs/` — regenerated on every run, so
there's nothing worth version-controlling here directly.
