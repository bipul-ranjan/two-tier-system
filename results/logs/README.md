# results/logs/

Raw, per-query output from `python -m src.pipeline` — one row per query
processed, before any aggregation. This is your ground-truth evidence:
if a number in a chart or table looks wrong, this is where you check it.

## Files you'll see here after running the pipeline

| File | Contents |
|---|---|
| `results_log_payments.csv` | Every query processed for the `payments` business unit |
| `results_log_retail_bank.csv` | Every query processed for the `retail_bank` business unit |
| `results_log_combined.csv` | Both of the above, concatenated — this is the file `src/evaluate.py` reads by default |

## Columns

| Column | Meaning |
|---|---|
| `business_unit` | Which unit this query belongs to |
| `tier1_model` | Which local model handled it (phi3:mini or qwen2.5:1.5b) |
| `query` | The original query text |
| `tier1_intent` | What Tier 1 classified the query as |
| `tier1_confidence` | Tier 1's confidence score (0–1) |
| `decision` | `LOCAL` (resolved at Tier 1) or `ESCALATE` (sent to Tier 2) |
| `tier1_latency_ms` | Time Tier 1 took to respond |
| `tier2_latency_ms` | Time Tier 2 took to respond (blank if not escalated) |
| `tier2_input_tokens` / `tier2_output_tokens` | Token counts for the Tier 2 call (blank if not escalated) |
| `estimated_cost_usd` | Estimated API cost for this query (0 if resolved locally) |
| `final_answer` | The actual answer returned to the caller |

## Not tracked by Git

These files are excluded in `.gitignore` — they're regenerated every
time you run the pipeline, so there's nothing to gain from
version-controlling a specific run's output. What matters for your
dissertation is the code that produced them (`src/`), not the CSVs
themselves. Take screenshots or export specific tables into your
write-up if you need to cite exact numbers.
