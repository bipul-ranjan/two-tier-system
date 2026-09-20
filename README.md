# Two-Tier System

Distributed in-house SLM (per business unit) with a single shared
escalation LLM, for cost optimisation, security hardening, and reduced
latency.

## Project Structure

```
two-tier-system/
├── src/                    All pipeline code — a proper Python package
│   ├── __init__.py
│   ├── config.py           Business-unit → model mapping
│   ├── tier1.py             Tier 1 SLM (Ollama) classification + confidence
│   ├── router.py            Confidence-threshold routing decision
│   ├── tier2_escalate.py    Tier 2 LLM API call + cost estimation
│   ├── pipeline.py          Runs the full cascade over a batch of queries
│   └── evaluate.py          Computes metrics, writes results tables + plots
├── data/
│   └── raw/                 Downloaded datasets (banking77_test.csv, etc.)
├── results/
│   ├── logs/                 Per-query run logs (results_log_*.csv)
│   ├── tables/                Summary metrics (summary_metrics.csv, etc.)
│   └── plots/                  Generated charts (tradeoff_curve.png)
├── tests/
│   └── test_router.py         Unit tests for the routing logic
├── venv/                      Virtual environment (not tracked by Git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
```

Set your Tier 2 API key:
```
setx OPENAI_API_KEY "your-key-here"
```
(Close and reopen your terminal after `setx` — it only applies to new sessions.)

## Getting the dataset

```
python -c "from datasets import load_dataset; ds = load_dataset('PolyAI/banking77'); ds['test'].to_pandas().to_csv('data/raw/banking77_test.csv', index=False)"
```

## Running

Everything is run as a module from the project root — not by `cd`-ing into `src/`:

```
python -m src.config       # sanity check: prints business unit → model mapping
python -m src.tier1        # sanity check: one test query to Phi-3-mini
python -m src.router       # sanity check: routing decisions at 5 confidence levels
python -m src.pipeline     # runs the full cascade for both business units
python -m src.evaluate     # computes summary + per-business-unit metrics
```

## Running tests

```
pip install pytest
pytest tests/
```

## Output

After running `pipeline.py` and `evaluate.py`, you'll have:
- `results/logs/results_log_payments.csv`, `results_log_retail_bank.csv`, `results_log_combined.csv`
- `results/tables/summary_metrics.csv`, `business_unit_metrics.csv`
- `results/plots/tradeoff_curve.png` (if you call `plot_tradeoff()` with your own threshold sweep)

These are exactly the files the live dashboard (separate `live-dashboard/` project) reads from — point its `TOPIC1_RESULTS_DIR` environment variable at this project's `results/tables` folder.
