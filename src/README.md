# src/

All pipeline code lives here, as a proper Python package (note the
`__init__.py`). This is why every script is run with `python -m src.<name>`
from the project root, rather than `python <name>.py` from inside this
folder — the `-m` flag is what makes the relative imports between these
files (`from .tier1 import ask_tier1`, etc.) resolve correctly.

## Files, in the order data actually flows through them

| File | Role |
|---|---|
| `config.py` | Maps each business unit to the local model it uses (payments → phi3:mini, retail_bank → qwen2.5:1.5b). Edit this file to add a new business unit or change which model one uses. |
| `tier1.py` | Sends a query to the local SLM (via Ollama) and parses out the intent + confidence score. This is the "cheap, fast" tier — it never leaves your machine. |
| `router.py` | Takes Tier 1's confidence score and decides: resolve locally (`LOCAL`) or send it up to Tier 2 (`ESCALATE`). The whole cost-saving argument of this project lives in this one small file. |
| `tier2_escalate.py` | Sends escalated queries to a paid LLM API (e.g. GPT-4o-mini) and estimates the token cost of that call. This is the "expensive, only-when-needed" tier. |
| `pipeline.py` | Wires the above three together — runs a batch of queries through Tier 1, routes them, escalates where needed, and logs every decision to `results/logs/`. Run this to actually generate data. |
| `evaluate.py` | Reads the logs `pipeline.py` produced and computes the headline numbers (resolution rate, cost reduction, latency, per-business-unit breakdown) into `results/tables/`. Run this after `pipeline.py`, not instead of it. |

## Typical run order

```
python -m src.config       # sanity check only — no Ollama call
python -m src.tier1        # sanity check — one real Ollama call
python -m src.router       # sanity check only — no dependencies
python -m src.pipeline     # the real run — produces results/logs/*.csv
python -m src.evaluate     # produces results/tables/*.csv from those logs
```
