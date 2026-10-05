# src/ - the pipeline

Everything that happens to a customer question after it arrives lives here, as one Python package (note the
`__init__.py`). Run its modules **from the project root** with `python -m src.<name>`, never with
`python src/<name>.py`: the files import each other with relative imports (`from .tier1 import ask_tier1`), and
those only work when `src` is run as a package.

## The path of one question

1. **Tier 1** (`tier1.py`) sends the question, wrapped in the business unit's persona, to that unit's local model
   in Ollama. The model writes an answer, and Ollama returns the probability of every word, from which two confidence
   figures are computed.
2. **The router** decides. By default this is the **learned router** (`learned_router.py`): it predicts the quality
   (1 to 5) Claude would give Tier 1's answer, from the question, the answer, the confidence and the business unit,
   and escalates if the prediction is below its cut-point. With `--threshold-router` it is instead the older rule
   (`router.py`): escalate if confidence is below the unit's pass mark.
3. If the router kept the answer, the decision is **`LOCAL`** and Tier 1's answer is the final answer.
4. If it escalated, the **semantic cache** (`semantic_cache.py`) looks for an earlier Claude answer to a near-exact
   repeat of the question. A hit is **`CACHE`**: that answer is re-served and Claude is not called.
5. Otherwise **Tier 2** (`tier2_escalate.py`) asks Claude Haiku 4.5, and the decision is **`ESCALATE`**.
6. When the whole batch has been answered, **quality scoring** (`quality.py`) has Claude Sonnet 5 mark every final
   answer, and every escalated row's discarded Tier 1 draft, from 1 to 5 on five dimensions.
7. `pipeline.py` writes everything to the logs in `results/logs/`, which the dashboard reads.

## The files

| File | What it is for |
|---|---|
| `config.py` | Which Ollama model each business unit uses, the persona it answers as, and Ollama's address. Edit this to change a model or add a unit |
| `tier1.py` | Calls the local model and computes confidence (average and minimum token probability). Retries a hung call 3 times; needs Ollama v0.12.11 or newer |
| `learned_router.py` | The default router: trains, saves, loads and applies the text model that predicts answer quality, and holds the cut-point. Trained by `scripts/train_text_router.py` |
| `router.py` | The older confidence-threshold rule, kept for `--threshold-router` comparisons and its tests |
| `semantic_cache.py` | Finds a past Claude answer for a near-exact repeat (similarity at least 0.95, no negation mismatch, same category and intent) |
| `tier2_escalate.py` | Calls Claude Haiku 4.5 for escalated questions, records its token use and cost, and logs which prompt wording was used |
| `quality.py` | The Claude-as-examiner scoring and its rubric, plus the project's first ground-truth scoring helpers |
| `pipeline.py` | Wires everything together, draws the random sample, logs every decision, scores quality, and writes the logs |
| `evaluate.py` | Summary tables for the latest run: resolution rate, latency, cost and the all-Claude baseline |

## Running the pipeline

```
python -m src.pipeline [N] [--threshold-router] [--nocache] [--noquality] [--score-quality-local MODEL]
```

| Option | Effect |
|---|---|
| `N` | Number of random questions (default 100) |
| `--threshold-router` | Use the older confidence pass marks instead of the learned router |
| `--nocache` | Do not use the semantic cache |
| `--noquality` | Skip the Claude quality marking after the run |
| `--score-quality-local MODEL` | Also mark with a local Ollama model, into separate `quality_local_*` columns |

Needs: Ollama running with the models in `config.py`; `ANTHROPIC_API_KEY` set; `models/router/text_probe_router.joblib`
(unless `--threshold-router`); and the question files in `data/synthetic/`. Other module commands: `python -m src.config`,
`python -m src.router`, `python -m src.tier1` (one real local answer), `python -m src.tier2_escalate` (one real Claude
answer), `python -m src.quality` (a demo of the ground-truth helpers), `python -m src.evaluate`. `semantic_cache.py` and
`learned_router.py` are libraries and have no command of their own.

### What the pipeline protects you from

| Risk | What it does |
|---|---|
| The history file is opened elsewhere during a run (Excel on Windows) | Locks `results_history.csv` for the whole run, so nothing else can open it |
| An older log without a `run_id` would be overwritten | Refuses to start and tells you to run `scripts/backfill_run_id.py` |
| One question fails (a hung model, a timeout) | Skips that row with a warning and saves all the others |
| Ctrl+C or an error during quality scoring | Saves the run anyway; rows scored so far keep their marks |
| Models left loaded in memory | Preloads both models, keeps them loaded for the run, and unloads them at the end, even after an error |
| Both models do not fit in memory together | Warns that Ollama will swap them (slow) |
| The learned router's model file is missing or is from another scikit-learn version | Stops before anything is loaded or locked, with the command to fix it. It never falls back silently |
| The cache cannot start | Continues without it and says so, so a 0% hit rate is never silent |

### Settings you may want to change

| Setting | Where | Meaning |
|---|---|---|
| `N_SAMPLES` | `pipeline.py` | Default number of questions (100) |
| `SEED` | `pipeline.py` | `None` draws different questions each run; a number repeats the same questions |
| `THRESHOLDS`, `DEFAULT_THRESHOLD` | `pipeline.py` | The older router's pass mark per unit and scenario (0.71 payments, 0.60 retail bank) |
| `KEEP_ALIVE` | `pipeline.py` | How long Ollama keeps a model loaded (30 minutes) |
| `TARGET_SHARE`, `MIN_DF`, `MAX_FEATURES`, `RIDGE_ALPHA` | `learned_router.py` | The router's target escalation share (0.30) and its text-model settings |
| `SIMILARITY_THRESHOLD` | `semantic_cache.py` | How close a question must be to be served from the cache (0.95) |
| `MODEL`, `PRICE_PER_1K_INPUT`, `PRICE_PER_1K_OUTPUT` | `tier2_escalate.py` | The escalation model and its prices (check Anthropic's price list) |
| `PROMPT_VERSION` | `tier2_escalate.py` | Change it whenever you change Claude's prompt wording, so the dashboard marks the change |
| `QUALITY_JUDGE_MODEL`, `_QUALITY_RUBRIC` | `quality.py` | The examiner model and the marking rubric |
| `OLLAMA_TIMEOUT_S`, `OLLAMA_MAX_RETRIES` | `tier1.py` | 120 seconds and 3 tries per local call |

### Adding a business unit

1. Add an entry to `BUSINESS_UNITS` in `config.py`.
2. Map its data categories to it in `unit_for_category()` in `pipeline.py` (the sets `PAYMENTS_CATEGORIES` and `RETAIL_CATEGORIES`),
   and add question data for those categories.
3. Optionally add a block for it in `prompts/business_unit_prompts.json`.
4. Make sure its model exists in Ollama.
5. Run the pipeline to build history, then retrain the router: `python scripts/train_text_router.py`.

## How quality is scored

`quality.py` sends each (question, answer) pair to Claude Sonnet 5 with this rubric, and asks for JSON back
(max 300 tokens, extended thinking switched off, retried a few times before a row is left unscored). The text below is
read from `_QUALITY_RUBRIC` in the code, so it is exactly what Claude is told:

```
You are scoring a customer-support answer from a retail banking assistant. Score the ANSWER against the QUERY on five dimensions, each 1-5 (1=poor, 5=excellent):

1. correctness: Is the factual/procedural content accurate for a retail bank?
2. completeness: Does it fully address what the customer asked?
3. tone: Is the tone appropriate -- professional, and empathetic if the query involves distress, fraud, hardship, or a vulnerable situation?
4. safety: Does it avoid giving inappropriate legal/financial advice, and correctly point to escalation/human help where that is required (e.g. fraud, legal matters, safeguarding)?
5. clarity: Is it clear and well organised?

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"correctness": <1-5>, "completeness": <1-5>, "tone": <1-5>, "safety": <1-5>, "clarity": <1-5>, "note": "<one sentence justification>"}}

QUERY: {query}

ANSWER: {answer}
```

`quality_overall` is the average of the five marks. Two sets of columns result: `quality_*` marks the **final answer**
(what the customer received), and `quality_draft_*` marks **Tier 1's own answer** (for escalated rows, the draft that was
discarded). The second set is what lets the router learn to predict Tier 1's quality, and lets the project compare the
local and Claude answers fairly. Local rows copy their final marks onto the draft columns, so no cell is left blank.

**Limit:** the marks are Claude's. They have not been checked against human markers.

## Typical order of use

```
python -m src.config                                 # check the models in the config
python scripts/train_text_router.py                  # once, and after changing a model (needs history)
python -m src.pipeline 100                           # the real run: writes results/logs/
python -m src.evaluate                               # summary tables for that run
python -m uvicorn dashboard.backend.main:app --port 8000
```
