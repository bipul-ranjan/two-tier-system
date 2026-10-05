# scripts/

Everything you run directly, as opposed to `src/`, which is the pipeline itself. Fourteen scripts, grouped by what
they are for. Run them all **from the project root** with the venv active:

```
python scripts/<name>.py [options]
```

No `-m` is needed. `src/` is a package whose files import each other, so it is run with `python -m src.<name>`.
Most scripts here are self-contained; the ones that read the logs or call Claude import a little from `src/`, and find it
because they are started from the project root. Every script starts with a header comment that says what it does, and the
ones with options print them with `--help`.

## Which script do I want?

| I want to... | Run |
|---|---|
| Get the practice questions the pipeline uses | They are already in `data/synthetic/`. To regenerate: `generate_synthetic_bitext.py` |
| Get training data for the two models | `fetch_bitext_banking.py`, then `generate_exception_training_data.py` for the retail unit |
| Run the pipeline and be sure every quality mark is filled in | `run_and_score.py` |
| Fill in quality marks that are blank | `backfill_quality_scores.py`, then again with `--draft` |
| Train, or retrain, the learned router | `train_text_router.py` |
| Recover a run that did not reach the history file | `recover_missing_run.py` |
| Reproduce the router research | `probe_reward_model.py`, then `train_router.py` |

---

## 1. Data preparation

### fetch_bitext_banking.py: training data for the two models

Downloads the Bitext retail-banking dataset (25,545 real instruction and response pairs) from Hugging Face and splits it
by category into the two business units.

```
python scripts/fetch_bitext_banking.py
```

| | |
|---|---|
| Needs | The `datasets` package; an internet connection |
| Writes | `data/raw/bitext_payments.jsonl` and `.csv` (CARD, TRANSFER, ATM, FEES) and `data/raw/bitext_retail_bank.jsonl` and `.csv` (ACCOUNT, LOAN, PASSWORD, CONTACT, FIND) |
| Re-run | Safe: if all four files exist it downloads nothing |
| Licence | CDLA-Sharing 1.0: free to use with attribution, and shared derivatives keep the licence. Cite: Bitext Innovations, "Bitext-retail-banking-llm-chatbot-training-dataset", 2024 |

### generate_synthetic_bitext.py: the questions the pipeline runs on

Generates synthetic customer-support data in the Bitext format (columns `instruction`, `category`, `intent`, `response`) as ten
CSV files, each a random mix of four scenario groups: payments normal, retail normal, payments exceptions (fraud, duplicate
charges, chargebacks, stuck or wrong transfers, compliance holds) and retail exceptions (frozen account, bereavement,
hardship, identity theft, complaints, vulnerable customers).

```
python scripts/generate_synthetic_bitext.py
```

| | |
|---|---|
| Settings | Constants at the top of the file: `SEED = 20260928`, `N_FILES = 10`, `ROWS_PER_FILE = 1600`, `OUT_DIR = data/synthetic` |
| Writes | `data/synthetic/synthetic_bitext_01.csv` to `_10.csv` (16,000 rows) and `synthetic_manifest.csv` (row count and group counts per file) |
| Re-run | The same `SEED` gives the same files, so re-running reproduces them. Change `SEED` for a different set. The ten files are committed, so you normally do not need to run this |
| Note | All bank policies, times and steps in the answers are invented and generic, not any real bank's |

### generate_exception_training_data.py: training examples for exception scenarios

The real Bitext data has no fraud, hardship or vulnerable-customer category, so a model trained on it alone has never seen
one. This builds synthetic training rows for those scenarios from the same templates, 350 per intent across 12 intents per
unit, **checked against every question in `data/synthetic/`** so that no training question can also appear as a pipeline test
question (a train/test leak).

```
python scripts/generate_exception_training_data.py
```

| | |
|---|---|
| Needs | `data/synthetic/synthetic_bitext_*.csv` to exist (it prints how many instructions it avoids duplicating; **if that says 0, stop**) |
| Writes | `data/raw/retail_exception_synthetic.jsonl` and `data/raw/payments_exception_synthetic.jsonl` |
| Run before | A retrain of the retail model (`training/train_retail.py` reads the retail file). The payments file was generated but is not used for training |

### fetch_data.py and convert_finqa_to_csv.py: the earliest datasets (optional)

Banking77 (labelled customer questions) and FinQA (financial questions with numeric answers) were used in the project's first
design. The live system does not need them; only the ground-truth helpers in `src/quality.py` do.

```
python scripts/fetch_data.py                 # data/raw/banking77_train.csv and _test.csv; clones FinQA into ./FinQA
python scripts/convert_finqa_to_csv.py       # data/raw/finqa_train.csv, finqa_dev.csv, finqa_test.csv (run after fetch_data.py)
```

Both are safe to re-run (anything already present is skipped). `fetch_data.py` uses the `mteb/banking77` mirror because the
original `PolyAI/banking77` repository uses a loading script that current versions of `datasets` refuse to run. FinQA needs Git.

---

## 2. Running and scoring

### run_and_score.py: the pipeline, then a safety net

```
python scripts/run_and_score.py N [pipeline options]
```

Runs `python -m src.pipeline N` with any options you add (`--nocache`, `--noquality`, `--threshold-router`,
`--score-quality-local MODEL`), then runs `backfill_quality_scores.py` and `backfill_quality_scores.py --draft`. The pipeline
already marks every row, so the two backfills normally find nothing and cost nothing. They are there for anything a run could
not finish: a failed marking call, or Ctrl+C partway through marking. It stops if the pipeline step fails, rather than scoring on
top of an unfinished run. `N` is required.

---

## 3. The learned router

### train_text_router.py: train the router the pipeline uses

```
python scripts/train_text_router.py [--history PATH] [--target-share FRACTION] [--model-path PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--history PATH` | The results history to learn from | `results/logs/results_history.csv` |
| `--target-share FRACTION` | Share of questions to escalate, 0.02 to 0.98. Sets the cut-point | 0.30 |
| `--model-path PATH` | Where to save the model | `models/router/text_probe_router.joblib` |

What it does, in three steps: (1) evaluates the router with five folds that each hold out whole intents, and prints it next to
Tier 1 confidence; (2) sets the cut-point as the percentile of those held-out predictions that escalates `--target-share` of
questions; (3) trains on every row and saves the model, `text_probe_router_info.json` and `what_it_learned.txt`.

| | |
|---|---|
| Needs | A history with at least 200 rows that have `quality_draft_overall`, covering at least 5 intents. If it is missing the column, it tells you to run the backfill with `--draft` |
| Time | Seconds |
| When | Before the first run; after changing a Tier 1 model; after adding a business unit; after the history grows a lot; when the pipeline reports a scikit-learn version mismatch |
| Note | The saved model only loads under the scikit-learn version that trained it. A fresh clone has no history: generate some with `python -m src.pipeline 300 --threshold-router` first |

---

## 4. Maintenance of the logs

All of these are safe to re-run, and say so when there is nothing to do.

### backfill_quality_scores.py: fill in blank quality marks

```
python scripts/backfill_quality_scores.py [--unit NAME] [--run RUN_ID] [--limit N] [--judge-local MODEL] [--draft]
```

| Option | Meaning |
|---|---|
| `--unit NAME` | Only rows of this business unit (`payments` or `retail_bank`) |
| `--run RUN_ID` | Only rows of this run |
| `--limit N` | Mark at most N rows this pass (start small to check cost, then expand) |
| `--judge-local MODEL` | Mark with this local Ollama model instead of Claude. Writes the separate `quality_local_*` columns, never the `quality_*` ones |
| `--draft` | Mark Tier 1's own answer (`tier1_answer`) instead of the final answer. Writes `quality_draft_*`. Escalated rows get a Claude call; local rows copy their final marks for free |

Run it **without** `--draft` first (so local rows have final marks to copy), then **with** it. Only rows missing a mark are
judged, so repeat runs cost nothing. Cost: about $0.0012 per mark with Claude (about $1.20 per 1,000). Needs `ANTHROPIC_API_KEY`
unless `--judge-local` is used.

### backfill_cache_decision.py: relabel old cache hits

```
python scripts/backfill_cache_decision.py
```

Before the `CACHE` decision existed, a cache hit was logged as `ESCALATE` with `cache_hit` true. This changes those rows to
`CACHE` in the history and the latest-run logs. The dashboard already reads the old form as `CACHE`, so skipping this breaks nothing.

### backfill_prompt_version.py: label which Claude prompt was used

```
python scripts/backfill_prompt_version.py
```

Fills `tier2_prompt_version` for escalated rows that predate the column: `v1-generic` before the run named in `CUTOFF_RUN_ID` (at the
top of the script), `v2-persona` from it onwards. Edit `CUTOFF_RUN_ID` if you ever need to repeat this for another prompt change.

### backfill_run_id.py: add run ids to old logs

```
python scripts/backfill_run_id.py                          # uses the id "run-default"
python scripts/backfill_run_id.py "run-28-Sep-26-06:29 PM"  # or any id you choose
```

Adds a `run_id` column to logs from before run ids existed and seeds `results_history.csv` with them. Run it once, before the next
pipeline run. The pipeline refuses to overwrite a log that has no `run_id`, and points you here.

### recover_missing_run.py: rescue a run that was not saved to the history

```
python scripts/recover_missing_run.py
```

If `results_history.csv` was open elsewhere (Excel) when a run finished, the run is in `results_log_combined.csv` but not in the
history. Close the other program and run this once: it appends the run, unless it is already there.

---

## 5. Research scripts (optional)

These produced the evidence for choosing the router. They need a history with Tier 1 quality marks (`quality_draft_overall`) and
PyTorch (installed with `sentence-transformers`).

### probe_reward_model.py: can a pretrained reward model rank answers like Claude does?

```
python scripts/probe_reward_model.py [--n N] [--model ID] [--history PATH] [--out-dir DIR] [--bf16]
```

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Rows to score, split evenly across the business units | 500 |
| `--model ID` | A Hugging Face reward model | `Skywork/Skywork-Reward-V2-Qwen3-0.6B` |
| `--history PATH` | The results history | `results/logs/results_history.csv` |
| `--out-dir DIR` | Where scores are saved | `results/analysis` |
| `--bf16` | Load in half precision: half the memory, slower on some processors | off |

It scores (question, Tier 1 answer) pairs with no training, and compares the result with Tier 1 confidence and with the text probe
on the same rows. About 2.9 seconds a row on a processor (500 rows is roughly 25 minutes) and 2.4 GB of memory; the first run downloads about
1.2 GB. Scores are saved as it goes, so Ctrl+C is safe: run the same command to continue. Finding: strong in retail bank, weak in
payments, blind to safety, so it was not adopted.

### train_router.py: do MiniLM-based routers beat confidence and the text probe?

```
python scripts/train_router.py [--mode frozen|finetune] [options]
```

| Option | Meaning | Default |
|---|---|---|
| `--mode` | `frozen`: MiniLM reads each text once and a ridge model learns on top. `finetune`: MiniLM itself is trained | `frozen` |
| `--base-model NAME` | The encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| `--history PATH`, `--out-dir DIR` | Input history and where fold predictions are saved | history, `results/analysis` |
| `--folds K` | Held-out-intent folds | 5 |
| `--stop-after N` | Finetune: stop after N new folds (a first look) | all |
| `--report-only` | Finetune: train nothing, report on the folds already saved (use the same settings as the run that saved them) | off |
| `--epochs E`, `--max-length L`, `--batch-size B`, `--lr LR` | Finetune settings | 3, 256, 16, 5e-5 |
| `--seed S` | Random seed | 0 |
| `--n N` | Use only N random rows (a quick smoke test) | all |

Frozen mode takes about 7 minutes. Finetune mode on a processor takes about an hour per fold (about five hours for five folds; a
GPU, such as a Colab session, takes minutes). Each fold is saved as it finishes, so Ctrl+C is safe and a re-run continues; the
saved files are named after the settings, so changing `--epochs` or `--max-length` starts afresh. Finding: neither mode beat the text
probe, which is why the text probe became the router. The saved predictions are in `results/analysis/`.
