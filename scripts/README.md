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
| See a run as one plain table (and spot rows that failed) | `export_run_report.py` |
| Make fresh customer questions so every run is different | `generate_questions.py` |
| Check whether the label judge can be trusted, and pick how to ask it | `evaluate_label_judge.py` |
| Fill in quality marks that are blank | `backfill_quality_scores.py`, then again with `--draft` |
| Train, or retrain, the learned router | `train_text_router.py` |
| Recover a run that did not reach the history file | `recover_missing_run.py` |
| Remove personal identifiers from logs already written | `redact_history.py` (dry run first) |
| Measure what the personal-data filter catches and misses | `evaluate_privacy_filter.py` |
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

### export_run_report.py: a run as one plain table

The pipeline prints a long block of text per question, which is hard to read or share. This script lays one run out as one row per
question, prints a summary, lists any problems with the command that fixes each, and saves a CSV and an Excel file to
`results/reports/` (not tracked by Git, because they hold question text).

It reads one of two sources:

* **The saved log** (default): the complete record of every row that was saved, with the full question and answers, the router's
  prediction, the decision, the timings and the quality marks.
* **Pasted console output** (`--console FILE`): only a printout, but the one place a row that **failed** still appears. A row that
  fails, for example when the Claude connection drops, is skipped and never saved, so the log leaves it out and the saved run
  understates how many questions went to Claude. Parsing the printout brings those rows back and counts them.

To capture console output while running: `python -m src.pipeline 100 | Tee-Object -FilePath run_console.txt` (Windows PowerShell)
or `python -m src.pipeline 100 | tee run_console.txt` (Mac and Linux). Or copy the text into a file.

| Option | Meaning | Default |
|---|---|---|
| `--run RUN_ID` | A run in the history | the newest |
| `--list` | List the 15 most recent runs and exit | off |
| `--console FILE` | Read pasted console output instead of the saved log | off |
| `--history PATH` | A different history file | `results/logs/results_history.csv` |
| `--out-dir DIR` | Where to write the CSV and Excel files | `results/reports` |
| `--no-files` | Print only; write nothing | off |
| `--rows N` | Print only the first N rows (the files always hold all rows) | all |

The Excel file needs `pip install openpyxl`; without it you still get the CSV. Older logs that recorded a cache hit as `ESCALATE` are
shown as `CACHE`, as on the dashboard.

### generate_questions.py: fresh customer questions from a separate model

Makes a batch of new customer questions with a general local model, so a test run does not keep drawing from the same fixed pool of
16,000. For each question it picks a request type (in the pool's own mix, or with a chosen share of exceptions), a tone, a few made-up
details (an amount, a card's last four digits, a merchant, a city, a time) and three existing questions of that type as examples, and
asks the model for one new message. It then rejects empty, too short or too long text, anything that is not a customer message, exact
duplicates (of the pool and of earlier batches) and near-duplicates by meaning, and checks that each question really expresses the intended
request type, by asking the model to choose among every intent of the business unit (or only of the category with `--label-scope category`). Each question is saved with its seed, model, prompt version, tone and label-check result, so the same seed repeats the batch.

Needs Ollama running and a general model, not one of your fine-tuned helpers: `ollama pull qwen2.5:7b`. It costs nothing and takes about
2 to 6 seconds per accepted question. Look at a few first: `python scripts/generate_questions.py --n 20 --show 20`.

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Questions to produce | 100 |
| `--seed N` | The batch seed (printed, so a batch can be repeated) | random |
| `--model NAME` | The generator model | `qwen2.5:7b` |
| `--out PATH` | The CSV to write | `data/generated/generated_questions_<date-time>.csv` |
| `--unit NAME` | Only `payments` or only `retail_bank` | both |
| `--exception-share X` | Share of questions from the exception categories, 0 to 1 | the pool's own mix |
| `--temperature X` | Randomness of the generator, 0 to 2 (higher is more varied and less reliable) | 0.9 |
| `--no-label-check` | Skip the check that each question expresses its intended request type | check on |
| `--label-scope NAME` | What the label check chooses among: `unit` (every intent of the business unit, 23 for payments or 27 for retail bank, which catches a question that belongs to another category) or `category` (only the same category's intents; faster but it passes almost anything for a category with one or two intents) | `unit` |
| `--per-slot-attempts N` | Tries per question slot before giving up on that request type. The types for the whole batch are drawn first, so the batch keeps the pool's mix; a type the model cannot produce is reported as unfilled and the batch is a little short, never padded with easier types | 6 |
| `--label-model NAME` | A separate model to do the label check, so a small model does not mark its own work. A larger general model you have downloaded (it must appear in `ollama list`) is slower per check but a steadier judge | the generator model |
| `--label-method NAME` | How the judge is asked: `examples` (default), `number`, `shuffled`, `name`, `two-step` or `category`. Measure them first with `scripts/evaluate_label_judge.py` |
| `--save-rejected PATH` | Also write every rejected candidate (its text, request type, the reason and, for a label mismatch, what the judge chose instead) to a CSV | off |
| `--dedup-pool` | Also refuse questions whose meaning is close to one in the 16,000-question pool (slower start) | exact copies only |
| `--dedup-threshold X` | How alike two questions must be to count as near-duplicates. Raise it to accept more | 0.92 |
| `--show N` | Print the first N generated questions | 0 |

You rarely need to run this script yourself: `python -m src.pipeline 100 --fresh` does the same thing as the first step of a run, using `data/generated/current_batch.csv` and replacing the previous batch. Use the script to look at a sample (`--show`), to keep a batch under its own name (`--out`), or to make a batch to share. Reading the report: after a run with label mismatches it prints the most common mix-ups (what a question was meant to be, and what the judge chose
instead) and the request types that passed the check least often. Read a few rows of the `--save-rejected` file. If the rejected questions look right and
the judge is the one that is wrong, a larger `--label-model` should help. If the questions are off-target, the generator or the prompt is the problem. A reply that
hits the length limit, or stops on a comma, colon or dash, is rejected as `cut_off`.

Then run the pipeline on a kept batch: `python -m src.pipeline 100 --questions data/generated/<file>.csv`. The summary prints the label-check
pass rate. A low rate means the model struggles with those request types, so treat the labels in that batch with care.

### evaluate_label_judge.py: can the label judge be trusted?

The generator keeps a question only if a judge model picks its intended request type from 23 or 27 options. In real runs the judge rejected most
good-looking questions and sometimes named unrelated types, so the generator's output cannot be trusted until the judge is measured. This script
tests the judge on pool questions whose labels are known to be right, so no generator is involved, no Claude call is made and nothing costs money.

It tests six ways of asking (`src/label_judge.py`): `number` (the original check), `shuffled` (options in random order), `name` (the model replies with the
option's text), `two-step` (the category first, then the intent), `examples` (each option shows a real example question) and `category` (the easier
old check, which is told the category and so is only a reference). For each model and method it prints the accuracy with a 95% interval, the share of
unreadable replies and the calls per question. It also shows whether the judge does worse on options near the end of the list, which request types it
gets wrong most often, the mix-ups that repeat (intended type against the type the judge chose), and a recommendation that says plainly when two methods cannot be told apart. With few questions per request type the list of wrong types is marked as rough; use `--n 150` or more for firmer patterns. Chance is about 4%. Below about 60% a judge is
too unreliable to decide which generated questions to keep. Start small: `python scripts/evaluate_label_judge.py --n 50 --methods number,examples`.
The full default (100 questions, six methods) is about 700 calls, so 15 to 30 minutes per model.

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Questions to test per method | 100 |
| `--model NAMES` | One judge model, or several separated by commas (each must be in `ollama list`), for example `qwen2.5:7b,llama3.1:8b` | `qwen2.5:7b` |
| `--methods NAMES` | The methods to test, separated by commas | all six |
| `--sample NAME` | `even` (the same number from every request type) or `pool` (random questions, as the pool is mixed) | `even` |
| `--seed N` | The seed for the choice of questions and shuffles; the same seed tests the same questions | 1 |
| `--show-wrong N` | Print N wrong answers for the best method of each model | 5 |
| `--out PATH` | Where to save every answer as a CSV | `results/analysis/label_judge_<date-time>.csv` |

Then use the best way of asking: `python scripts/generate_questions.py --label-method examples --label-model NAME`, or in a run
`python -m src.pipeline 100 --fresh --label-method examples --label-model NAME`.

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
| `--no-redact` | Send the question and answer to Claude exactly as logged. By default personal identifiers are replaced first, because older rows may still hold them |
| `--draft` | Mark Tier 1's own answer (`tier1_answer`) instead of the final answer. Writes `quality_draft_*`. Escalated rows get a Claude call; local rows copy their final marks for free |

Run it **without** `--draft` first (so local rows have final marks to copy), then **with** it. Only rows missing a mark are
judged, so repeat runs cost nothing. Cost: about $0.0012 per mark with Claude (about $1.20 per 1,000). Needs `ANTHROPIC_API_KEY`
unless `--judge-local` is used.

### redact_history.py: remove personal identifiers from logs already written

Rows logged before the personal-data filter existed hold the question and answers exactly as typed. This script applies the same
filter to the stored text (`query`, `tier1_answer`, `final_answer`, `cache_matched_query` and the quality notes) and fills the audit
columns `pii_filter`, `pii_count` and `pii_types`. Questions are filtered in full; answers only for customer-identifier types and for
values that appeared in the same row's question, so public helplines and addresses in answers survive.

| Option | Meaning |
|---|---|
| (none) | Dry run: reports how many identifiers it found and how many questions would change. Writes nothing |
| `--apply` | Writes the changes to `results_history.csv` and the latest-run files, after copying the originals to `results/logs/backup-redact-<time>/` |

Do not run it while a pipeline run is in progress. Redaction changes the text the router learns from, so retrain the router afterwards
(`python scripts/train_text_router.py`).

### evaluate_privacy_filter.py: measure what the filter catches and misses

Builds labelled test sentences (made-up identifiers in realistic formats), counts how often each identifier is removed, and checks
for false alarms on your real synthetic questions and on tricky look-alikes (amounts, dates, rates, reference numbers). Free-text names
and street addresses are included on purpose, as types the filter is not designed to catch, so the gap is measured.

| Option | Meaning | Default |
|---|---|---|
| `--n N` | Test cases per identifier type | 200 |
| `--seed N` | Random seed, so the result repeats | 7 |
| `--sample N` | Real synthetic questions to check for false alarms | 3000 |
| `--min-recall X` | Exit with code 1 if any covered type is below X (for a pass/fail check) | none |
| `--out PATH` | Write every test case and its result to a CSV | none |

Recall on generated cases is a ceiling, not proof: the generator writes identifiers in the formats the filter knows. Test on real,
labelled samples with the privacy team before relying on it.

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
