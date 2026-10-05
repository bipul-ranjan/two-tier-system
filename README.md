# Two-Tier System: a cost-aware banking assistant

A banking assistant that answers most customer questions **on your own computer** with small fine-tuned
language models, and calls a paid model (Claude) only when a learned router predicts that the local answer is
not good enough. It is the practical part of an MS research project at Liverpool John Moores University, and it
is built to be measured: every decision, timing, cost and quality mark is logged, and a live dashboard shows them.

> **Status (5 October 2026).** The system is complete and in use: two business units, a learned router, a
> semantic cache, Claude as examiner, and a dashboard. All questions are **synthetic** (no real customer data),
> and Claude's quality marks have **not** been validated against human markers. Read "Limits and responsible use"
> (section 14) before quoting any number from this project.

## Contents

1. [What it does](#1-what-it-does)
2. [What the project found](#2-what-the-project-found)
3. [What is in this repository](#3-what-is-in-this-repository)
4. [What you need](#4-what-you-need)
5. [Installation, step by step](#5-installation-step-by-step)
6. [Running the system: every command and option](#6-running-the-system-every-command-and-option)
7. [The learned router: training and retraining](#7-the-learned-router-training-and-retraining)
8. [Quality scoring, the cache and the maintenance scripts](#8-quality-scoring-the-cache-and-the-maintenance-scripts)
9. [Training your own models on Google Colab](#9-training-your-own-models-on-google-colab)
10. [Research scripts](#10-research-scripts)
11. [Tests](#11-tests)
12. [Documentation map](#12-documentation-map)
13. [Troubleshooting](#13-troubleshooting)
14. [Limits and responsible use](#14-limits-and-responsible-use)
15. [Data, models and licences](#15-data-models-and-licences)

---

## 1. What it does

```
 customer question (payments or retail banking)
        |
        v
 Tier 1: the business unit's own small model, running in Ollama on your computer
        |     writes an answer and reports how confident it was
        v
 Router: predicts how good that answer is (1 to 5) from the question, the answer,
        |     Tier 1's confidence and the business unit
        |
        +-- predicted quality at or above the cut-point --> LOCAL     send Tier 1's answer        costs $0
        |
        +-- below the cut-point --> semantic cache: is this a near-exact repeat of an earlier question?
                                       +-- yes --> CACHE     re-serve the earlier Claude answer     costs $0
                                       +-- no  --> ESCALATE  Claude Haiku 4.5 writes the answer     about $0.002

 afterwards: Claude Sonnet 5 marks every answer, and every discarded Tier 1 draft, from 1 to 5 on five
 dimensions (correctness, completeness, tone, safety, clarity); every row is logged; the dashboard shows it all
```

Every question ends as exactly one of three outcomes, written to the `decision` column of the log:

| Decision | What happened | Claude called? |
|---|---|---|
| `LOCAL` | The router kept Tier 1's answer and it was sent as written | No |
| `CACHE` | The router escalated it, but a near-identical earlier question had a Claude answer, which was re-served | No |
| `ESCALATE` | The router escalated it and Claude wrote the answer | Yes |

The two business units are **payments** (card, transfer, ATM and fee questions, answered by a fine-tuned
Phi-3-mini) and **retail bank** (account, loan, password, contact and branch questions, answered by a fine-tuned
Qwen2.5-1.5B). Each also has "exception" scenarios (fraud, hardship, bereavement and similar) that are harder
and are escalated more often.

The router used to compare Tier 1's confidence with a pass mark. It was replaced on 5 October 2026 by a learned
router (section 7), and the old rule remains available with `--threshold-router` for comparison.

## 2. What the project found

All figures come from the project's own logs and held-out tests (whole question types kept out of training).
They describe **synthetic** data and a **Claude-marked** quality score. The detailed record is in the two
project documents listed in section 12.

| Finding | Number |
|---|---|
| Answers logged and marked up to 4 October | 6,310, from 16 runs and 50 question types |
| Average quality of Tier 1 answers, and of Claude's answers (out of 5) | 3.54 and 4.38 |
| How well Tier 1's confidence predicts quality (correlation, within each unit) | +0.33 payments, +0.095 retail bank |
| How well the adopted router predicts quality (mean of the two units) | +0.477 (a fine-tuned MiniLM reached +0.472, a frozen one +0.362) |
| Quality delivered at 30% escalation: random, confidence, adopted router, best possible | 3.79, 3.85, 3.90, 4.02 |
| The first real run of the live router: answered locally, from the cache, by Claude | 69, 23 and 8 of 100 questions |
| Claude's cost per escalated answer | about $0.0014 |

In plain words: how sure a small model sounds says very little about whether it is right, and a simple router that
reads the question and the answer picks out the answers worth escalating about twice as well.

## 3. What is in this repository

```
two-tier-system/
|-- README.md                  this file
|-- requirements.txt           Python packages for running the system
|-- User-Manual/               the step-by-step manual for beginners (Markdown and PDF)
|-- src/                       the pipeline: Tier 1, router, cache, Tier 2, quality scoring, logging
|-- scripts/                   data preparation, maintenance, router training and research scripts
|-- training/                  fine-tuning scripts for Google Colab (need a GPU)
|-- models/router/             the trained learned router and its list of "words it weighs"
|-- prompts/                   extra per-business-unit instructions added to Tier 1's prompt
|-- data/synthetic/            the ten synthetic question files the pipeline samples from (committed)
|-- data/raw/                  downloaded datasets (not committed; fetched by scripts)
|-- dashboard/                 the live dashboard: FastAPI backend and prebuilt React frontend
|-- tests/                     16 automated tests
|-- results/logs/              every run's records (not committed; created by the pipeline)
|-- results/analysis/          saved outputs of the research scripts (committed)
```

Each folder has its own README that explains its files, options and outputs.

## 4. What you need

| Item | Details |
|---|---|
| Computer | Windows, macOS or Linux. 8 GB of memory is the least that works (the author's machine has 7.8 GB); 16 GB is comfortable. About 15 GB of free disk. A graphics card is not required, but helps: on the author's machine with a 6 GB card an answer takes about 3 to 8 seconds |
| Python | 3.11 or newer. Tested on 3.12 and 3.14 |
| Ollama | v0.12.11 or newer (older versions do not return the token probabilities that confidence is computed from). Free, from ollama.com |
| Git | To download the project |
| Anthropic API account | Pay-as-you-go, from console.anthropic.com. Needed for escalations and for the Claude quality marking |
| Node.js 18 or newer | **Optional.** Only to change the dashboard's appearance. The built dashboard is included |
| Google account | **Optional.** Only to train your own models on Google Colab |
| Hugging Face token | **Optional.** Avoids download rate limits for the datasets and the embedding model |

**Cost.** Running the local models is free. In the project's own logs a Claude answer to an escalated question
cost about $0.0014, and each quality mark about $0.0012. A 100-question run with every default on therefore costs
roughly **$0.15 to $0.25**, almost all of it quality marking. `--noquality` brings that to a few cents. These are
estimates from the project's logs, and prices change: check docs.claude.com before relying on them.

## 5. Installation, step by step

Commands are shown for **Windows PowerShell** first and for **macOS / Linux** after it. Run them from the
project folder unless a step says otherwise. The [user manual](User-Manual/Two-Tier-System-User-Manual.md) explains each step in
more detail and assumes no programming knowledge.

### Step 1. Install Python, Git and Ollama

* Python: download from python.org/downloads. **On Windows, tick "Add python.exe to PATH" on the first screen.**
* Git: git-scm.com/downloads (macOS: running `git --version` offers to install it).
* Ollama: ollama.com/download.

Check each one (open a new terminal first so it sees the new programs):

```powershell
python --version        # 3.11 or newer.  On macOS/Linux use python3 if python is not found
git --version
ollama --version        # 0.12.11 or newer
```

### Step 2. Download the project

```powershell
git clone https://github.com/bipul-ranjan/two-tier-system.git
cd two-tier-system
```

### Step 3. Create the Python environment and install the packages

```powershell
# Windows PowerShell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned     # one-time permission for this window
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt                    # several GB: includes PyTorch for the cache and the router
pip install -r dashboard/backend/requirements.txt  # the dashboard's web server
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r dashboard/backend/requirements.txt
```

The prompt now starts with `(venv)`. **Repeat the activation line in every new terminal** before working on the project.
If you later see `No module named ...`, the venv is almost always the reason.

### Step 4. Give the project your Claude key

Create a key at console.anthropic.com (Settings, Billing, then API Keys). **The key is shown once; copy it at
once. Never paste it into a file in the project, a chat, or an email.**

```powershell
# Windows PowerShell. Takes effect in NEW windows only: close this one, open a new one, activate the venv again
setx ANTHROPIC_API_KEY "paste-your-key-here"
echo $env:ANTHROPIC_API_KEY          # in the new window: should print the key
```

```bash
# macOS / Linux (zsh; use ~/.bashrc for bash)
echo 'export ANTHROPIC_API_KEY="paste-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Step 5. Put the two local models in Ollama

`src/config.py` names the models each business unit uses. As shipped it names the **fine-tuned version 2** models
`payment-assistant-v2` and `retail-bank-assistant-v2`. Those are not downloadable: they are built from model files that
the training scripts produce. Choose one path:

**Path A: try the system with the base models (about 30 minutes, no training).**

```powershell
ollama pull phi3:mini
ollama pull qwen2.5:1.5b
```

Then open `src/config.py` and change the two `"model"` values to `"phi3:mini"` and `"qwen2.5:1.5b"` (and, if you like, the
two `"assistant_name"` values). Answers and confidence will differ from the fine-tuned models, so the learned
router should be retrained (section 7).

**Path B: use the fine-tuned version 2 models.** Train them on Google Colab (section 9, about 3.7 and 5.5 hours of
waiting) or obtain the two `.gguf` files from the project owner, then in the folder holding each file:

```powershell
ollama create payment-assistant-v2 -f Modelfile        # Modelfile contains one line: FROM ./payment_assistant.gguf
ollama create retail-bank-assistant-v2 -f Modelfile    # Modelfile contains: FROM ./retail_bank_assistant.gguf
ollama list                                            # both names should now appear
```

### Step 6. Check the learned router's model file

The router's trained model is included in `models/router/`. It was trained on the author's machine under one
scikit-learn version, and a saved model only loads under that same version. See which one:

```powershell
python -c "import json; print(json.load(open('models/router/text_probe_router_info.json'))['sklearn_version'])"
python -c "import sklearn; print(sklearn.__version__)"
```

If the two differ, either install the matching version (`pip install "scikit-learn==<that version>"`) or retrain the router
(section 7). The pipeline checks this before anything else and tells you which to do.

### Step 7. Check everything works

```powershell
python -m pytest tests/ -q          # 16 passed. No Ollama or Claude key needed
python -m src.config                # prints the models each unit will use
ollama list                         # both models for Step 5 are listed
python -m src.tier1                 # one real local answer, with its confidence
python -m src.tier2_escalate        # one real Claude answer (costs a fraction of a cent)
```

### Step 8. First run, then the dashboard

```powershell
python -m src.pipeline 10           # 10 random synthetic questions through the whole system
python -m uvicorn dashboard.backend.main:app --port 8000     # leave this window open
```

Open **http://localhost:8000**. After the run you should see files in `results/logs/` (including
`results_history.csv`) and the run on the dashboard. From here, see section 6.

## 6. Running the system: every command and option

Run all commands from the project folder with the venv active, Ollama running, and (for Claude) the key set.

### The pipeline

```
python -m src.pipeline [N] [--threshold-router] [--nocache] [--noquality] [--score-quality-local MODEL]
```

| Part | Meaning | Default |
|---|---|---|
| `N` | How many random synthetic questions to run | 100 |
| `--threshold-router` | Route with the older confidence pass marks (0.71 payments, 0.60 retail bank) instead of the learned router. Use it to compare the two on the same questions | learned router |
| `--nocache` | Do not use the semantic cache, so every escalation calls Claude | cache on |
| `--noquality` | Skip the Claude quality marking after the run. Quality columns stay blank (fill them later, section 8) | marking on |
| `--score-quality-local MODEL` | Also mark answers with the local Ollama model `MODEL`: a cheap first pass written to separate `quality_local_*` columns, not the numbers to report | off |

Unknown options are rejected with the list of valid ones. Examples:

```powershell
python -m src.pipeline                              # 100 questions, everything on
python -m src.pipeline 50 --noquality               # cheap: no marking
python -m src.pipeline 200 --nocache                # measure without the cache
python -m src.pipeline 100 --threshold-router       # the old router, for comparison
python scripts/run_and_score.py 100                 # the pipeline, then both quality backfills as a safety net
```

What a run does: loads both local models (and unloads them at the end), runs each question through Tier 1, the
router, the cache and Claude, then marks quality, then saves. Questions are drawn at random from
`data/synthetic/`; to draw the same ones every time, set `SEED` to a number near the top of `src/pipeline.py`.
The history file `results/logs/results_history.csv` is **locked for the whole run**, so do not open it in Excel
while a run is in progress. If a run fails, one bad question is skipped with a warning and the rest are still saved.

### The dashboard

```powershell
python -m uvicorn dashboard.backend.main:app --port 8000     # then open http://localhost:8000
python -m uvicorn dashboard.backend.main:app --port 8010     # if port 8000 is already in use
```

Set the environment variable `TWO_TIER_LOGS_DIR` to read logs from another folder. The page refreshes every five
seconds, so you can leave it open during a run. To change its appearance (needs Node 18 or newer):
`cd dashboard/frontend`, `npm install`, `npm run dev` (live preview on port 5173), `npm run build` (rebuilds the copy the backend serves).

### Summary tables

```powershell
python -m src.evaluate        # writes results/tables/summary_metrics.csv and business_unit_metrics.csv for the LATEST run
```

### Small checks, with no pipeline run

| Command | What it shows |
|---|---|
| `python -m src.config` | The Ollama address and the model each business unit uses |
| `python -m src.router` | The old threshold rule on sample scores (no dependencies) |
| `python -m src.tier1` | One real local answer and its confidence (needs Ollama and the models) |
| `python -m src.tier2_escalate` | One real Claude answer (needs the key; costs a fraction of a cent) |
| `python -m src.quality` | A demonstration of the ground-truth scoring helpers (no API call) |
| `python -m pytest tests/ -q` | The 16 automated tests |

### Ollama commands you will use

```powershell
ollama list                       # models Ollama has
ollama ps                         # models currently loaded in memory (the pipeline keeps both loaded during a run)
ollama run payment-assistant-v2 "Why was my card payment declined?"     # talk to a model directly
```

## 7. The learned router: training and retraining

The router is a small text model that learns, from your own history, to predict the quality mark Claude would give
a Tier 1 answer. It reads the question, the answer, Tier 1's confidence and the business unit, and escalates when the
prediction falls below a cut-point.

```
python scripts/train_text_router.py [--history PATH] [--target-share FRACTION] [--model-path PATH]
```

| Option | Meaning | Default |
|---|---|---|
| `--history PATH` | The results history to learn from | `results/logs/results_history.csv` |
| `--target-share FRACTION` | The share of questions to escalate, between 0.02 and 0.98. The cut-point is set to match | 0.30 |
| `--model-path PATH` | Where to save the model | `models/router/text_probe_router.joblib` |

It first evaluates itself honestly (five folds that each hold out whole question types), prints how it compares
with confidence, sets the cut-point from those held-out predictions, then trains on every row and saves three files in
`models/router/`: the model, `text_probe_router_info.json` (version, cut-point, scikit-learn version) and
`what_it_learned.txt` (the words and phrases it weighs most). It takes seconds.

**When to retrain:** after changing the Tier 1 models, after adding a business unit, after the history has grown a lot,
or when the pipeline says the saved model came from a different scikit-learn version.

**What it needs:** at least 200 history rows that have Tier 1 quality marks (`quality_draft_overall`), covering at
least 5 question types. A fresh clone has no history, so generate one first: `python -m src.pipeline 300 --threshold-router`
(more is better, 1,000 is comfortable), then train. If the history lacks the draft marks, run
`python scripts/backfill_quality_scores.py` and then `python scripts/backfill_quality_scores.py --draft`.

The pipeline will not start the learned router without a usable model file. It stops with a message and never
quietly falls back to another rule. `--threshold-router` is the deliberate way to run the old rule.

## 8. Quality scoring, the cache and the maintenance scripts

**Quality scoring** is on by default. After a run, Claude (Sonnet 5) marks each row's final answer, and each escalated
row's discarded Tier 1 draft, from 1 to 5 on correctness, completeness, tone, safety and clarity. Local rows copy their
final marks to the draft columns, so no quality cell is left blank.

**The semantic cache** is on by default. For a question the router escalated, it looks for an earlier Claude answer to a
near-identical question (similarity at least 0.95, no negation mismatch, same category and intent). The first use
downloads a small embedding model (about 90 MB).

| Script | What it does | Options |
|---|---|---|
| `scripts/backfill_quality_scores.py` | Marks rows that are missing quality marks. Safe to re-run: marked rows are not judged again | `--unit NAME` only one business unit; `--run RUN_ID` only one run; `--limit N` at most N rows this pass; `--judge-local MODEL` use a local model and write `quality_local_*` columns; `--draft` mark Tier 1's discarded draft (escalated rows are judged, local rows copy their marks for free) |
| `scripts/run_and_score.py N [flags]` | The pipeline followed by both backfills. Flags after `N` go to the pipeline | any pipeline flag |
| `scripts/backfill_cache_decision.py` | One-off: relabels old cache hits (`ESCALATE` with `cache_hit` true) as `CACHE` | none |
| `scripts/backfill_prompt_version.py` | One-off: tags old escalated rows with which Claude prompt wording was in force | none (edit `CUTOFF_RUN_ID` to change the boundary) |
| `scripts/backfill_run_id.py [RUN_ID]` | One-off: adds a `run_id` column to logs from before run ids existed | the id to use, default `run-default` |
| `scripts/recover_missing_run.py` | Adds a finished run that did not reach the history file (it was open in Excel) | none |

## 9. Training your own models on Google Colab

This is optional and needs a GPU, so it runs on Google Colab, not on your computer. In short: clone the repository into
Colab (keeping the code on Google Drive so a disconnect cannot wipe it), install `training/requirements.txt`, fetch
the data (`scripts/fetch_bitext_banking.py`, and `scripts/generate_exception_training_data.py` for the retail unit),
then run `training/train_payments.py` and, in a fresh runtime, `training/train_retail.py`. Models are saved to Google Drive
as `.gguf` files with checkpoints every 250 steps, so a dropped session resumes. On a Colab L4, version 2 took about
3 hours 44 minutes (payments) and 5.5 hours (retail bank). The complete walk-through, including the Ollama import, is in
[training/README.md](training/README.md) and in sections 14 and 15 of the user manual.

## 10. Research scripts

These produced the evidence for choosing the router. You do not need them to run the system. Both need
the history with Tier 1 quality marks and a few extra GB for PyTorch models.

| Script | Purpose | Options |
|---|---|---|
| `scripts/probe_reward_model.py` | Tests whether a pretrained reward model predicts Claude's marks without any training | `--n N` rows (500), `--model ID`, `--history PATH`, `--out-dir DIR`, `--bf16` half the memory. Scores are saved as it goes: Ctrl+C is safe |
| `scripts/train_router.py` | Compares MiniLM-based routers with confidence and the text probe, on held-out question types | `--mode frozen\|finetune` (frozen), `--base-model`, `--history`, `--out-dir`, `--folds` (5), `--stop-after N` new folds, `--report-only`, `--epochs` (3), `--max-length` (256), `--batch-size` (16), `--lr` (5e-5), `--seed` (0), `--n N` quick smoke test |

The finished results are in `results/analysis/` and in the project documents. Fine-tuning MiniLM on a processor
took about an hour per fold; frozen mode takes about 7 minutes in all.

## 11. Tests

```powershell
python -m pytest tests/ -q          # 16 tests, about two seconds
```

Twelve tests cover the learned router (predictions, the exact cut-point rule, saving and loading, and each
failure message) and four cover the older threshold rule. None needs Ollama, a Claude key or any data.
See [tests/README.md](tests/README.md).

## 12. Documentation map

| Where | What it is |
|---|---|
| This file | Overview, installation and every command |
| [User-Manual/Two-Tier-System-User-Manual.md](User-Manual/Two-Tier-System-User-Manual.md) (and `.pdf`) | The complete beginner's manual, with no prior knowledge assumed |
| [src/README.md](src/README.md) | The pipeline code, module by module, and the quality rubric |
| [scripts/README.md](scripts/README.md) | Every script, its options and when to run it |
| [training/README.md](training/README.md) | Fine-tuning on Colab and importing the models into Ollama |
| [dashboard/README.md](dashboard/README.md) | The dashboard: what each view shows, how to run and change it |
| [data/README.md](data/README.md), [models/README.md](models/README.md), [results/README.md](results/README.md), [prompts/README.md](prompts/README.md), [tests/README.md](tests/README.md) | Each folder's contents |
| Every `.py` file | A header comment that says what the file is for, how to run it, and what it needs and writes |

The research record is kept in two separate project documents outside this repository: "How We Chose the Router" (the
router, quality scoring and the cache, step by step) and "D-CASHCADE Project Record: Every Decision, Change and Test"
(the whole project, from the first idea to the final system).

## 13. Troubleshooting

| You see | Cause and fix |
|---|---|
| `No module named ...` | The venv is not active. Run the activation line from Step 3 |
| `'ollama' is not recognized` | Ollama is not installed, or the terminal was opened before installing it. Open a new terminal |
| The run stops with "The learned router has not been trained yet" | `models/router/text_probe_router.joblib` is missing. Run `python scripts/train_text_router.py` (needs history, see section 7), or run once with `--threshold-router` |
| "was trained with scikit-learn X, but Y is installed" | Install the matching version, or retrain: `python scripts/train_text_router.py` |
| `model "payment-assistant-v2" not found` | The fine-tuned model is not in Ollama. Create it (Step 5, Path B) or switch `src/config.py` to the base models (Path A) |
| Escalations fail with an authorization error | `ANTHROPIC_API_KEY` is not set in this window. Windows: close the terminal, open a new one, activate the venv |
| `PermissionError` on `results_history.csv`, or "locked" | Another program (Excel, an editor, a second run) has it open. Close it. If a run finished but is missing, run `python scripts/recover_missing_run.py` |
| Warning: models "not resident" | Both models do not fit in memory together, so Ollama swaps them and every answer is slow. Close other programs, or use smaller models |
| "semantic cache unavailable" | `sentence-transformers` is missing or its model could not download. The run continues without the cache. Check the internet connection, or use `--nocache` |
| Dashboard: "Cannot reach the backend" | Start it with the `uvicorn` command in section 6 and keep that window open |
| Dashboard: `Address already in use` | Another program uses port 8000. Add `--port 8010` and open that port |
| Colab training stops partway | Run the same command again: it resumes from the latest checkpoint on Drive |
| Quality columns are blank | The run used `--noquality`, or scoring was interrupted. Run `python scripts/backfill_quality_scores.py`, then again with `--draft` |

More, with step-by-step fixes, is in section 19 of the user manual.

## 14. Limits and responsible use

* **Synthetic data only.** Every test question was generated from templates, and the bank policies in the answers
  are generic and invented. Real customers phrase things differently, so every rate here may change on real traffic.
* **The quality marks are Claude's.** They have not been checked against human markers. Every quality figure depends on
  that assumption, and a hand-marked sample would be the way to test it.
* **Not a production system.** There is no authentication, no audit trail beyond the logs, no real-time monitoring, and no
  attack testing (prompt injection and data leakage have not been tested). Do not point it at real customer data.
* **Speed.** On the author's machine a local answer takes about as long as a Claude call, so escalating adds time. The
  benefits shown are cost and keeping most answers on the computer, not speed.
* **Small samples.** Several tests used 10 to 500 questions; read the intervals in the project documents, not just the averages.
* **The router is tied to the models it was trained on.** Retrain it after changing a Tier 1 model.

## 15. Data, models and licences

| Item | Source and licence |
|---|---|
| Bitext retail-banking chatbot dataset (training data) | Hugging Face, CDLA-Sharing 1.0: free to use with attribution, and shared derivatives must keep the licence. Cite: Bitext Innovations, "Bitext-retail-banking-llm-chatbot-training-dataset", 2024 |
| Banking77 and FinQA (earliest experiments only) | Public research datasets; see their repositories for terms |
| Phi-3-mini, Qwen2.5-1.5B, all-MiniLM-L6-v2 | Open models. Check each model card for its licence before redistributing anything derived from it |
| Claude Haiku 4.5 and Claude Sonnet 5 | Anthropic API, under Anthropic's terms of use. Used through your own key |
| Synthetic questions and the code in this repository | Written for this project |

No repository-wide licence file has been added yet. Until one is, treat the code as all rights reserved by its author and ask
before reusing it.

---

*Liverpool John Moores University, MS research project. Repository owner: bipul-ranjan.*
