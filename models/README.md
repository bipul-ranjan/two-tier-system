# models/

Trained models that the pipeline itself uses. (The fine-tuned Tier 1 language models are **not** here: they live inside Ollama on your
computer, built from `.gguf` files that the scripts in `training/` produce on Google Colab.)

## router/ : the learned router

| File | What it is |
|---|---|
| `text_probe_router.joblib` | The trained router: a text model that predicts the quality (1 to 5) of a Tier 1 answer from the question, the answer, Tier 1's confidence and the business unit |
| `text_probe_router_info.json` | Its label and settings (see below). Readable in any text editor |
| `what_it_learned.txt` | The words and phrases in the question and in the answer that push the predicted quality down or up the most. Useful for explaining the router: they are associations in the training data, not causes |

All three are written by `python scripts/train_text_router.py` and read by `src/learned_router.py`. They are committed to the repository,
because your history (which the router learns from) is not, so a fresh clone could otherwise not run the learned router at all.

### What the info file records

| Field | Meaning |
|---|---|
| `version` | A label made from the training date and time (for example `text-probe-05Oct26-1102`). It is logged on every row as `router`, and the dashboard shows a "new router" marker when it changes |
| `trained_rows`, `intents` | How many logged answers, and how many question types, it learned from |
| `target_share` | The share of questions it was told to escalate (0.30) |
| `cutoff` | The escalation line: an answer predicted below this (3.27) is escalated. Logged as `router_cutoff` |
| `claude_quality_assumed`, `implied_minimum_gain` | Claude's average mark (4.38) and the smallest expected improvement worth paying for (4.38 minus the cut-point) |
| `heldout_correlation_mean_of_units` | How well it predicted quality on question types it never saw (+0.477) |
| `heldout_share_escalated_by_unit` | The share it escalated in each unit on those held-out predictions (payments 14%, retail bank 45%) |
| `sklearn_version` | The scikit-learn version it was trained under |

## The scikit-learn version rule

A saved model only loads under **the scikit-learn version that trained it**. If the version differs, the pipeline stops before
anything else with a message that says so. Two fixes:

1. Install the matching version: `pip install "scikit-learn==<sklearn_version from the info file>"`.
2. Retrain (seconds): `python scripts/train_text_router.py`. This needs a results history, so on a fresh clone first run
   `python -m src.pipeline 300 --threshold-router` to create one.

Retrain also whenever you change a Tier 1 model, add a business unit, or the history has grown a lot: the router is trained on a
particular model's answers. Training overwrites these three files, so commit the new ones if you want others to use them.

To run **without** the learned router on purpose: `python -m src.pipeline 100 --threshold-router`.
