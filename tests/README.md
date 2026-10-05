# tests/

Automated tests for the parts of the pipeline that can be tested **instantly, anywhere, with no setup**: no Ollama, no Claude key,
no downloaded data and no history file. There are 16 tests and they run in about two seconds.

## Running them

From the project root, with the venv active:

```
python -m pytest tests/ -q          # all 16: expect "16 passed"
python -m pytest tests/ -v          # one line per test
python -m pytest tests/test_router.py -q            # only the older threshold rule
python -m pytest tests/test_learned_router.py -q    # only the learned router
```

`pytest` is in `requirements.txt`. Run it from the project root (not from inside this folder) so the tests can import `src`.
Run the tests after any change to `src/router.py` or `src/learned_router.py`, after upgrading scikit-learn, and before sharing a copy.

## What is covered

### test_router.py: the older confidence-threshold rule (4 tests)

| Test | What it checks |
|---|---|
| `test_high_confidence_resolves_locally` | A high confidence score gives `LOCAL` |
| `test_low_confidence_escalates` | A low confidence score gives `ESCALATE` |
| `test_exact_threshold_resolves_locally` | A score exactly at the pass mark stays `LOCAL`: the boundary is `>=` |
| `test_custom_threshold_override` | Passing a pass mark of your own changes the decision |

### test_learned_router.py: the learned router (12 tests)

These use a small synthetic data set in which answers that mention a "specialist" score high and answers that mention a "declined"
request score low, and in which confidence carries no information, so a router that works must be using the text.

| Test | What it checks |
|---|---|
| `test_predicts_higher_quality_for_good_answers` | An answer written like the high-scoring ones is predicted to score well above one written like the low-scoring ones |
| `test_single_prediction_matches_batch` | Predicting one answer at a time gives the same numbers as predicting a batch |
| `test_escalates_exactly_below_the_cutpoint` | Below the cut-point escalates; above it stays local; **exactly at it stays local** |
| `test_decisions_follow_the_prediction` | Every decision equals "predicted quality below the cut-point", and the router separates the two kinds of answer |
| `test_an_unfamiliar_business_unit_still_gets_a_prediction` | A unit the router never saw still gets a finite prediction |
| `test_empty_text_does_not_crash` | An empty question and answer still give a prediction |
| `test_top_features_name_the_words_that_move_the_prediction` | The words that push the prediction down or up are the ones from the matching template |
| `test_save_and_load_round_trip` | A saved router loads back with identical predictions, cut-point and version, and writes its info file |
| `test_loading_a_missing_model_says_how_to_train_it` | A missing model file raises an error naming `scripts/train_text_router.py` and `--threshold-router` |
| `test_a_model_from_another_scikit_learn_version_is_refused` | A model saved under another scikit-learn version is refused with a message to retrain |
| `test_a_model_without_a_cutpoint_is_refused` | A trained model that was never given a cut-point is refused |
| `test_a_corrupt_model_file_is_refused_clearly` | A file that is not a model is refused with a message to retrain |

## What is not covered, and why

`tier1.py`, `tier2_escalate.py`, `quality.py`, `semantic_cache.py`, `pipeline.py` and the dashboard backend call Ollama or Claude, read the
logs, or download models, so testing them needs those to be present or replaced with stand-ins. During development they were exercised
with stand-in models and real logged questions (decisions follow the cut-point, Claude is called for exactly the escalated rows, a missing
router model stops the run first, no quality cell is left blank), but those throw-away harnesses are not part of this folder. A natural
next step is to add stand-in-based tests for `pipeline.py` here. This is a limitation to state in any write-up, not to leave unsaid.
