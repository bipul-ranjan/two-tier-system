# tests/

Automated tests for the parts of the pipeline that can be tested **instantly, anywhere, with no setup**: no Ollama, no Claude key,
no downloaded data and no history file. There are 45 tests and they run in about three seconds.

## Running them

From the project root, with the venv active:

```
python -m pytest tests/ -q          # all 35: expect "45 passed"
python -m pytest tests/ -v          # one line per test
python -m pytest tests/test_router.py -q            # only the older threshold rule
python -m pytest tests/test_learned_router.py -q    # only the learned router
python -m pytest tests/test_privacy.py -q           # only the personal-data filter
python -m pytest tests/test_run_report.py -q        # only the run report
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

### test_privacy.py: the personal-data filter and its use in the pipeline (19 tests)

| Test | What it proves |
|---|---|
| `test_luhn_accepts_a_real_test_card_and_rejects_a_near_miss` | The card check passes a valid number and rejects one digit out |
| `test_iban_check_accepts_a_valid_iban_and_rejects_a_made_up_one` | The IBAN check passes a valid IBAN and rejects a made-up one |
| `test_card_number_with_spaces_or_dashes_is_replaced` | A card number is replaced however it is spaced |
| `test_sixteen_digits_that_fail_the_card_check_are_treated_as_an_account_number` | A 16-digit number that is not a card is still removed, as an account number |
| `test_email_phone_sort_code_and_national_ids` | Emails, phones, sort codes, PAN, US SSN and UK National Insurance numbers are removed |
| `test_card_last4_keeps_the_cue_words_and_drops_the_digits` | "card ending 9982" becomes "card ending [CARD_LAST4]" |
| `test_account_number_long_and_after_the_word_account` | Long numbers and short numbers after "account" are removed |
| `test_date_of_birth_only_after_a_cue_so_plain_dates_survive` | A birth date is removed, an ordinary date is not |
| `test_name_only_after_a_cue_or_a_title` | Names after "my name is" or a title are removed |
| `test_postcode_uk_format` | A UK postcode is removed |
| `test_amounts_dates_references_and_rates_are_left_alone` | Amounts, dates, references and rates are unchanged |
| `test_known_gap_free_text_names_and_street_addresses_are_not_caught` | Documents the known gap: free-text names and addresses are not caught |
| `test_filtering_twice_changes_nothing_more` | The filter is idempotent |
| `test_non_text_input_does_not_crash` | None, NaN and numbers are handled |
| `test_answer_keeps_public_contacts_but_hides_customer_identifiers_and_echoed_values` | An answer keeps a helpline and an ombudsman address, but loses the customer's card digits and any email the customer typed |
| `test_findings_can_be_merged_and_listed` | The per-type counts and the type list for the log are correct |
| `test_pipeline_sends_claude_the_redacted_question_and_logs_only_redacted_text` | Tier 2 and the log receive only redacted text; Tier 1 sees the original |
| `test_no_redact_switch_really_turns_the_filter_off` | `--no-redact` sends and logs the raw text |
| `test_cache_lookup_uses_the_redacted_question` | The cache is searched with the redacted question |

### test_run_report.py: the run report (10 tests)

Uses five real console blocks as the sample: a local answer, a cache hit, a row that failed with a connection error, an answer printed
over several lines, and an exception question.

| Test | What it proves |
|---|---|
| `test_five_questions_are_found_in_order_and_the_run_summary_is_ignored` | Every question is found, in order, and the text after the last one is ignored |
| `test_a_local_row_has_its_confidence_prediction_cut_point_and_times` | The fields of a local row are read correctly |
| `test_a_cache_hit_is_decision_cache_even_though_the_router_said_escalate` | A cache hit is labelled CACHE with its similarity |
| `test_a_failed_row_is_kept_marked_failed_and_has_no_timing` | A row that failed is kept, marked FAILED, with the reason |
| `test_an_answer_printed_over_several_lines_is_joined_into_one` | A long answer split over lines becomes one value |
| `test_only_the_exception_questions_are_flagged` | The exception flag is right |
| `test_the_summary_counts_a_failed_row_as_sent_up_by_the_router` | The summary counts failed rows as sent up, so the run is not understated |
| `test_problems_explain_failed_rows_and_name_the_command_that_fixes_blank_marks` | Each warning names the command that fixes it, and a clean run has none |
| `test_the_newest_run_is_chosen_and_an_old_style_cache_hit_counts_as_cache` | The newest run is the default, and old-style cache hits are read as CACHE |
| `test_the_csv_is_written_with_every_row` | The CSV holds every row, with a safe file name |

## What is not covered, and why

`tier1.py`, `tier2_escalate.py`, `quality.py`, `semantic_cache.py`, `pipeline.py` and the dashboard backend call Ollama or Claude, read the
logs, or download models, so testing them needs those to be present or replaced with stand-ins. During development they were exercised
with stand-in models and real logged questions (decisions follow the cut-point, Claude is called for exactly the escalated rows, a missing
router model stops the run first, no quality cell is left blank), but those throw-away harnesses are not part of this folder. A natural
next step is to add stand-in-based tests for `pipeline.py` here. This is a limitation to state in any write-up, not to leave unsaid.
