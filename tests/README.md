# tests/

Automated tests for the parts of the pipeline that can be tested **instantly, anywhere, with no setup**: no Ollama, no Claude key,
no downloaded data and no history file. There are 158 tests and they run in about three seconds.

## Running them

From the project root, with the venv active:

```
python -m pytest tests/ -q          # all 158: expect "158 passed"
python -m pytest tests/ -v          # one line per test
python -m pytest tests/test_router.py -q            # only the older threshold rule
python -m pytest tests/test_learned_router.py -q    # only the learned router
python -m pytest tests/test_privacy.py -q           # only the personal-data filter
python -m pytest tests/test_run_report.py -q        # only the run report
python -m pytest tests/test_question_generator.py -q # only the question generator
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

### test_privacy.py: the personal-data filter and its use in the pipeline (33 tests)

| Test | What it proves |
|---|---|
| `test_card_digits_are_found_whether_or_not_the_word_ending_is_used` (7 cases) | `card 6775`, `card number 0361`, `card no. 1234`, `card #4321`, `Mastercard 5555`, `card ending 123456` and `card number is 7788` are all replaced, and filtering again changes nothing |
| `test_years_counts_and_amounts_near_the_word_card_are_left_alone` (6 cases) | `card expires in 2027`, `card in 2026`, `card payments of 2500 rupees`, `3 cards`, `2024 card offer` and `declined at 10 shops` are untouched |
| `test_the_known_limits_of_the_card_digit_rule_are_pinned_down` | `Visa 4421` stays unchanged and `card 2026` is replaced, so the two documented limits cannot change unnoticed |
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

### test_question_generator.py: fresh questions, and the pipeline's `--questions`, `--seed`, `--fresh`, `--pool` and `--generator-model` (75 tests)

The generator model is replaced by small stand-ins that write a message containing the request type's words, so a correct question
passes the label check and a deliberately wrong one fails. No Ollama is needed.

| Group | What it proves |
|---|---|
| Building a prompt (3 tests) | Made-up details follow the category and repeat for the same seed; the prompt names the request type, tone, details and examples and forbids copying and mentioning AI; replies are cleaned (quotes, labels, first paragraph) |
| Rejecting bad replies (7 tests) | Empty, too short, too long and non-customer replies are refused with the right reason; the label answer is read correctly, including "none of these" |
| Not repeating (3 tests) | Exact copies are refused whatever the case or punctuation; near-duplicates by meaning are refused and different questions are not; questions from earlier batches are not repeated |
| Making a batch (7 tests) | The right number of distinct questions and the expected columns; the same seed repeats a batch and a different seed does not; wrong labels are rejected and the attempt limit stops the loop; the exception share and unit filter control the mix; refusals are counted; skipping the label check is recorded |
| Talking to Ollama (2 tests) | The seed and sampling options are sent; a missing model and a stopped Ollama become plain instructions |
| A batch keeps its mix (6 tests) | Every request type is drawn before anything is generated, so rejections only remove slots and never change the others; an unfillable type is reported and not replaced by an easier one; a slot is retried with the same request type until it passes; the number of tries per slot is respected; a normal batch has no unfilled slots; the mix summary compares exception shares and names missing categories |
| The label check's scope (4 tests) | Options are shown as "category: intent" with a "none of these" choice and a "single best match" instruction; the unit scope offers every intent of the unit and none of the other unit's; a question that belongs to another category is caught by the unit scope but passes the category scope; an unknown scope is refused |
| Unfinished messages (10 cases) | A reply stopping on a comma, semicolon, colon or dash is rejected as cut off; so is a reply that hit the length limit; finished messages (with or without final punctuation) are not mistaken for cut-off ones; the Ollama call marks a length stop and allows longer replies |
| What the judge got wrong (4 tests) | The judge's answer is read as an option, "none of these" or unusable; mix-ups are counted by what was meant and what the judge chose; "none of these" and unusable answers are named; every rejected candidate is kept with its request type, reason and the judge's choice |
| A separate judge (3 tests) | The model that did the label check is recorded with each question; a fresh batch uses the named judge; the judge option needs `--fresh`, and both models are checked for availability and unloaded afterwards |
| Choosing how the judge is asked (5 tests) | The generator can ask the judge by name and records the method; the category scope forces the category method and the default is number; an unknown method is refused; the pipeline's `--label-method` needs `--fresh` and must be a known method; a fresh run passes the chosen method to the generator |
| The defaults (2 tests) | The default generator is `qwen2.5:7b` and the default way of asking the judge is `examples`; with no judge named, the generator model also does the label check, and a fresh batch records both |
| A fresh batch replaces the previous one (9 tests) | The previous batch file is deleted and replaced and a first batch reports nothing replaced; the ledger holds only fingerprints, never the questions; a blocked fingerprint cannot come back; a seeded batch repeats exactly and leaves the ledger alone, while unseeded batches each get their own seed; a failed or empty generation keeps the previous batch and the ledger |
| The pipeline's question options (5 tests) | `--fresh`, `--pool`, `--generator-model`, `--seed` and `--questions` are separated from the other arguments; contradictory combinations stop with a message; `FRESH_BY_DEFAULT` can be overridden for one run; a fresh run hands the pipeline the new questions with the right source and unloads the generator; a stopped Ollama ends a fresh run before anything else |
| The pipeline's file loading (5 tests) | A file loads with units mapped and the source recorded; a folder is pooled and a seed repeats the draw; a missing file, a missing column and an unknown category stop with clear messages; the built-in pool labels its source; options are taken out of the argument list and a missing value is an error |

### test_label_judge.py: the six ways of asking the judge, and measuring them (24 tests)

A "perfect judge" stand-in reads the prompt and answers correctly in whatever form it asks for, so every method must score 100% with it; any shortfall would
be a bug in the method. An "always answers 1" judge shows that the position analysis detects a model that cannot cope with a long list.

| Test | What it proves |
|---|---|
| `test_a_perfect_judge_is_right_every_time_with_every_method` (6 cases) | Each of the six methods can in principle get every answer right |
| `test_an_example_is_shown_beside_each_option_when_one_is_given` | Examples appear beside the options, and only when given |
| `test_a_reply_naming_an_option_is_matched_exactly_or_by_its_longest_text` | A reply naming an option is matched, the longest option text wins ("activate card international usage" is not mistaken for "activate card"), and "none of these" and nonsense are told apart |
| `test_the_context_lists_options_by_unit_by_category_and_the_categories_of_a_unit` | The options for a unit, a category and the categories of a unit are listed correctly |
| `test_shuffling_changes_the_order_the_model_sees_but_not_the_answer` | Shuffling changes the order but keeps the same options, and the same seed gives the same order |
| `test_the_unit_methods_offer_the_whole_unit_and_the_category_method_only_one_category` | The unit methods show the whole business unit; the category method shows one category |
| `test_the_examples_method_never_shows_the_question_being_judged_as_an_example` | The question being judged never appears as its own example |
| `test_two_step_asks_about_the_category_first_and_then_only_that_categorys_intents` | Two model calls; the second list holds only the chosen category's intents; a failed first step stops it |
| `test_an_unknown_method_is_refused` | A misspelt method name is an error |
| `test_the_wilson_interval_matches_known_values_and_stays_in_range` | The 95% interval is correct (50 of 100 gives 40% to 60%) and stays between 0 and 100% |
| `test_even_sampling_takes_the_same_number_from_every_request_type_and_repeats_for_a_seed` | Even sampling, uneven remainders, fewer questions than types, pool sampling and repeatability |
| `test_an_evaluation_has_a_row_per_case_and_method_and_counts_the_model_calls` | One row per case and method; two-step makes two calls; a perfect judge scores 100% |
| `test_a_judge_that_always_answers_one_fails_late_in_the_list_and_the_position_table_shows_it` | The position table shows 0% for the last third of the list when the judge always answers 1 |
| `test_unreadable_replies_are_counted_separately_from_wrong_ones` | Unreadable replies are reported on their own |
| `test_the_hardest_types_are_listed_worst_first` | Only types with errors are listed, worst first, and a perfect judge gives an empty list |
| `test_the_same_seed_repeats_an_evaluation_exactly` | The same seed gives the same results |
| `test_a_type_tested_only_once_and_judged_wrong_is_still_listed_and_the_note_knows_it_is_rough` | Pins down a real bug: with one question per request type, a hidden minimum of two tests per type hid every error and the report said none were wrong |
| `test_the_most_common_mix_ups_are_counted_by_intended_type_and_what_the_judge_chose` | Mix-ups are counted by intended type and the type the judge chose |
| `test_none_of_these_and_unusable_answers_are_named_in_the_mix_ups` | "None of these" and unusable answers appear as such in the mix-ups |

## What is not covered, and why

`tier1.py`, `tier2_escalate.py`, `quality.py`, `semantic_cache.py`, `pipeline.py` and the dashboard backend call Ollama or Claude, read the
logs, or download models, so testing them needs those to be present or replaced with stand-ins. During development they were exercised
with stand-in models and real logged questions (decisions follow the cut-point, Claude is called for exactly the escalated rows, a missing
router model stops the run first, no quality cell is left blank), but those throw-away harnesses are not part of this folder. A natural
next step is to add stand-in-based tests for `pipeline.py` here. This is a limitation to state in any write-up, not to leave unsaid.
