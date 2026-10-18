"""
Tests for src/run_report.py: turning a pipeline run into a plain table, from pasted console output or from the saved log.

The console sample below is real output from a run, trimmed to five questions: a local answer, a cache hit, a row that FAILED
with a connection error (and so was never saved to the log), an answer printed over several lines, and an exception question.
No model, key or network is needed.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_report import COLUMNS, parse_console, problems, summarize, tidy_from_history, write_outputs

CONSOLE = """Start Time     : 2026-10-18 09:08:44
Business Unit  : payments (Payment Assistant v2)
Scenario       : CARD / activate_card_international_usage
Query          : Hello, my platinum credit card does not work outside the country, how do I switch on overseas payments?
Tier 1 Answer  : I understand that you're experiencing issues with using your platinum credit card abroad. I can help...
Confidence AVG : 0.7625
Confidence MIN : 0.0875
Router         : text-probe-05Oct26-1102 | predicted quality 3.53 (cut-point 3.27)
Decision       : LOCAL
Latency        : Tier1 4878 ms (model load 19 ms) | Tier2 - | Total 4882 ms
------------------------------------------------------------
Start Time     : 2026-10-18 09:08:52
Business Unit  : retail_bank (Retail Bank Assistant v2)
Scenario       : RETAIL_EXCEPTION / deceased_account_holder  [EXCEPTION]
Query          : Please help: I need to inform the bank about the death of my husband and settle his accounts.
Tier 1 Answer  : I am very sorry for your loss. Please take your time, and we will treat this seriously with minimum ...
Confidence AVG : 0.7425
Confidence MIN : 0.0367
Router         : text-probe-05Oct26-1102 | predicted quality 2.69 (cut-point 3.27)
Decision       : ESCALATE
Cache hit      : similarity 0.9912 -> decision CACHE (no Claude call)
Latency        : Tier1 3112 ms (model load 11 ms) | Tier2 - | Total 3136 ms
------------------------------------------------------------
Start Time     : 2026-10-18 09:08:55
Business Unit  : payments (Payment Assistant v2)
Scenario       : CARD / check_card_annual_fee
Query          : Hello, could you tell me the yearly charges on the card ending 8530? Please advise.
Tier 1 Answer  : I'd be happy to assist you with that. However, as a chatbot, I don't have access to specific account...
Confidence AVG : 0.7371
Confidence MIN : 0.0789
Router         : text-probe-05Oct26-1102 | predicted quality 3.22 (cut-point 3.27)
Decision       : ESCALATE
  WARNING: row failed and will be skipped (APIConnectionError: Connection error.)
Start Time     : 2026-10-18 09:14:05
Business Unit  : retail_bank (Retail Bank Assistant v2)
Scenario       : LOAN / check_mortgage_payments
Query          : hi team, what is the remaining balance on my mortgage
Tier 1 Answer  : I can help you with that! To check your mortgage balance, you have a couple of options:

1. Visit ou...
Confidence AVG : 0.6043
Confidence MIN : 0.0087
Router         : text-probe-05Oct26-1102 | predicted quality 3.52 (cut-point 3.27)
Decision       : LOCAL
Latency        : Tier1 3521 ms (model load 4 ms) | Tier2 - | Total 3523 ms
------------------------------------------------------------
Start Time     : 2026-10-18 09:09:10
Business Unit  : retail_bank (Retail Bank Assistant v2)
Scenario       : RETAIL_EXCEPTION / mortgage_arrears_support  [EXCEPTION]
Query          : i am behind on my mortgage payments and worried about losing my home. please advise.
Tier 1 Answer  : I understand how worrying you are about your mortgage payments. I want to reassure you that we are h...
Confidence AVG : 0.7627
Confidence MIN : 0.0176
Router         : text-probe-05Oct26-1102 | predicted quality 3.37 (cut-point 3.27)
Decision       : LOCAL
Latency        : Tier1 3323 ms (model load 6 ms) | Tier2 - | Total 3325 ms
------------------------------------------------------------

11 of 100 rows failed and were skipped (see warnings above). The other 89 rows completed normally and will still be saved.
Unloading models...
Decisions by scenario type:
decision      CACHE  LOCAL
"""


def table():
    return parse_console(CONSOLE)


def test_five_questions_are_found_in_order_and_the_run_summary_is_ignored():
    t = table()
    assert len(t) == 5 and list(t["n"]) == [1, 2, 3, 4, 5] and list(t.columns) == COLUMNS
    assert list(t["intent"]) == ["activate_card_international_usage", "deceased_account_holder", "check_card_annual_fee",
                                 "check_mortgage_payments", "mortgage_arrears_support"]


def test_a_local_row_has_its_confidence_prediction_cut_point_and_times():
    r = table().iloc[0]
    assert (r.business_unit, r.category, r.final_decision) == ("payments", "CARD", "LOCAL")
    assert r.tier1_confidence_avg == 0.7625 and r.predicted_quality == 3.53 and r.cut_point == 3.27
    assert r.router == "text-probe-05Oct26-1102" and r.tier1_ms == 4878 and r.total_ms == 4882 and pd.isna(r.tier2_ms)


def test_a_cache_hit_is_decision_cache_even_though_the_router_said_escalate():
    r = table().iloc[1]
    assert r.router_decision == "ESCALATE" and r.final_decision == "CACHE" and r.cache_similarity == 0.9912


def test_a_failed_row_is_kept_marked_failed_and_has_no_timing():
    r = table().iloc[2]
    assert r.final_decision == "FAILED" and r.router_decision == "ESCALATE"
    assert "APIConnectionError" in r.status and pd.isna(r.total_ms)


def test_an_answer_printed_over_several_lines_is_joined_into_one():
    r = table().iloc[3]
    assert r.tier1_answer.startswith("I can help you with that!") and "Visit ou" in r.tier1_answer and "\n" not in r.tier1_answer


def test_only_the_exception_questions_are_flagged():
    assert list(table()["is_exception"]) == [False, True, False, False, True]


def test_the_summary_counts_a_failed_row_as_sent_up_by_the_router():
    s = summarize(table())
    assert s["rows"] == 5 and s["decisions"] == {"LOCAL": 3, "CACHE": 1, "FAILED": 1}
    assert s["sent_up_by_router"] == 2 and abs(s["sent_up_share"] - 0.4) < 1e-9 and s["failed"] == 1


def test_problems_explain_failed_rows_and_name_the_command_that_fixes_blank_marks():
    msgs = problems(table(), "console")
    assert len(msgs) == 1 and "FAILED" in msgs[0] and "never saved" in msgs[0] and "python -m src.pipeline 5" in msgs[0]
    t = table().iloc[[0, 1, 3, 4]].copy()  # drop the failed row, add marks to all but one saved row
    t["quality_overall"] = [4.0, None, 3.5, 3.0]
    hist = problems(t, "history")
    assert len(hist) == 1 and "1 saved row has no quality mark" in hist[0] and "backfill_quality_scores.py" in hist[0]
    t["quality_overall"] = [4.0, 3.0, 3.5, 3.0]
    assert problems(t, "history") == []


def test_the_newest_run_is_chosen_and_an_old_style_cache_hit_counts_as_cache():
    h = pd.DataFrame({
        "run_id": ["run-a", "run-a", "run-b", "run-b", "run-b"],
        "request_start_time": ["t1", "t2", "t3", "t4", "t5"], "business_unit": ["payments"] * 5,
        "category": ["CARD"] * 5, "intent": ["x"] * 5, "is_exception": [False] * 5, "query": list("abcde"),
        "tier1_confidence_avg": [0.7] * 5, "decision": ["LOCAL", "LOCAL", "LOCAL", "ESCALATE", "ESCALATE"],
        "cache_hit": [False, False, False, True, False], "total_latency_ms": [1, 2, 3, 4, 5]})
    t, rid = tidy_from_history(h)
    assert rid == "run-b" and len(t) == 3 and list(t["final_decision"]) == ["LOCAL", "CACHE", "ESCALATE"]
    t2, rid2 = tidy_from_history(h, "run-a")
    assert rid2 == "run-a" and len(t2) == 2


def test_the_csv_is_written_with_every_row(tmp_path):
    csv_path, _ = write_outputs(table(), str(tmp_path), "run: with/odd chars")
    back = pd.read_csv(csv_path)
    assert os.path.basename(csv_path) == "run_with_odd_chars.csv" and len(back) == 5 and "FAILED" in set(back["final_decision"])
