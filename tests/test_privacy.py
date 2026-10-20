"""
Tests for the personal-data filter (src/privacy.py) and for its use in the pipeline.

The first group checks each detector on its own. The second checks the rule for answers (customer identifiers
only, plus values the customer typed). The third runs src.pipeline.process_query with Tier 1, Tier 2 and the
cache replaced by stand-ins, to prove that what is sent to Claude and what is written to the log is the
redacted copy, and that --no-redact really turns the filter off. No model, key or network is needed.
"""
import os
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src import pipeline
from src.privacy import ANSWER_TYPES, FILTER_VERSION, iban_ok, luhn_ok, merge_findings, redact


# ---------- the checks that stop false alarms ----------

def test_luhn_accepts_a_real_test_card_and_rejects_a_near_miss():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("12345")


def test_iban_check_accepts_a_valid_iban_and_rejects_a_made_up_one():
    assert iban_ok("GB82 WEST 1234 5698 7654 32")
    assert not iban_ok("GB00 AAAA 1111 2222 3333 44")


# ---------- each identifier type ----------

def test_card_number_with_spaces_or_dashes_is_replaced():
    for text in ("card 4111 1111 1111 1111 was used", "card 4111-1111-1111-1111 was used"):
        r = redact(text)
        assert "4111" not in r.text and "[CARD_NUMBER]" in r.text and r.findings == {"CARD_NUMBER": 1}


def test_sixteen_digits_that_fail_the_card_check_are_treated_as_an_account_number():
    r = redact("ref 1234567890123456 please")
    assert "[CARD_NUMBER]" not in r.text and "[ACCOUNT_NUMBER]" in r.text


def test_email_phone_sort_code_and_national_ids():
    r = redact("mail a.b@bank.co.uk or +44 7700 900123, sort 20-45-45, PAN ABCDE1234F, SSN 123-45-6789, NI AB 12 34 56 C")
    for leaked in ("a.b@bank", "7700", "20-45-45", "ABCDE1234F", "123-45-6789", "AB 12 34 56 C"):
        assert leaked not in r.text
    assert {"EMAIL", "PHONE", "SORT_CODE", "NATIONAL_ID"} <= set(r.findings)


def test_card_last4_keeps_the_cue_words_and_drops_the_digits():
    assert redact("my card ending 9982 was stolen").text == "my card ending [CARD_LAST4] was stolen"
    assert redact("last four digits are 1234").text == "last four digits are [CARD_LAST4]"


@pytest.mark.parametrize("text,kept", [
    ("my international transfer from my card 6775 is on hold", "card [CARD_LAST4]"),
    ("the card number 0361 was never activated", "card number [CARD_LAST4]"),
    ("card no. 1234 was declined", "card no. [CARD_LAST4]"),
    ("my card #4321 is blocked", "card #[CARD_LAST4]"),
    ("my Mastercard 5555 was charged twice", "Mastercard [CARD_LAST4]"),
    ("what is wrong with card ending 123456", "ending [CARD_LAST4]"),
    ("my card number is 7788 and it is lost", "card number is [CARD_LAST4]"),
])
def test_card_digits_are_found_whether_or_not_the_word_ending_is_used(text, kept):
    r = redact(text)
    assert kept in r.text and r.findings == {"CARD_LAST4": 1}
    assert redact(r.text).text == r.text  # and filtering again changes nothing


@pytest.mark.parametrize("clean", ["My card expires in 2027.", "I renewed my card in 2026.", "Card payments of 2500 rupees went through.",
                                   "I have 3 cards with you.", "The 2024 card offer ended.", "my card was declined at 10 shops"])
def test_years_counts_and_amounts_near_the_word_card_are_left_alone(clean):
    assert redact(clean).text == clean


def test_the_known_limits_of_the_card_digit_rule_are_pinned_down():
    """Documents two limits. If the rule changes, change this test and the notes in src/privacy.py together."""
    assert redact("my Visa 4421 was charged").text == "my Visa 4421 was charged"          # Visa is also a travel document
    assert redact("renew card 2026 please").text == "renew card [CARD_LAST4] please"      # a year straight after "card"


def test_account_number_long_and_after_the_word_account():
    assert "[ACCOUNT_NUMBER]" in redact("account 12345678 please").text
    assert redact("account number is 123456").text == "account number is [ACCOUNT_NUMBER]"


def test_date_of_birth_only_after_a_cue_so_plain_dates_survive():
    assert redact("born on 14/03/1985").text == "born on [DATE_OF_BIRTH]"
    assert redact("I paid on 14/03/2024").text == "I paid on 14/03/2024"


def test_name_only_after_a_cue_or_a_title():
    assert redact("My name is Rahul Sharma and I need help").text == "My name is [NAME] and I need help"
    assert "[NAME]" in redact("Dr Mehta asked").text


def test_postcode_uk_format():
    assert redact("I live at SW1A 1AA").text == "I live at [POSTCODE]"


# ---------- what must NOT change ----------

def test_amounts_dates_references_and_rates_are_left_alone():
    clean = "I was charged $1,300 and 50000 rupees on 12/05/2024, reference 4829, at 5.75% for 24 months."
    r = redact(clean)
    assert r.text == clean and r.findings == {}


def test_known_gap_free_text_names_and_street_addresses_are_not_caught():
    """This test documents a limit. If a future filter catches these, change the test and the docs together."""
    assert redact("I'm Rahul and I live at 12 Baker Street").text == "I'm Rahul and I live at 12 Baker Street"


def test_filtering_twice_changes_nothing_more():
    once = redact("card 4111 1111 1111 1111, mail a@b.com, ending 1234").text
    assert redact(once).text == once


def test_non_text_input_does_not_crash():
    for bad in (None, float("nan"), 12):
        assert redact(bad).findings == {}


# ---------- answers are treated differently from questions ----------

def test_answer_keeps_public_contacts_but_hides_customer_identifiers_and_echoed_values():
    q = redact("card ending 9982 misused, email rahul.k@example.com")
    a = redact("Write to rahul.k@example.com about card ending 9982. Ombudsman, London E14 9SR, call 0800 023 4567, "
               "or email support@bank.com.", only=ANSWER_TYPES, known=q.originals)
    assert "rahul.k@example.com" not in a.text and "9982" not in a.text
    assert "E14 9SR" in a.text and "0800 023 4567" in a.text and "support@bank.com" in a.text


def test_findings_can_be_merged_and_listed():
    r1, r2 = redact("mail a@b.com"), redact("card ending 1234")
    assert merge_findings(r1, r2) == {"EMAIL": 1, "CARD_LAST4": 1}
    assert redact("mail a@b.com and ending 1234").types == "CARD_LAST4,EMAIL"


# ---------- the pipeline sends and stores only the redacted copy ----------

def _stub_pipeline(monkeypatch, sent):
    """Replace Tier 1, Tier 2 and the router's inputs with stand-ins that record what they are given."""
    def fake_tier1(query, **kw):
        sent["tier1"] = query
        return {"business_unit": "payments", "assistant_name": "Payment Assistant", "model": "stub",
                "answer": "I will block the card ending 9982 straight away.", "confidence_score": 0.1,
                "confidence_score_avg": 0.1, "confidence_score_min": 0.05, "ollama_load_ms": 0, "ollama_output_tokens": 9}

    def fake_tier2(query, business_unit=None, context=""):
        sent["tier2"] = query
        return {"answer": "Thanks. We have blocked card ending 9982 and will email rahul.k@example.com.",
                "input_tokens": 10, "output_tokens": 10, "confidence": 0.9, "prompt_version": "stub"}

    monkeypatch.setattr(pipeline, "ask_tier1", fake_tier1)
    monkeypatch.setattr(pipeline, "ask_tier2", fake_tier2)


QUESTION = "Block my card ending 9982 now, my email is rahul.k@example.com and phone +44 7700 900123."


def test_pipeline_sends_claude_the_redacted_question_and_logs_only_redacted_text(monkeypatch):
    sent = {}
    _stub_pipeline(monkeypatch, sent)
    monkeypatch.setattr(pipeline, "REDACT_PII", True)
    row = pipeline.process_query(QUESTION, "payments", "CARD", "block_card", False, "run-test")
    assert sent["tier1"] == QUESTION                                   # Tier 1 is on the organisation's servers
    for leaked in ("9982", "rahul.k@example.com", "7700"):
        assert leaked not in sent["tier2"]                             # nothing identifying went to Claude
        assert leaked not in row["query"] and leaked not in row["tier1_answer"] and leaked not in row["final_answer"]
    assert row["decision"] == "ESCALATE" and row["pii_filter"] == FILTER_VERSION
    assert row["pii_count"] >= 3 and "EMAIL" in row["pii_types"] and "PHONE" in row["pii_types"]


def test_no_redact_switch_really_turns_the_filter_off(monkeypatch):
    sent = {}
    _stub_pipeline(monkeypatch, sent)
    monkeypatch.setattr(pipeline, "REDACT_PII", False)
    row = pipeline.process_query(QUESTION, "payments", "CARD", "block_card", False, "run-test")
    assert sent["tier2"] == QUESTION and row["query"] == QUESTION
    assert row["pii_filter"] == "off" and row["pii_count"] == 0


def test_cache_lookup_uses_the_redacted_question(monkeypatch):
    sent = {}
    _stub_pipeline(monkeypatch, sent)
    monkeypatch.setattr(pipeline, "REDACT_PII", True)

    def fake_cache(index, query, category=None, intent=None):
        sent["cache"] = query
        return None

    monkeypatch.setattr(pipeline, "try_semantic_cache", fake_cache)
    pipeline.process_query(QUESTION, "payments", "CARD", "block_card", False, "run-test", cache_index=object())
    assert "9982" not in sent["cache"] and "[CARD_LAST4]" in sent["cache"]
