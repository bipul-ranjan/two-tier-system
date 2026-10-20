"""
Tests for src/question_generator.py (making fresh customer questions with a separate model) and for the pipeline's
--questions, --seed, --fresh, --pool, --generator-model, --label-model and --label-method options.

The generator model is replaced by small stand-in functions, so no Ollama, key or network is needed. The stand-ins write a
message that contains the request type's words, and the label check looks for those words, so a correct question passes and a
deliberately wrong one fails.
"""
import os
import random
import re
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import requests

from src import pipeline, question_generator as qg


def pool():
    rows = []
    for cat, intent in [("CARD", "block_card"), ("CARD", "cancel_card"), ("LOAN", "apply_for_loan"),
                        ("PAYMENT_EXCEPTION", "duplicate_charge"), ("RETAIL_EXCEPTION", "deceased_account_holder")]:
        for i in range(4):
            rows.append({"instruction": f"example {intent} {i}", "category": cat, "intent": intent})
    return pd.DataFrame(rows)


def stub_generate(prompt, seed):
    """A 'model' that writes a message containing the request type's words, so the label check can recognise it."""
    line = [l for l in prompt.splitlines() if l.startswith("Request type:")][0]
    intent_words = line.split(": ", 2)[2]
    return f"hello team, {intent_words}, can you help me with number {seed}"


def stub_classify(prompt):
    """A 'model' that picks the option whose words appear in the message."""
    message = prompt.split('Message: "')[1].split('"\n')[0]
    # options read "category: intent words", with ' (for example: "...")' after them when the examples method is used
    options = [re.sub(r' \(for example: ".*"\)$', "", l.split(". ", 1)[1]) for l in prompt.splitlines() if l[:1].isdigit()]
    for i, o in enumerate(options, start=1):
        if o.split(": ")[-1] in message:
            return str(i)
    return str(len(options))


def fake_embed(texts):
    """Hashed character-pair counts: similar strings give similar vectors."""
    out = np.zeros((len(texts), 64))
    for r, t in enumerate(texts):
        t = qg.normalise(t)
        for a, b in zip(t, t[1:]):
            out[r, hash(a + b) % 64] += 1
    return out


# ---------- building a prompt and cleaning a reply ----------

def test_details_follow_the_category_and_repeat_for_the_same_seed():
    a = qg.make_details(random.Random(5), "CARD")
    assert a == qg.make_details(random.Random(5), "CARD")
    seen = [d for s in range(200) for d in qg.make_details(random.Random(s), "LOAN")]
    assert not any("card ending" in d for d in seen) and any("amount" in d for d in seen)


def test_prompt_names_the_request_type_style_details_and_examples_and_forbids_copying():
    p = qg.build_prompt("CARD", "block_card", ["ex one", "ex two"], "very brief", ["the city Dubai"])
    assert "card: block card" in p and "very brief" in p and "the city Dubai" in p
    assert "- ex one" in p and "- ex two" in p and "do not copy" in p and "never mention that you are an AI" in p


def test_clean_message_strips_quotes_and_labels_and_keeps_the_first_paragraph():
    assert qg.clean_message('Customer: "Please block my card now."\n\nLet me know if you need more.') == ("Please block my card now.", "")


@pytest.mark.parametrize("text,reason", [("", "empty"), ("   ", "empty"), ("hi", "too_short"), ("x" * 301, "too_long"),
                                         ("Here is a message you could use for the customer", "not_a_customer_message"),
                                         ("As an AI I cannot write that kind of text for you", "not_a_customer_message")])
def test_clean_message_rejects_unusable_replies(text, reason):
    assert qg.clean_message(text) == (None, reason)


def test_label_answer_picks_the_option_or_none():
    opts = ["block_card", "cancel_card", "apply_for_loan"]
    assert qg.parse_label_answer("2", opts) == "cancel_card"
    assert qg.parse_label_answer("Answer: 1.", opts) == "block_card"
    assert qg.parse_label_answer("4", opts) is None and qg.parse_label_answer("none", opts) is None
    assert "4. none of these" in qg.build_label_prompt("msg", opts)


# ---------- not repeating questions ----------

def test_exact_copies_are_refused_even_if_case_and_punctuation_differ():
    d = qg.Deduper()
    d.add_exact(["Please block my card!"])
    assert d.check("please block my card")[:2] == (False, "duplicate")
    assert d.check("a different question entirely")[0] is True


def test_a_near_duplicate_by_meaning_is_refused_but_a_different_question_is_not():
    d = qg.Deduper(fake_embed, threshold=0.9)
    d.add_embedded(["please block my lost card right now"])
    ok, reason, sim, _ = d.check("please block my lost card right now thanks")
    assert (ok, reason) == (False, "near_duplicate") and sim >= 0.9
    assert d.check("how do I apply for a mortgage next year")[0] is True


# ---------- generating a batch ----------

def run(n=12, seed=1, **kw):
    return qg.generate_questions(pool(), n, seed, stub_generate, classify_fn=stub_classify, **kw)


def test_a_batch_has_the_requested_size_distinct_questions_and_the_expected_columns():
    df, stats = run()
    assert len(df) == 12 and stats["accepted"] == 12 and df["instruction"].nunique() == 12
    assert list(df.columns) == ["instruction", "category", "intent", "is_exception", "generator_model", "prompt_version",
                                "seed", "style", "label_scope", "label_method", "label_model", "label_check", "nearest_similarity"]
    assert set(df["label_check"]) == {"pass"} and set(df["label_scope"]) == {"unit"} and df["seed"].is_unique


def test_the_same_seed_repeats_a_batch_and_a_different_seed_does_not():
    a, _ = run(seed=3)
    b, _ = run(seed=3)
    c, _ = run(seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["instruction"]) != list(c["instruction"])


def test_a_wrong_label_is_rejected_and_the_loop_stops_at_its_attempt_limit():
    df, stats = qg.generate_questions(pool(), 5, 1, stub_generate, classify_fn=lambda p: "99", per_slot_attempts=3)
    assert df.empty and stats["attempts"] == 15 and stats["label_failed"] == 15 and stats["rejected"]["label_mismatch"] == 15


def test_refusals_are_rejected_and_counted_and_the_batch_still_fills():
    calls = {"n": 0}

    def noisy(prompt, seed):
        calls["n"] += 1
        return "As an AI I cannot do that for you" if calls["n"] % 4 == 0 else stub_generate(prompt, seed)

    df, stats = qg.generate_questions(pool(), 10, 2, noisy, classify_fn=stub_classify)
    assert len(df) == 10 and stats["rejected"]["not_a_customer_message"] >= 3 and stats["attempts"] > 10


def test_questions_from_earlier_batches_are_not_repeated():
    first, _ = run(n=1, seed=9)
    again, stats = qg.generate_questions(pool(), 1, 9, stub_generate, classify_fn=stub_classify, previous=list(first["instruction"]))
    assert again["instruction"].iloc[0] != first["instruction"].iloc[0] and stats["rejected"].get("duplicate", 0) >= 1


def test_the_exception_share_controls_the_mix():
    only_exc, _ = run(n=15, exception_share=1.0)
    none_exc, _ = run(n=15, exception_share=0.0)
    assert only_exc["is_exception"].all() and not none_exc["is_exception"].any()


def test_the_unit_filter_keeps_only_that_business_unit():
    df, _ = run(n=15, unit="payments")
    assert set(df["category"]) <= {"CARD", "ATM", "TRANSFER", "FEES", "PAYMENT_EXCEPTION"}


def test_skipping_the_label_check_marks_the_rows_skipped():
    df, _ = qg.generate_questions(pool(), 4, 1, stub_generate, classify_fn=None)
    assert set(df["label_check"]) == {"skipped"} and set(df["label_scope"]) == {"skipped"} and len(df) == 4


# ---------- talking to Ollama ----------

class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {"response": "hello there"}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_the_ollama_call_sends_the_seed_and_sampling_options(monkeypatch):
    seen = {}
    monkeypatch.setattr(qg.requests, "post", lambda url, json, timeout: seen.update(json) or FakeResponse())
    assert qg.make_ollama_fn("m:1b", temperature=0.7)("a prompt", 42) == "hello there"
    assert seen["model"] == "m:1b" and seen["options"]["seed"] == 42 and seen["options"]["temperature"] == 0.7 and seen["stream"] is False


def test_ollama_errors_become_plain_instructions(monkeypatch):
    monkeypatch.setattr(qg.requests, "post", lambda *a, **k: FakeResponse(404))
    with pytest.raises(RuntimeError, match="ollama pull m:1b"):
        qg.make_ollama_fn("m:1b")("p", 1)

    def down(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(qg.requests, "post", down)
    with pytest.raises(RuntimeError, match="Open the Ollama app"):
        qg.make_ollama_fn("m:1b")("p", 1)


# ---------- the pipeline reads a generated file ----------

def write_batch(tmp_path, name="batch.csv", rows=None):
    rows = rows or [{"instruction": f"q{i}", "category": c, "intent": "x"} for i, c in enumerate(["CARD", "LOAN", "RETAIL_EXCEPTION", "ATM"])]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_the_pipeline_loads_a_file_maps_units_and_records_the_source(tmp_path):
    s = pipeline.load_question_file(write_batch(tmp_path), n=10, seed=1)
    assert len(s) == 4 and set(s["question_source"]) == {"batch.csv"}
    by = dict(zip(s["category"], s["business_unit"]))
    assert by["CARD"] == "payments" and by["LOAN"] == "retail_bank"
    assert bool(dict(zip(s["category"], s["is_exception"]))["RETAIL_EXCEPTION"]) and not bool(dict(zip(s["category"], s["is_exception"]))["CARD"])


def test_a_folder_of_files_is_pooled_and_the_same_seed_draws_the_same_questions(tmp_path):
    write_batch(tmp_path, "a.csv")
    write_batch(tmp_path, "b.csv", [{"instruction": "z1", "category": "CARD", "intent": "x"}, {"instruction": "z2", "category": "FEES", "intent": "y"}])
    a = pipeline.load_question_file(str(tmp_path), n=3, seed=7)
    b = pipeline.load_question_file(str(tmp_path), n=3, seed=7)
    assert len(a) == 3 and list(a["instruction"]) == list(b["instruction"])


def test_a_bad_questions_file_stops_with_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_questions"):
        pipeline.load_question_file(str(tmp_path / "missing.csv"))
    with pytest.raises(ValueError, match="intent"):
        pipeline.load_question_file(write_batch(tmp_path, "bad.csv", [{"instruction": "q", "category": "CARD"}]))
    with pytest.raises(ValueError, match="Unknown category"):
        pipeline.load_question_file(write_batch(tmp_path, "odd.csv", [{"instruction": "q", "category": "WEALTH", "intent": "x"}]))


def test_the_built_in_pool_labels_its_source(tmp_path, monkeypatch):
    pd.DataFrame({"instruction": ["q1", "q2", "q3"], "category": ["CARD", "LOAN", "ATM"], "intent": ["a", "b", "c"]}).to_csv(
        tmp_path / "synthetic_bitext_01.csv", index=False)
    monkeypatch.setattr(pipeline, "SYNTHETIC_DIR", str(tmp_path))
    s = pipeline.load_random_sample(3, seed=1)
    assert set(s["question_source"]) == {"synthetic-pool"} and len(s) == 3


def test_options_are_taken_out_of_the_argument_list_and_a_missing_value_is_an_error():
    assert pipeline._pop_option(["50", "--questions", "f.csv", "--nocache"], "--questions") == ("f.csv", ["50", "--nocache"])
    assert pipeline._pop_option(["50"], "--seed") == (None, ["50"])
    with pytest.raises(SystemExit):
        pipeline._pop_option(["50", "--questions"], "--questions")
    with pytest.raises(SystemExit):
        pipeline._pop_option(["--seed", "--nocache"], "--seed")


# ---------- a fresh batch replaces the previous one ----------

def fresh(tmp_path, **kw):
    return qg.make_fresh_batch(kw.pop("n", 6), pool=pool(), generate_fn=kw.pop("generate_fn", stub_generate),
                               classify_fn=stub_classify, embed_fn=None, out_path=str(tmp_path / "current_batch.csv"),
                               ledger_path=str(tmp_path / "seen.txt"), **kw)


def test_text_hash_ignores_case_and_punctuation_and_cannot_be_reversed_by_eye():
    assert qg.text_hash("Block my card!") == qg.text_hash("block my CARD") and qg.text_hash("a") != qg.text_hash("b")
    assert len(qg.text_hash("x")) == 40 and "block" not in qg.text_hash("block my card")


def test_a_question_whose_fingerprint_is_blocked_cannot_come_back():
    first, _ = run(n=1, seed=9)
    again, stats = qg.generate_questions(pool(), 1, 9, stub_generate, classify_fn=stub_classify,
                                         blocked_hashes={qg.text_hash(first["instruction"].iloc[0])})
    assert again["instruction"].iloc[0] != first["instruction"].iloc[0] and stats["rejected"]["duplicate"] >= 1


def test_the_previous_batch_is_deleted_and_replaced_by_the_new_one(tmp_path):
    out = tmp_path / "current_batch.csv"
    out.write_text("instruction,category,intent\nOLD QUESTION,CARD,x\n")
    df, stats, seed, replaced = fresh(tmp_path)
    back = pd.read_csv(out)
    assert replaced is True and "OLD QUESTION" not in set(back["instruction"]) and len(back) == 6 == len(df)
    assert not (tmp_path / "current_batch.csv.tmp").exists()


def test_a_first_batch_reports_that_nothing_was_replaced(tmp_path):
    assert fresh(tmp_path)[3] is False


def test_the_ledger_keeps_only_fingerprints_of_the_questions_used(tmp_path):
    df, _, _, _ = fresh(tmp_path)
    text = (tmp_path / "seen.txt").read_text()
    assert set(text.split()) == {qg.text_hash(q) for q in df["instruction"]} and not any(q in text for q in df["instruction"])


def test_a_seeded_batch_repeats_exactly_and_leaves_the_ledger_alone(tmp_path):
    a = fresh(tmp_path, seed=21)[0]
    b = fresh(tmp_path, seed=21)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not (tmp_path / "seen.txt").exists()


def test_without_a_seed_each_batch_gets_its_own_seed(tmp_path):
    assert fresh(tmp_path)[2] != fresh(tmp_path)[2]


def test_a_failed_generation_keeps_the_previous_batch_and_the_ledger(tmp_path):
    out = tmp_path / "current_batch.csv"
    out.write_text("instruction,category,intent\nOLD QUESTION,CARD,x\n")

    def broken(prompt, seed):
        raise RuntimeError("Cannot reach Ollama")
    with pytest.raises(RuntimeError, match="Ollama"):
        fresh(tmp_path, generate_fn=broken)
    assert "OLD QUESTION" in out.read_text() and not (tmp_path / "seen.txt").exists()


def test_an_empty_result_keeps_the_previous_batch(tmp_path):
    out = tmp_path / "current_batch.csv"
    out.write_text("instruction,category,intent\nOLD QUESTION,CARD,x\n")
    with pytest.raises(RuntimeError, match="previous batch was kept"):
        qg.make_fresh_batch(3, pool=pool(), generate_fn=lambda p, s: "", classify_fn=stub_classify, embed_fn=None,
                            out_path=str(out), ledger_path=str(tmp_path / "seen.txt"))
    assert "OLD QUESTION" in out.read_text()


# ---------- the pipeline's question options ----------

def test_the_question_options_are_separated_from_the_other_arguments():
    o = pipeline.parse_question_options(["50", "--fresh", "--seed", "7", "--nocache", "--generator-model", "m:1b"])
    assert o["fresh"] and o["seed"] == 7 and o["seed_given"] and o["generator_model"] == "m:1b" and o["rest"] == ["50", "--nocache"]
    plain = pipeline.parse_question_options(["50"])
    assert not plain["fresh"] and plain["questions_path"] is None and not plain["seed_given"] and plain["rest"] == ["50"]


def test_contradictory_question_options_stop_with_a_message():
    for bad in (["--fresh", "--pool"], ["--fresh", "--questions", "f.csv"], ["--generator-model", "m:1b"], ["--seed", "abc"]):
        with pytest.raises(SystemExit):
            pipeline.parse_question_options(bad)


def test_fresh_by_default_can_be_switched_off_for_one_run(monkeypatch):
    monkeypatch.setattr(pipeline, "FRESH_BY_DEFAULT", True)
    assert pipeline.parse_question_options(["20"])["fresh"] is True
    assert pipeline.parse_question_options(["20", "--pool"])["fresh"] is False
    assert pipeline.parse_question_options(["20", "--questions", "f.csv"])["fresh"] is False


def test_a_fresh_run_hands_the_pipeline_the_new_questions_and_unloads_the_generator(tmp_path, monkeypatch, capsys):
    batch = tmp_path / "current_batch.csv"
    pd.DataFrame({"instruction": [f"fresh question {i} about my card" for i in range(4)], "category": ["CARD"] * 4,
                  "intent": ["block_card"] * 4}).to_csv(batch, index=False)
    got, unloaded = {}, []
    monkeypatch.setattr(qg, "CURRENT_BATCH_PATH", str(batch))
    monkeypatch.setattr(qg, "check_model_available", lambda m: None)
    monkeypatch.setattr(qg, "make_fresh_batch", lambda n, seed, model, progress=None, **kw: (
        pd.read_csv(batch), {"accepted": 4, "attempts": 5, "rejected": {"duplicate": 1}, "label_checked": 4, "label_failed": 0, "unfilled": {"FEES / check_fees": 2}}, 77, True))
    monkeypatch.setattr(pipeline, "unload_models", lambda models: unloaded.extend(models))
    monkeypatch.setattr(pipeline, "run_pipeline", lambda sample, **kw: got.update(sample=sample, kw=kw))
    pipeline.main(["4", "--fresh", "--noquality", "--nocache", "--threshold-router"])
    s = got["sample"]
    assert len(s) == 4 and set(s["question_source"]) == {f"generated:{qg.GENERATOR_MODEL}:seed77"}
    assert set(s["business_unit"]) == {"payments"} and unloaded == [qg.GENERATOR_MODEL]
    assert "Unfilled: 2 of 4 slots" in capsys.readouterr().out


def test_a_stopped_ollama_ends_a_fresh_run_before_anything_else(monkeypatch):
    def down(model):
        raise RuntimeError("Cannot reach Ollama. Open the Ollama app")
    monkeypatch.setattr(qg, "check_model_available", down)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda *a, **k: pytest.fail("the run must not start"))
    with pytest.raises(SystemExit, match="Open the Ollama app"):
        pipeline.main(["4", "--fresh"])


# ---------- the label check chooses among every intent of the business unit ----------

def pay_pool():
    rows = []
    for cat, intent in [("CARD", "block_card"), ("TRANSFER", "make_transfer"), ("LOAN", "apply_for_loan")]:
        for i in range(4):
            rows.append({"instruction": f"example {intent} {i}", "category": cat, "intent": intent})
    return pd.DataFrame(rows)


def test_options_are_shown_with_their_category_so_similar_intents_can_be_told_apart():
    p = qg.build_label_prompt("a message", [("CARD", "cancel_card"), ("TRANSFER", "cancel_transfer")])
    assert "1. card: cancel card" in p and "2. transfer: cancel transfer" in p and "3. none of these" in p
    assert "single best match" in p
    assert qg.parse_label_answer("2", [("CARD", "cancel_card"), ("TRANSFER", "cancel_transfer")]) == ("TRANSFER", "cancel_transfer")


def test_the_unit_scope_offers_every_intent_of_the_unit_and_none_of_the_other_unit():
    seen = []

    def spy(prompt):
        seen.append(prompt)
        return stub_classify(prompt)
    qg.generate_questions(pay_pool(), 6, 1, stub_generate, classify_fn=spy, unit="payments")
    text = " ".join(seen)
    assert "card: block card" in text and "transfer: make transfer" in text and "apply for loan" not in text
    qg.generate_questions(pay_pool(), 6, 1, stub_generate, classify_fn=spy)  # no unit filter: the loan question sees only retail options
    loan_prompts = [p for p in seen if 'apply for loan' in p.split("Options:")[0] or "loan: apply for loan" in p]
    assert loan_prompts and all("card: block card" not in p for p in loan_prompts)


def test_a_question_that_belongs_to_another_category_is_caught_by_the_unit_scope_but_not_the_category_scope():
    def prefers_transfer(prompt):
        options = [l.split(". ", 1)[1] for l in prompt.splitlines() if l[:1].isdigit()]
        for i, o in enumerate(options, start=1):
            if "make transfer" in o:
                return str(i)
        return "1"
    unit_df, unit_stats = qg.generate_questions(pay_pool(), 4, 3, stub_generate, classify_fn=prefers_transfer, unit="payments")
    cat_df, _ = qg.generate_questions(pay_pool(), 4, 3, stub_generate, classify_fn=prefers_transfer, unit="payments", label_scope="category")
    assert set(unit_df["category"]) == {"TRANSFER"} and unit_stats["rejected"]["label_mismatch"] > 0
    assert "CARD" in set(cat_df["category"])  # the easier check lets the mislabelled card questions through
    assert set(unit_df["label_scope"]) == {"unit"} and set(cat_df["label_scope"]) == {"category"}


def test_an_unknown_label_scope_is_refused():
    with pytest.raises(ValueError, match="label_scope"):
        qg.generate_questions(pool(), 2, 1, stub_generate, classify_fn=stub_classify, label_scope="everything")


# ---------- the batch keeps its mix: slots are drawn first and retried ----------

def mixed_pool():
    rows = []
    for cat, intent in [("CARD", "block_card"), ("TRANSFER", "make_transfer"), ("FEES", "check_fees")]:
        for i in range(4):
            rows.append({"instruction": f"example {intent} {i}", "category": cat, "intent": intent})
    return pd.DataFrame(rows)


def rejects_card(prompt):
    """A label check that never accepts a card question: it answers 'none of these' for it."""
    message = prompt.split('Message: "')[1].split('"\n')[0]
    options = [l.split(". ", 1)[1] for l in prompt.splitlines() if l[:1].isdigit()]
    if "block card" in message:
        return str(len(options))
    return stub_classify(prompt)


def test_every_slot_is_drawn_before_anything_is_generated_so_rejections_cannot_bend_the_mix():
    free, _ = qg.generate_questions(mixed_pool(), 12, 5, stub_generate, classify_fn=stub_classify, unit="payments")
    bent, stats = qg.generate_questions(mixed_pool(), 12, 5, stub_generate, classify_fn=rejects_card, unit="payments")
    assert list(bent["category"]) == [c for c in free["category"] if c != "CARD"]  # same slots, minus the ones that could not be filled
    assert len(bent) < 12 and stats["unfilled"]["CARD / block_card"] == 12 - len(bent)


def test_an_unfillable_request_type_is_reported_and_never_replaced_by_an_easier_one():
    df, stats = qg.generate_questions(mixed_pool(), 12, 5, stub_generate, classify_fn=rejects_card, unit="payments")
    assert "CARD" not in set(df["category"]) and list(stats["unfilled"]) == ["CARD / block_card"]
    assert len(df) + sum(stats["unfilled"].values()) == stats["slots"] == 12


def test_a_slot_is_retried_with_the_same_request_type_until_it_passes():
    calls, prompts = {"n": 0}, []

    def flaky(prompt):
        calls["n"] += 1
        return "99" if calls["n"] <= 2 else stub_classify(prompt)

    def spy(prompt, seed):
        prompts.append(prompt)
        return stub_generate(prompt, seed)
    df, stats = qg.generate_questions(mixed_pool(), 1, 3, spy, classify_fn=flaky, unit="payments", per_slot_attempts=5)
    assert len(df) == 1 and stats["attempts"] == 3 and stats["unfilled"] == {}
    assert len({[l for l in p.splitlines() if l.startswith("Request type:")][0] for p in prompts}) == 1


def test_the_number_of_tries_per_slot_is_respected():
    df, stats = qg.generate_questions(mixed_pool(), 1, 3, stub_generate, classify_fn=lambda p: "99", unit="payments", per_slot_attempts=4)
    assert df.empty and stats["attempts"] == 4 and sum(stats["unfilled"].values()) == 1


def test_a_normal_batch_has_no_unfilled_slots():
    df, stats = qg.generate_questions(mixed_pool(), 9, 2, stub_generate, classify_fn=stub_classify, unit="payments")
    assert stats["unfilled"] == {} and stats["slots"] == 9 and len(df) == 9


def test_the_mix_summary_compares_exception_share_and_names_missing_categories():
    df, _ = qg.generate_questions(pool(), 6, 1, stub_generate, classify_fn=None, exception_share=0.0)
    m = qg.mix_summary(df, pool())
    assert m["batch_exception_share"] == 0.0 and abs(m["pool_exception_share"] - 0.4) < 1e-9
    assert set(m["missing_categories"]) <= set(pool()["category"]) and "PAYMENT_EXCEPTION" in m["missing_categories"]


# ---------- unfinished messages are rejected ----------

@pytest.mark.parametrize("ending", [",", ";", ":", " -", "\u2014"])
def test_a_message_that_stops_on_a_comma_colon_or_dash_is_rejected_as_cut_off(ending):
    assert qg.clean_message("I need help with my card and the payment" + ending) == (None, "cut_off")


def test_a_reply_that_hit_the_length_limit_is_rejected_whatever_it_ends_with():
    assert qg.clean_message(qg.CutOff("I was charged twice for the same order and I would like")) == (None, "cut_off")


@pytest.mark.parametrize("fine", ["Can you help me block my card?", "i lost my card please help asap", "My card was stolen yesterday."])
def test_finished_messages_are_not_mistaken_for_cut_off_ones(fine):
    assert qg.clean_message(fine) == (fine, "")


def test_the_ollama_call_marks_a_length_stop_as_cut_off_and_allows_longer_replies(monkeypatch):
    seen = {}
    replies = iter([FakeResponse(200, {"response": "half a sentence and", "done_reason": "length"}),
                    FakeResponse(200, {"response": "a whole one.", "done_reason": "stop"})])
    monkeypatch.setattr(qg.requests, "post", lambda url, json, timeout: seen.update(json) or next(replies))
    fn = qg.make_ollama_fn("m:1b")
    first, second = fn("p", 1), fn("p", 2)
    assert isinstance(first, qg.CutOff) and not isinstance(second, qg.CutOff) and second == "a whole one."
    assert seen["options"]["num_predict"] == 160


# ---------- what the judge got wrong ----------

def test_the_judge_s_answer_is_read_as_an_option_none_of_these_or_unusable():
    opts = [("CARD", "block_card"), ("TRANSFER", "make_transfer")]
    assert qg.parse_label_choice("2", opts) == ("option", ("TRANSFER", "make_transfer"))
    assert qg.parse_label_choice("3", opts) == ("none", None)
    assert qg.parse_label_choice("I think it is a card", opts) == ("unparsed", None)
    assert qg.parse_label_choice("17", opts) == ("unparsed", None)


def choose_transfer(prompt):
    options = [l.split(". ", 1)[1] for l in prompt.splitlines() if l[:1].isdigit()]
    return str(next(i for i, o in enumerate(options, start=1) if "make transfer" in o))


def test_mix_ups_are_counted_by_what_was_meant_and_what_the_judge_chose():
    df, stats = qg.generate_questions(pay_pool(), 10, 3, stub_generate, classify_fn=choose_transfer, unit="payments", per_slot_attempts=2)
    assert stats["confusions"]["CARD / block_card -> TRANSFER / make_transfer"] >= 3
    assert set(df["intent"]) == {"make_transfer"} and stats["by_intent"]["TRANSFER / make_transfer"]["passed"] == len(df)
    lines = qg.confusion_lines(stats)
    assert lines[0].startswith("CARD / block_card was judged as TRANSFER / make_transfer x")
    assert any(l.startswith("CARD / block_card: 0 of ") and l.endswith("(0%)") for l in qg.hardest_lines(stats))


def test_none_of_these_and_unusable_answers_are_named_in_the_report():
    def none_of_these(prompt):  # the last numbered line is always "none of these"
        return str(len([l for l in prompt.splitlines() if l[:1].isdigit()]))
    _, none_stats = qg.generate_questions(pay_pool(), 4, 1, stub_generate, classify_fn=none_of_these, unit="payments", per_slot_attempts=2)
    assert any(k.endswith("-> none of these") for k in none_stats["confusions"])
    _, junk_stats = qg.generate_questions(pay_pool(), 4, 1, stub_generate, classify_fn=lambda p: "no idea", unit="payments", per_slot_attempts=2)
    assert any(k.endswith("-> no usable answer") for k in junk_stats["confusions"])


def test_every_rejected_candidate_is_kept_with_its_request_type_reason_and_the_judge_s_choice():
    calls = {"n": 0}

    def sometimes_cut(prompt, seed):
        calls["n"] += 1
        text = stub_generate(prompt, seed)
        return qg.CutOff(text) if calls["n"] % 3 == 0 else text

    _, stats = qg.generate_questions(pay_pool(), 6, 2, sometimes_cut, classify_fn=choose_transfer, unit="payments", per_slot_attempts=3)
    frame = qg.rejected_to_frame(stats)
    assert list(frame.columns) == ["instruction", "category", "intent", "reason", "judged_as"]
    assert {"cut_off", "label_mismatch"} <= set(frame["reason"]) and len(frame) == sum(stats["rejected"].values())
    assert set(frame[frame["reason"] == "label_mismatch"]["judged_as"]) == {"TRANSFER / make_transfer"}
    assert frame[frame["reason"] == "label_mismatch"]["instruction"].str.len().gt(0).all()


# ---------- a separate judge ----------

def test_the_model_that_did_the_label_check_is_recorded_with_each_question():
    default, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=stub_classify, model="gen:3b")
    named, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=stub_classify, model="gen:3b", label_model="judge:7b")
    skipped, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=None, model="gen:3b")
    assert set(default["label_model"]) == {"gen:3b"} and set(named["label_model"]) == {"judge:7b"} and set(skipped["label_model"]) == {"skipped"}


def test_a_fresh_batch_uses_the_named_judge_for_the_label_check(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(qg, "make_classify_fn", lambda model, **kw: used.append(model) or stub_classify)
    df, _, _, _ = qg.make_fresh_batch(4, pool=pool(), generate_fn=stub_generate, embed_fn=None, label_model="judge:7b",
                                      out_path=str(tmp_path / "b.csv"), ledger_path=str(tmp_path / "l.txt"))
    assert used == ["judge:7b"] and set(df["label_model"]) == {"judge:7b"}


def test_the_judge_option_needs_fresh_and_both_models_are_checked_and_unloaded(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="label-model only applies"):
        pipeline.parse_question_options(["10", "--label-model", "judge:7b"])
    assert pipeline.parse_question_options(["10", "--fresh", "--label-model", "judge:7b"])["label_model"] == "judge:7b"
    batch = tmp_path / "current_batch.csv"
    pd.DataFrame({"instruction": ["fresh question one about my card", "fresh question two about my card"], "category": ["CARD"] * 2,
                  "intent": ["block_card"] * 2}).to_csv(batch, index=False)
    checked, unloaded, got = [], [], {}
    monkeypatch.setattr(qg, "CURRENT_BATCH_PATH", str(batch))
    monkeypatch.setattr(qg, "check_model_available", lambda m: checked.append(m))

    def fake_batch(n, seed, model, progress=None, **kw):
        got.update(kw)
        return pd.read_csv(batch), {"accepted": 2, "attempts": 3, "rejected": {"label_mismatch": 1}, "label_checked": 3, "label_failed": 1,
                                    "unfilled": {}, "confusions": {"CARD / block_card -> TRANSFER / make_transfer": 1}}, 5, False
    monkeypatch.setattr(qg, "make_fresh_batch", fake_batch)
    monkeypatch.setattr(pipeline, "unload_models", lambda models: unloaded.extend(models))
    monkeypatch.setattr(pipeline, "run_pipeline", lambda sample, **kw: None)
    pipeline.main(["2", "--fresh", "--label-model", "judge:7b", "--noquality", "--nocache", "--threshold-router"])
    assert checked == [qg.GENERATOR_MODEL, "judge:7b"] and unloaded == [qg.GENERATOR_MODEL, "judge:7b"] and got["label_model"] == "judge:7b"


# ---------- choosing how the judge is asked ----------

def name_judge(prompt):
    """Answers a name-style prompt correctly: replies with the option text whose words appear in the message."""
    message = prompt.split('Message: "')[1].split('"\n')[0]
    options = [l[2:] for l in prompt.splitlines() if l.startswith("- ") and "none of these" not in l]
    return next((o for o in options if o.split(": ")[-1] in message), "none of these")


def test_the_generator_can_ask_the_judge_by_name_and_records_the_method():
    df, stats = qg.generate_questions(pool(), 6, 1, stub_generate, classify_fn=name_judge, label_method="name")
    assert len(df) == 6 and set(df["label_method"]) == {"name"} and stats["label_failed"] == 0


def test_the_category_scope_forces_the_category_method_and_the_default_is_examples():
    cat, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=stub_classify, label_scope="category", label_method="name")
    default, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=stub_classify)
    skipped, _ = qg.generate_questions(pool(), 3, 1, stub_generate, classify_fn=None)
    assert set(cat["label_method"]) == {"category"} and set(default["label_method"]) == {"examples"} and set(skipped["label_method"]) == {"skipped"}


def test_an_unknown_label_method_is_refused_by_the_generator():
    with pytest.raises(ValueError, match="label_method must be one of"):
        qg.generate_questions(pool(), 2, 1, stub_generate, classify_fn=stub_classify, label_method="telepathy")


def test_the_label_method_option_needs_fresh_and_must_be_a_known_method():
    assert pipeline.parse_question_options(["10", "--fresh", "--label-method", "examples"])["label_method"] == "examples"
    assert pipeline.parse_question_options(["10", "--fresh"])["label_method"] is None
    with pytest.raises(SystemExit, match="label-method only applies"):
        pipeline.parse_question_options(["10", "--label-method", "examples"])
    with pytest.raises(SystemExit, match="must be one of"):
        pipeline.parse_question_options(["10", "--fresh", "--label-method", "telepathy"])


def test_a_fresh_run_passes_the_chosen_label_method_to_the_generator(tmp_path, monkeypatch):
    batch = tmp_path / "current_batch.csv"
    pd.DataFrame({"instruction": ["fresh question about my card"], "category": ["CARD"], "intent": ["block_card"]}).to_csv(batch, index=False)
    got = {}
    monkeypatch.setattr(qg, "CURRENT_BATCH_PATH", str(batch))
    monkeypatch.setattr(qg, "check_model_available", lambda m: None)
    monkeypatch.setattr(qg, "make_fresh_batch", lambda n, seed, model, progress=None, **kw: got.update(kw) or (
        pd.read_csv(batch), {"accepted": 1, "attempts": 1, "rejected": {}, "label_checked": 1, "label_failed": 0, "unfilled": {}, "confusions": {}}, 3, False))
    monkeypatch.setattr(pipeline, "unload_models", lambda models: None)
    monkeypatch.setattr(pipeline, "run_pipeline", lambda sample, **kw: None)
    pipeline.main(["1", "--fresh", "--label-method", "examples", "--noquality", "--nocache", "--threshold-router"])
    assert got["label_method"] == "examples"
    got.clear()
    pipeline.main(["1", "--fresh", "--noquality", "--nocache", "--threshold-router"])
    assert got["label_method"] == "examples"


# ---------- the defaults: one model, asked with examples ----------

def test_the_default_generator_is_qwen_and_the_default_way_of_asking_the_judge_is_examples():
    assert qg.GENERATOR_MODEL == "qwen2.5:7b" and qg.DEFAULT_LABEL_METHOD == "examples"
    assert qg.DEFAULT_LABEL_METHOD in qg.METHODS


def test_with_no_judge_named_the_generator_model_also_does_the_label_check_by_default(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(qg, "make_classify_fn", lambda model, **kw: used.append(model) or stub_classify)
    df, _, _, _ = qg.make_fresh_batch(3, pool=pool(), generate_fn=stub_generate, embed_fn=None,
                                      out_path=str(tmp_path / "b.csv"), ledger_path=str(tmp_path / "l.txt"))
    assert used == [qg.GENERATOR_MODEL] and set(df["label_model"]) == {qg.GENERATOR_MODEL} and set(df["label_method"]) == {"examples"}
