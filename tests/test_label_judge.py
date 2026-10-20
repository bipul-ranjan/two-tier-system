"""
Tests for src/label_judge.py: the six ways of asking a model which request type a question expresses, and the evaluation that
measures each way on questions with known labels.

The model is replaced by small stand-ins. A "perfect judge" reads the prompt, finds the right option and answers in whatever form the
prompt asks for, so every method must score 100% with it; any shortfall is then a bug in the method, not the model. An "always answers
1" judge proves the position analysis can detect a model that cannot cope with a long list.
"""
import os
import random
import re
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from src import label_judge as lj

PAIRS = [("CARD", "block_card"), ("CARD", "cancel_card"), ("TRANSFER", "make_transfer"), ("TRANSFER", "cancel_transfer"),
         ("ATM", "recover_swallowed_card"), ("LOAN", "apply_for_loan"), ("CONTACT", "human_agent")]
PAYMENTS = {"CARD", "TRANSFER", "ATM"}


def unit(category):
    return "payments" if category in PAYMENTS else "retail_bank"


def pool_frame(per=6):
    return pd.DataFrame([{"instruction": f"{i} please help with {intent.replace('_', ' ')} number {k}", "category": c, "intent": intent}
                         for i, (c, intent) in enumerate(PAIRS) for k in range(per)])


def context(per=6):
    pool = pool_frame(per)
    return lj.JudgeContext(set(zip(pool["category"], pool["intent"])), unit,
                           {k: list(v) for k, v in pool.groupby(["category", "intent"])["instruction"].apply(list).items()}), pool


def perfect_judge(truth_by_message):
    """Answers correctly in the form the prompt asks for: a number, the option text, or a category number."""
    def answer(prompt):
        message = prompt.split('Message: "')[1].split('"\n')[0]
        category, intent = truth_by_message[message]
        if "Categories:" in prompt:
            lines = [l for l in prompt.splitlines() if re.match(r"\d+\. ", l)]
            return next(l.split(".")[0] for l in lines if l.split(". ", 1)[1] == lj.words(category))
        if "exact option text" in prompt:
            return lj.option_text((category, intent))
        lines = [re.match(r"(\d+)\. (.*?)(?: \(for example: .*\))?$", l) for l in prompt.splitlines() if re.match(r"\d+\. ", l)]
        return next(m.group(1) for m in lines if m.group(2) == lj.option_text((category, intent)))
    return answer


def cases_of(pool):
    return [{"message": r.instruction, "category": r.category, "intent": r.intent} for r in pool.itertuples()]


def truth_map(cases):
    return {c["message"]: (c["category"], c["intent"]) for c in cases}


# ---------- prompts and reading replies ----------

def test_an_example_is_shown_beside_each_option_when_one_is_given():
    opts = [("CARD", "block_card"), ("CARD", "cancel_card")]
    p = lj.build_label_prompt("a message", opts, {("CARD", "block_card"): "please block it"})
    assert '1. card: block card (for example: "please block it")' in p and "2. card: cancel card\n" in p and "3. none of these" in p
    assert "for example" not in lj.build_label_prompt("a message", opts)


def test_a_reply_naming_an_option_is_matched_exactly_or_by_its_longest_text():
    opts = [("CARD", "activate_card"), ("CARD", "activate_card_international_usage")]
    assert lj.match_name("card: activate card international usage", opts) == ("option", opts[1])
    assert lj.match_name("Card: Activate Card.", opts) == ("option", opts[0])
    assert lj.match_name("I think it is card: activate card international usage, sure", opts) == ("option", opts[1])
    assert lj.match_name("none of these", opts) == ("none", None)
    assert lj.match_name("banana", opts) == ("unparsed", None) and lj.match_name("", opts) == ("unparsed", None)


def test_the_context_lists_options_by_unit_by_category_and_the_categories_of_a_unit():
    ctx, _ = context()
    assert set(ctx.options_for_unit("CARD")) == {p for p in PAIRS if p[0] in PAYMENTS} and len(ctx.options_for_unit("CARD")) == 5
    assert ctx.options_for_category("TRANSFER") == [("TRANSFER", "cancel_transfer"), ("TRANSFER", "make_transfer")]
    assert ctx.categories_of_unit("ATM") == ["ATM", "CARD", "TRANSFER"] and ctx.categories_of_unit("LOAN") == ["CONTACT", "LOAN"]


# ---------- the six methods ----------

@pytest.mark.parametrize("method", lj.METHODS)
def test_a_perfect_judge_is_right_every_time_with_every_method(method):
    ctx, pool = context()
    cases = cases_of(pool)
    judge_fn = perfect_judge(truth_map(cases))
    for c in cases:
        kind, pred, position = lj.judge(method, judge_fn, c["message"], (c["category"], c["intent"]), ctx, random.Random(1))
        assert (kind, pred) == ("option", (c["category"], c["intent"])), (method, c)


def test_shuffling_changes_the_order_the_model_sees_but_not_the_answer():
    ctx, pool = context()
    seen = {"number": [], "shuffled": []}
    c = cases_of(pool)[0]
    for m in seen:
        lj.judge(m, lambda p, m=m: seen[m].append(p) or "1", c["message"], (c["category"], c["intent"]), ctx, random.Random(5))
    order = lambda p: [l.split(". ", 1)[1] for l in p.splitlines() if re.match(r"\d+\. ", l)]
    assert sorted(order(seen["number"][0])) == sorted(order(seen["shuffled"][0])) and order(seen["number"][0]) != order(seen["shuffled"][0])
    again = []
    lj.judge("shuffled", lambda p: again.append(p) or "1", c["message"], (c["category"], c["intent"]), ctx, random.Random(5))
    assert again == [seen["shuffled"][0]]  # the same seed gives the same order


def test_the_unit_methods_offer_the_whole_unit_and_the_category_method_only_one_category():
    ctx, pool = context()
    c = next(c for c in cases_of(pool) if c["category"] == "CARD")
    prompts = {}
    for m in ("number", "category"):
        lj.judge(m, lambda p, m=m: prompts.setdefault(m, p) and "1", c["message"], (c["category"], c["intent"]), ctx)
    assert "transfer: make transfer" in prompts["number"] and "loan" not in prompts["number"]
    assert "card: block card" in prompts["category"] and "transfer" not in prompts["category"]


def test_the_examples_method_never_shows_the_question_being_judged_as_an_example():
    ctx, pool = context()
    c = cases_of(pool)[0]
    seen = []
    lj.judge("examples", lambda p: seen.append(p) or "1", c["message"], (c["category"], c["intent"]), ctx, random.Random(2))
    shown = " ".join(re.findall(r'\(for example: "(.*?)"\)', seen[0]))
    assert 'for example: "' in seen[0] and c["message"][:60] not in shown.replace("\n", " ")


def test_two_step_asks_about_the_category_first_and_then_only_that_categorys_intents():
    ctx, pool = context()
    c = next(c for c in cases_of(pool) if c["category"] == "TRANSFER")
    prompts = []
    judge_fn = perfect_judge(truth_map(cases_of(pool)))
    lj.judge("two-step", lambda p: prompts.append(p) or judge_fn(p), c["message"], (c["category"], c["intent"]), ctx)
    assert len(prompts) == 2 and "Categories:" in prompts[0] and "Categories:" not in prompts[1]
    assert "transfer: make transfer" in prompts[1] and "card: block card" not in prompts[1]
    stopped = []
    kind, pred, _ = lj.judge("two-step", lambda p: stopped.append(p) or "99", c["message"], (c["category"], c["intent"]), ctx)
    assert (kind, pred) == ("unparsed", None) and len(stopped) == 1


def test_an_unknown_method_is_refused():
    ctx, pool = context()
    with pytest.raises(ValueError, match="method must be one of"):
        lj.judge("telepathy", lambda p: "1", "m", ("CARD", "block_card"), ctx)


# ---------- measuring ----------

def test_the_wilson_interval_matches_known_values_and_stays_in_range():
    lo, hi = lj.wilson(50, 100)
    assert abs(lo - 0.404) < 0.002 and abs(hi - 0.596) < 0.002
    assert lj.wilson(0, 0) == (0.0, 0.0) and abs(lj.wilson(100, 100)[1] - 1.0) < 1e-9 and lj.wilson(0, 50)[0] == 0.0


def test_even_sampling_takes_the_same_number_from_every_request_type_and_repeats_for_a_seed():
    _, pool = context(per=10)
    cases = lj.sample_cases(pool, 14, seed=3)
    counts = pd.Series([(c["category"], c["intent"]) for c in cases]).value_counts()
    assert len(cases) == 14 and set(counts) == {2} and len(counts) == 7
    assert [c["message"] for c in cases] == [c["message"] for c in lj.sample_cases(pool, 14, seed=3)]
    few = lj.sample_cases(pool, 3, seed=3)
    assert len(few) == 3 and len({(c["category"], c["intent"]) for c in few}) == 3
    uneven = pd.Series([(c["category"], c["intent"]) for c in lj.sample_cases(pool, 10, seed=3)]).value_counts()
    assert sorted(uneven) == [1, 1, 1, 1, 2, 2, 2]
    assert len(lj.sample_cases(pool, 20, seed=1, even=False)) == 20


def test_an_evaluation_has_a_row_per_case_and_method_and_counts_the_model_calls():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 14, seed=1)
    df = lj.run_evaluation(cases, ["number", "two-step"], perfect_judge(truth_map(cases)), ctx, seed=1)
    assert len(df) == 28 and df["correct"].all()
    s = lj.summarize_evaluation(df).set_index("method")
    assert s.loc["number", "accuracy"] == 1.0 and s.loc["number", "calls_per_question"] == 1.0 and s.loc["two-step", "calls_per_question"] == 2.0
    assert s.loc["number", "unreadable"] == 0.0 and s.loc["number", "low"] > 0.7


def test_a_judge_that_always_answers_one_fails_late_in_the_list_and_the_position_table_shows_it():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 28, seed=2)
    df = lj.run_evaluation(cases, ["number"], lambda p: "1", ctx, seed=1)
    pos = lj.position_table(df, "number")
    assert pos["last third"][1] > 0 and pos["last third"][0] == 0.0
    assert pos["first third"][0] is not None and pos["first third"][0] > 0.0


def test_unreadable_replies_are_counted_separately_from_wrong_ones():
    ctx, pool = context()
    df = lj.run_evaluation(lj.sample_cases(pool, 7, seed=1), ["number", "name"], lambda p: "banana", ctx, seed=1)
    s = lj.summarize_evaluation(df)
    assert (s["unreadable"] == 1.0).all() and (s["accuracy"] == 0.0).all()


def test_the_hardest_types_are_listed_worst_first():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 28, seed=4)
    good = perfect_judge(truth_map(cases))

    def fails_on_make_transfer(prompt):
        return "999" if "make transfer number" in prompt else good(prompt)
    df = lj.run_evaluation(cases, ["number"], fails_on_make_transfer, ctx, seed=1)
    assert lj.hardest_table(df, "number", top=3) == [("TRANSFER / make_transfer", 0, 4)]  # only types with errors are listed
    everything = lj.hardest_table(df, "number", top=3, only_errors=False)
    assert everything[0] == ("TRANSFER / make_transfer", 0, 4) and all(c == t for _, c, t in everything[1:])
    perfect = lj.run_evaluation(cases, ["number"], good, ctx, seed=1)
    assert lj.hardest_table(perfect, "number") == []


def test_the_same_seed_repeats_an_evaluation_exactly():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 14, seed=9)
    a = lj.run_evaluation(cases, ["shuffled", "examples"], lambda p: "2", ctx, seed=9)
    b = lj.run_evaluation(cases, ["shuffled", "examples"], lambda p: "2", ctx, seed=9)
    pd.testing.assert_frame_equal(a, b)


def test_a_type_tested_only_once_and_judged_wrong_is_still_listed_and_the_note_knows_it_is_rough():
    """The bug this pins down: with one question per type, a minimum of two tests per type hid every error and the report said 'none wrong'."""
    ctx, pool = context()
    cases = lj.sample_cases(pool, 7, seed=1)  # exactly one question for each of the 7 request types
    df = lj.run_evaluation(cases, ["number"], lambda p: "999", ctx, seed=1)
    listed = lj.hardest_table(df, "number")
    assert len(listed) == 6 and all(c == 0 and t == 1 for _, c, t in listed)
    assert lj.tests_per_type(df, "number") == 1


def test_the_most_common_mix_ups_are_counted_by_intended_type_and_what_the_judge_chose():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 14, seed=2)

    def says_make_transfer(prompt):
        options = [l.split(". ", 1)[1] for l in prompt.splitlines() if re.match(r"\d+\. ", l)]
        return str(next(i for i, o in enumerate(options, start=1) if o == "transfer: make transfer")) if any(o == "transfer: make transfer" for o in options) else "1"
    df = lj.run_evaluation(cases, ["number"], says_make_transfer, ctx, seed=1)
    top = lj.confusion_table(df, "number")
    assert top[0][1] == "TRANSFER / make_transfer" and top[0][2] >= 2 and top[0][0].split(" / ")[0] in PAYMENTS
    assert all(a != b for a, b, _ in top)


def test_none_of_these_and_unusable_answers_are_named_in_the_mix_ups():
    ctx, pool = context()
    cases = lj.sample_cases(pool, 7, seed=3)
    none_df = lj.run_evaluation(cases, ["number"], lambda p: str(len([l for l in p.splitlines() if re.match(r"\d+\. ", l)])), ctx, seed=1)
    junk_df = lj.run_evaluation(cases, ["number"], lambda p: "banana", ctx, seed=1)
    assert {b for _, b, _ in lj.confusion_table(none_df, "number")} == {"none of these"}
    assert {b for _, b, _ in lj.confusion_table(junk_df, "number")} == {"no usable answer"}
