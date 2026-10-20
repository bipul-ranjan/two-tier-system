"""
Measures how well a model can tell which request type a customer question expresses, using questions from your pool whose
labels are known to be right. No generator is involved, no Claude call is made and nothing costs money.

Why: the question generator keeps a question only if a judge model picks its intended request type. In real runs the judge rejected
most good-looking questions and sometimes named unrelated types. This script measures the judge directly, for six ways of asking
(see src/label_judge.py) and for any models you name, so the way of asking and the model can be chosen on evidence.

It prints, for each model: the accuracy of each method with its 95% interval, the share of unreadable replies and the model calls
per question; whether the judge does worse on options near the end of the list; the request types it gets wrong most often; and
a recommendation that says plainly when two methods cannot be told apart. Chance is about 4%, because a question has 23 to 27 options.

Speed: one call per question per method (two for two-step), about 1 to 3 seconds each on a recent Mac. The default of 100 questions and
all six methods is about 700 calls, so 15 to 30 minutes per model. Start small to see it work:  --n 50 --methods number,examples

Needs Ollama running and each judge model downloaded (ollama pull NAME). The model names must appear in ollama list.

Options:
    --n N                questions to test per method (default 100)
    --model NAMES        one judge model, or several separated by commas (default qwen2.5:7b), for example qwen2.5:7b,llama3.1:8b
    --methods NAMES      the methods to test, separated by commas (default all six: number, shuffled, name, two-step, examples, category)
    --sample NAME        even (the same number from every request type; the default) or pool (random questions, as the pool is mixed)
    --seed N             the seed for the choice of questions and shuffles (default 1); the same seed tests the same questions
    --show-wrong N       print N wrong answers for the best method of each model (default 5)
    --out PATH           where to save every answer as a CSV (default results/analysis/label_judge_<date-time>.csv)

Then use the best way of asking in the generator:  python scripts/generate_questions.py --label-method examples --label-model NAME
or in a run:  python -m src.pipeline 100 --fresh --label-method examples --label-model NAME
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.label_judge import (METHODS, JudgeContext, confusion_table, hardest_table, position_table, run_evaluation, sample_cases,
                             summarize_evaluation, tests_per_type)
from src.question_generator import GENERATOR_MODEL, check_model_available, load_pool, make_classify_fn


def show_model(model: str, df: pd.DataFrame, show_wrong: int) -> pd.DataFrame:
    """Print the results for one model and return its summary table."""
    summary = summarize_evaluation(df)
    print(f"\n=== Judge model: {model} ({df['message'].nunique()} questions) ===")
    print(f"{'method':10s} {'accuracy':>9s}   {'95% interval':14s} {'unreadable':>10s} {'calls/question':>15s}")
    for r in summary.itertuples():
        print(f"{r.method:10s} {r.accuracy:9.0%}   {r.low:5.0%} to {r.high:4.0%}   {r.unreadable:10.0%} {r.calls_per_question:15.1f}")
    for method in ("number", "shuffled"):
        if method in set(df["method"]):
            pos = position_table(df, method)
            print(f"Where the right answer stood in the list ({method}): " + ", ".join(
                f"{k} {'n/a' if v[0] is None else f'{v[0]:.0%}'} (of {v[1]})" for k, v in pos.items()))
    best = summary.sort_values(["accuracy", "calls_per_question"], ascending=[False, True]).iloc[0]
    real = summary[summary["method"] != "category"]  # "category" assumes the category is known, so it cannot be a recommendation
    top = real.sort_values(["accuracy", "calls_per_question"], ascending=[False, True]).iloc[0] if len(real) else best
    hard = hardest_table(df, top["method"])
    print(f"Request types {top['method']} got wrong most often: " + ("; ".join(f"{n} {c}/{t} right" for n, c, t in hard) if hard else "none, every tested type was judged right"))
    per_type = tests_per_type(df, top["method"])
    if hard and per_type < 3:
        print(f"  (Each request type was tested only {per_type} time{'s' if per_type != 1 else ''}, so this list is rough. Use --n 150 or more for a firmer one.)")
    repeated = [(a, b, c) for a, b, c in confusion_table(df, top["method"]) if c >= 2]
    if repeated:
        print(f"Mix-ups that repeated under {top['method']} (intended -> chosen): " + "; ".join(f"{a} -> {b} x{c}" for a, b, c in repeated[:6]))
    elif hard:
        print(f"No mix-up has repeated yet under {top['method']}; more questions are needed to see patterns.")
    ties = [r.method for r in real.itertuples() if r.method != top["method"] and r.high >= top["low"] and r.low <= top["high"]]
    verdict = f"Best method that does not assume the category: {top['method']} at {top['accuracy']:.0%} ({top['low']:.0%} to {top['high']:.0%})."
    if ties:
        verdict += f" Not clearly better than: {', '.join(ties)} (the intervals overlap), so prefer the cheaper one."
    print(verdict)
    if "category" in set(summary["method"]):
        c = summary[summary["method"] == "category"].iloc[0]
        print(f"(The old category-only check scored {c['accuracy']:.0%}, but it is told the category, so it is only a reference.)")
    if top["accuracy"] < 0.6:
        print("Warning: below about 60%, this judge is too unreliable to decide which generated questions to keep. Try a larger --model.")
    if show_wrong:
        wrong = df[(df["method"] == top["method"]) & (~df["correct"])].head(show_wrong)
        for w in wrong.itertuples():
            print(f"  wrong: [{w.true_category} / {w.true_intent}] judged as [{w.pred_category or w.kind} / {w.pred_intent}]  {w.message[:90]}")
    return summary.assign(model=model)


def main() -> None:
    """Run the evaluation for each model and print a comparison."""
    ap = argparse.ArgumentParser(description="Measure how well a model judges which request type a question expresses.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default=GENERATOR_MODEL)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--sample", choices=["even", "pool"], default="even")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show-wrong", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        sys.exit(f"Unknown method(s): {', '.join(bad)}. Choose from: {', '.join(METHODS)}")
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    try:
        for m in models:
            check_model_available(m)
    except RuntimeError as e:
        sys.exit(str(e))

    pool = load_pool()
    cases = sample_cases(pool, args.n, args.seed, even=args.sample == "even")
    ctx = JudgeContext(set(zip(pool["category"], pool["intent"])), examples={k: list(v) for k, v in pool.groupby(["category", "intent"])["instruction"].apply(list).items()})
    calls = len(cases) * sum(2 if m == "two-step" else 1 for m in methods) * len(models)
    print(f"Testing {len(cases)} pool questions ({args.sample}) with {len(methods)} method(s) on {len(models)} model(s): about {calls} model calls, "
          f"roughly {calls * 2 // 60 + 1} minutes. Seed {args.seed}.")

    summaries, frames = [], []
    for model in models:
        def progress(method, i, total, model=model):
            if i == total:
                print(f"  {model} / {method}: {total} questions judged")
        df = run_evaluation(cases, methods, make_classify_fn(model), ctx, args.seed, progress)
        df.insert(0, "model", model)
        frames.append(df)
        summaries.append(show_model(model, df, args.show_wrong))

    if len(models) > 1:
        both = pd.concat(summaries)
        print("\n=== Comparison (accuracy, by method and model) ===")
        print(both.pivot(index="method", columns="model", values="accuracy").apply(lambda col: col.map(lambda v: f"{v:.0%}")).to_string())
    out = args.out or f"results/analysis/label_judge_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.concat(frames).to_csv(out, index=False)
    print(f"\nSaved every answer to {out}")


if __name__ == "__main__":
    main()
