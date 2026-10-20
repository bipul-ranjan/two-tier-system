"""
Generates a batch of fresh customer questions with a separate local model, and saves them as a CSV the pipeline can use.

Why: so a test run does not keep drawing from the same fixed pool of 16,000 questions. Freshly worded questions give an
honest cache hit rate, and test whether the router copes with wording it has never seen. See src/question_generator.py for how
a question is made and checked.

Needs: Ollama running, and a general model downloaded (not one of your fine-tuned helpers). The default is qwen2.5:7b:
    ollama pull qwen2.5:7b
The near-duplicate check uses the same small embedding model as the cache; if it is not available the script says so and checks
exact copies only. Nothing here calls Claude and nothing costs money.

Speed: about 2 to 6 seconds per accepted question on a recent Mac (the label check adds a second call). Expect some attempts to be
rejected; the summary says why.

The mix is kept: the request types for the whole batch are drawn first (the pool's own mix, or a chosen exception share), and each is
retried up to --per-slot-attempts times. A type the model cannot produce within that many tries is reported as unfilled and the batch
ends up a little short of N, rather than being filled with easier types and drifting away from the pool's mix.

Always look at a few results before trusting a batch:  python scripts/generate_questions.py --n 20 --show 20

Options:
    --n N                 how many questions to produce (default 100)
    --seed N              the batch seed (default: a random one, which is printed so the batch can be repeated)
    --model NAME          the generator model (default qwen2.5:7b)
    --out PATH            the CSV to write (default data/generated/generated_questions_<date-time>.csv)
    --unit NAME           only payments or only retail_bank
    --exception-share X   the share of questions from the exception categories, between 0 and 1 (default: the pool's own mix)
    --temperature X       randomness of the generator, 0 to 2 (default 0.9; higher is more varied and less reliable)
    --no-label-check      skip checking that each question really expresses its intended request type (faster, less reliable labels)
    --label-scope NAME    what the label check chooses among: unit (default; every intent of the business unit, 23 or 27 options, which
                          catches a question that belongs to another category) or category (only the same category's intents: faster
                          and easier, but it passes almost anything for a category with one or two intents)
    --label-method NAME   how the judge is asked: examples (default), number, shuffled, name, two-step or category. Measure them first with
                          scripts/evaluate_label_judge.py and use the best one (see src/label_judge.py for what each does)
    --label-model NAME    a separate model to do the label check (default: the generator model itself). A larger model here stops a small
                          model marking its own work, for example one you have downloaded with ollama pull; it must be in ollama list
    --save-rejected PATH  also write every rejected candidate (the text, its request type, the reason and, for a label mismatch, what the
                          judge chose instead) to a CSV, so you can read what was thrown away and judge whether the judge or the question was wrong
    --per-slot-attempts N tries per question slot before giving up on that request type (default 6)
    --dedup-pool          also refuse questions whose meaning is close to one in the 16,000-question pool (slower start)
    --dedup-threshold X   how alike two questions must be to count as near-duplicates (default 0.92)
    --show N              print the first N generated questions

Then run the pipeline on the batch:
    python -m src.pipeline 100 --questions data/generated/<the file it wrote>.csv
"""
import argparse
import os
import random
import sys
from datetime import datetime


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.label_judge import METHODS
from src.question_generator import (DEFAULT_LABEL_METHOD, GENERATOR_MODEL, check_model_available, confusion_lines, default_embed_fn, generate_questions,
                                    hardest_lines, load_pool, load_previous, make_classify_fn, make_ollama_fn, mix_summary,
                                    rejected_to_frame)


def main() -> None:
    """Generate the batch, save it, and print a summary."""
    ap = argparse.ArgumentParser(description="Generate fresh customer questions with a separate local model.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=GENERATOR_MODEL)
    ap.add_argument("--out", default=None)
    ap.add_argument("--unit", choices=["payments", "retail_bank"], default=None)
    ap.add_argument("--exception-share", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--no-label-check", action="store_true")
    ap.add_argument("--label-scope", choices=["unit", "category"], default="unit")
    ap.add_argument("--label-method", choices=list(METHODS), default=DEFAULT_LABEL_METHOD)
    ap.add_argument("--label-model", default=None)
    ap.add_argument("--save-rejected", default=None)
    ap.add_argument("--per-slot-attempts", type=int, default=6)
    ap.add_argument("--dedup-pool", action="store_true")
    ap.add_argument("--dedup-threshold", type=float, default=0.92)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    if args.exception_share is not None and not 0 <= args.exception_share <= 1:
        sys.exit("--exception-share must be between 0 and 1")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(1, 10**6)

    try:
        for needed in {args.model, args.label_model} - {None}:
            check_model_available(needed)
    except RuntimeError as e:
        sys.exit(str(e))
    pool, previous = load_pool(), load_previous()
    print(f"Generator model: {args.model} | seed {seed} | {len(pool)} pool questions, {len(previous)} from earlier batches")
    embed_fn = default_embed_fn()
    out = args.out or f"data/generated/generated_questions_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

    def progress(done, total, attempts):
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} accepted after {attempts} attempts")

    df, stats = generate_questions(
        pool, args.n, seed, make_ollama_fn(args.model, temperature=args.temperature),
        classify_fn=None if args.no_label_check else make_classify_fn(args.label_model or args.model), embed_fn=embed_fn, previous=previous,
        model=args.model, unit=args.unit, exception_share=args.exception_share, dedup_threshold=args.dedup_threshold,
        dedup_pool_semantically=args.dedup_pool, progress=progress,
        label_scope=args.label_scope, per_slot_attempts=args.per_slot_attempts, label_model=args.label_model,
        label_method=args.label_method)

    if df.empty:
        sys.exit("No questions were accepted. Run again with --show on a small --n, or check the model with: ollama run " + args.model)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nAccepted {stats['accepted']} of {args.n} requested, from {stats['attempts']} attempts.")
    if stats["rejected"]:
        print("Rejected: " + ", ".join(f"{k} {v}" for k, v in sorted(stats["rejected"].items(), key=lambda kv: -kv[1])))
    if stats["label_checked"]:
        print(f"Label check (among every intent of the {'business unit' if args.label_scope == 'unit' else 'category'}): "
              f"{stats['label_checked'] - stats['label_failed']} of {stats['label_checked']} questions expressed their intended request type "
              f"({1 - stats['label_failed'] / stats['label_checked']:.0%}). Low numbers mean the model struggles with these request types.")
    if stats["label_failed"]:
        print("Most common label mix-ups (what the question was meant to be -> what the judge chose):")
        for line in confusion_lines(stats):
            print("  " + line)
        print("Request types that passed the label check least often:")
        for line in hardest_lines(stats):
            print("  " + line)
    if args.save_rejected:
        os.makedirs(os.path.dirname(args.save_rejected) or ".", exist_ok=True)
        rejected = rejected_to_frame(stats)
        rejected.to_csv(args.save_rejected, index=False)
        print(f"Wrote {len(rejected)} rejected candidates to {args.save_rejected} (read a few: is the judge wrong, or is the question?)")
    mix = mix_summary(df, pool)
    print(f"Exception questions: {int(df['is_exception'].sum())} of {len(df)} ({mix['batch_exception_share']:.0%}; the pool has "
          f"{mix['pool_exception_share']:.0%}). Distinct intents: {df['intent'].nunique()}.")
    if stats["unfilled"]:
        print(f"Unfilled: {sum(stats['unfilled'].values())} of {args.n} slots, because the model could not produce these within "
              f"{args.per_slot_attempts} tries each: " + ", ".join(f"{k} x{v}" for k, v in sorted(stats["unfilled"].items(), key=lambda kv: -kv[1])))
        print("  The batch is short by that many questions and keeps the pool's mix for everything else. To fill them, raise --per-slot-attempts, "
              "try --label-scope category, or use a larger generator with --model.")
    if mix["missing_categories"] and args.n >= 40:
        print("Note: no question from " + ", ".join(mix["missing_categories"]) + ". The label check or the model may be struggling with these categories.")
    for _, r in df.head(args.show).iterrows():
        print(f"  [{r['category']} / {r['intent']}] {r['instruction']}")
    print(f"\nWrote {out}\nRun the pipeline on it with:\n  python -m src.pipeline {len(df)} --questions {out}")


if __name__ == "__main__":
    main()
