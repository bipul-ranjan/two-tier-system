"""
Generates additional SYNTHETIC training examples for the exception scenarios
(PAYMENT_EXCEPTION, RETAIL_EXCEPTION) that the real Bitext dataset does not
cover at all -- see fetch_bitext_banking.py's category mapping, which has no
exception category (PAYMENTS_CATEGORIES / RETAIL_BANK_CATEGORIES are both
"normal" scenarios only). This reuses the exact templates already used for
pipeline testing (generate_synthetic_bitext.py's INTENTS["retail_exception"]
and INTENTS["payments_exception"]), formatted as training rows instead.

Why this exists: both local models were fine-tuned only on the real Bitext
data's "normal" categories, so neither has seen anything resembling a fraud,
hardship, or vulnerable-customer scenario during training. That out-of-
distribution gap is the main reason exception queries score the lowest
confidence of any category for both business units in pipeline runs -- and
worst of all for retail_bank, whose smaller base model generalizes worse to
scenarios it was never trained on.

Output is a SEPARATE file per business unit, not merged into the real Bitext
files, so provenance stays clear for your methodology write-up:
    data/raw/retail_exception_synthetic.jsonl
    data/raw/payments_exception_synthetic.jsonl

Deduplicated against every instruction already in
data/synthetic/synthetic_bitext_*.csv, so none of this training data can
overlap with what the pipeline uses as test queries in a run -- otherwise a
query the model was fine-tuned on verbatim could show artificially high
confidence in a pipeline run, which would be a train/test leak.

Run once, from the project root, before your next Colab retrain:
    python scripts/generate_exception_training_data.py
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_synthetic_bitext as gen  # reuse its templates + helper functions

SEED = 20260928 + 1  # different from the pipeline-test generator's seed, so this data is distinct
ROWS_PER_INTENT = 350  # brings exception intents to roughly the same volume as retail's normal intents get in pipeline-test data (~374 avg)
MAX_ATTEMPTS_PER_ROW = 40  # give up on an intent's template pool after this many failed unique-instruction attempts per row
OUT_DIR = "data/raw"

GROUPS = {
    "retail_exception": f"{OUT_DIR}/retail_exception_synthetic.jsonl",
    "payments_exception": f"{OUT_DIR}/payments_exception_synthetic.jsonl",
}


def existing_synthetic_instructions() -> set:
    """Every instruction already used as a pipeline test query, so training data never overlaps with it."""
    seen = set()
    for path in glob.glob("data/synthetic/synthetic_bitext_*.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["instruction"])
    return seen


def main():
    rng = gen.random.Random(SEED)
    blocked = existing_synthetic_instructions()
    print(f"Loaded {len(blocked)} existing pipeline-test instructions to avoid duplicating.\n")

    for group, out_path in GROUPS.items():
        seen_here = set()
        rows = []
        for category, intent, tone, instrs, resps in gen.INTENTS[group]:
            made = 0
            attempts = 0
            while made < ROWS_PER_INTENT and attempts < ROWS_PER_INTENT * MAX_ATTEMPTS_PER_ROW:
                attempts += 1
                slots = gen.make_slots(rng)
                template = rng.choice(instrs)
                usable = [r for r in resps if gen.fields(r) <= gen.fields(template)]
                body = rng.choice(usable) if usable else resps[-1]
                instruction = gen.style(rng, template.format(**slots))
                if instruction in blocked or instruction in seen_here:
                    continue
                response = gen.build_response(rng, tone, body.format(**slots))
                seen_here.add(instruction)
                rows.append({"instruction": instruction, "category": category, "intent": intent, "response": response})
                made += 1
            flag = "" if made == ROWS_PER_INTENT else f"  <-- only {made}/{ROWS_PER_INTENT}, template pool ran dry"
            print(f"  {group}/{intent}: {made} rows{flag}")

        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {out_path}: {len(rows)} rows across {len(gen.INTENTS[group])} intents\n")


if __name__ == "__main__":
    main()
