"""
Measures the personal-data filter (src/privacy.py) on labelled test cases, so a claim about it rests on a number.

What it does, with a fixed random seed so the result repeats:
1. RECALL. For every identifier type, it builds N test sentences that each contain one made-up identifier in a
   realistic format (valid card numbers, valid IBANs, phone numbers in several formats, and so on) and counts how
   often the identifier is gone from the filtered text. Recall is only meaningful for the types the filter is
   designed to catch. Two extra types are included on purpose because the filter is NOT designed for them, so the
   gap is measured and not assumed: NAME_FREE ("I'm Priya") and ADDRESS_FREE ("I live at 12 Baker Street").
2. FALSE ALARMS. It runs the filter over (a) a sample of your real synthetic questions and (b) a set of tricky
   look-alikes (amounts, dates, rates, reference numbers) and reports how often text was changed that should not
   have been, and which types fired.

It does NOT prove the filter is safe for real customer data: the test values are generated, real text is messier,
and the identifier formats are the ones this script knows. Treat the numbers as a floor for what to test on real,
labelled samples with the privacy team.

Usage:
    python scripts/evaluate_privacy_filter.py                    # default: 200 cases per type, 3000 real questions
    python scripts/evaluate_privacy_filter.py --n 500 --seed 3
    python scripts/evaluate_privacy_filter.py --min-recall 0.95  # exit code 1 if any covered type is below 0.95
    python scripts/evaluate_privacy_filter.py --out results/analysis/privacy_filter_eval.csv   # per-case details
"""
import argparse
import glob
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.privacy import FILTER_VERSION, redact

COVERED = ["EMAIL", "PHONE", "CARD_NUMBER", "IBAN", "ACCOUNT_NUMBER", "SORT_CODE", "NATIONAL_ID", "CARD_LAST4", "CARD_BARE",
           "CARD_ENDING_LONG", "DATE_OF_BIRTH", "POSTCODE", "NAME_CUE", "NAME_TITLE"]
NOT_COVERED = ["NAME_FREE", "ADDRESS_FREE"]
FIRST = ["Priya", "Rahul", "Anita", "Vikram", "Meera", "Arjun", "Sara", "Daniel", "Emma", "Omar"]
LAST = ["Nair", "Sharma", "Rao", "Mehta", "Khan", "Patel", "Smith", "Jones", "Brown", "Singh"]
STREETS = ["Baker Street", "Park Lane", "MG Road", "High Street", "Church Road", "Station Road"]
FRAMES = ["Hi, {v}. Can you help me with my account?", "Please check this for me: {v}, it is urgent.",
          "I need to update my details, {v}, before Friday.", "About my last transaction, {v}. Thanks.",
          "Good morning, {v}. I have a question about fees.", "Quick one: {v}. Please advise."]


def luhn_complete(prefix: str) -> str:
    """Append the Luhn check digit to a string of digits."""
    total = 0
    for i, ch in enumerate(reversed(prefix)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return prefix + str((10 - total % 10) % 10)


def make_iban(rng) -> str:
    """A valid GB IBAN with random digits, written in groups of four."""
    bban = "WEST" + "".join(rng.choice("0123456789") for _ in range(14))
    digits = "".join(str(int(c, 36)) for c in bban + "GB00")
    check = 98 - int(digits) % 97
    iban = f"GB{check:02d}{bban}"
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))


def digits(rng, n: int) -> str:
    """n random digits."""
    return "".join(rng.choice("0123456789") for _ in range(n))


def make_case(kind: str, rng):
    """Return (the sensitive value, the text containing it) for one test case of the given type."""
    frame = rng.choice(FRAMES)
    if kind == "EMAIL":
        v = f"{rng.choice(FIRST).lower()}.{rng.choice(LAST).lower()}{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'example.org', 'mail.co.uk'])}"
        return v, frame.format(v=f"my email is {v}")
    if kind == "PHONE":
        v = rng.choice([f"+44 7{digits(rng, 3)} {digits(rng, 6)}", f"07{digits(rng, 9)}", f"+91 9{digits(rng, 4)} {digits(rng, 5)}",
                        f"({digits(rng, 3)}) {digits(rng, 3)}-{digits(rng, 4)}", f"9{digits(rng, 9)}"])
        return v, frame.format(v=f"call me on {v}")
    if kind == "CARD_NUMBER":
        raw = luhn_complete(rng.choice(["4", "5"]) + digits(rng, 14))
        v = rng.choice([raw, " ".join(raw[i:i + 4] for i in range(0, 16, 4)), "-".join(raw[i:i + 4] for i in range(0, 16, 4))])
        return v, frame.format(v=f"my card {v}")
    if kind == "IBAN":
        v = make_iban(rng)
        return v, frame.format(v=f"send it to IBAN {v}")
    if kind == "ACCOUNT_NUMBER":
        v = rng.choice([digits(rng, 8), digits(rng, 12), digits(rng, 10)])
        return v, frame.format(v=f"my account number is {v}")
    if kind == "SORT_CODE":
        v = f"{digits(rng, 2)}-{digits(rng, 2)}-{digits(rng, 2)}"
        return v, frame.format(v=f"sort code {v}")
    if kind == "NATIONAL_ID":
        v = rng.choice([f"{digits(rng, 3)}-{digits(rng, 2)}-{digits(rng, 4)}", f"{rng.choice('ABCEGHJKLMNPRSTWXYZ')}{rng.choice('ABCEGHJKLMNPRSTWXYZ')} {digits(rng, 2)} {digits(rng, 2)} {digits(rng, 2)} {rng.choice('ABCD')}",
                        "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)) + digits(rng, 4) + rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ"),
                        f"{digits(rng, 4)} {digits(rng, 4)} {digits(rng, 4)}"])
        return v, frame.format(v=f"my ID is {v}")
    if kind == "CARD_LAST4":
        v = digits(rng, 4)
        return v, frame.format(v=rng.choice([f"the card ending {v}", f"last four digits {v}"]))
    if kind == "CARD_BARE":
        v = digits(rng, 4)
        return v, frame.format(v=rng.choice([f"my card {v}", f"the card number {v}", f"card no. {v}", f"Mastercard {v}", f"card #{v}"]))
    if kind == "CARD_ENDING_LONG":
        v = digits(rng, rng.choice([5, 6]))
        return v, frame.format(v=f"my card ending {v}")
    if kind == "DATE_OF_BIRTH":
        v = f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1950, 2004)}"
        return v, frame.format(v=rng.choice([f"I was born on {v}", f"DOB: {v}"]))
    if kind == "POSTCODE":
        v = rng.choice(["SW1A 1AA", "E14 9SR", "M1 1AE", "B33 8TH", "EC1A 1BB", "W1A 0AX", "N1 9GU", "CR2 6XH"])
        return v, frame.format(v=f"I live at {v}")
    if kind == "NAME_CUE":
        v = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        return v, frame.format(v=f"my name is {v}")
    if kind == "NAME_TITLE":
        v = f"{rng.choice(['Mr', 'Mrs', 'Ms', 'Dr'])} {rng.choice(LAST)}"
        return v, frame.format(v=f"{v} is the account holder")
    if kind == "NAME_FREE":
        v = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        return v, frame.format(v=rng.choice([f"I'm {v}", f"this is {v} calling"]))
    if kind == "ADDRESS_FREE":
        v = f"{rng.randint(1, 99)} {rng.choice(STREETS)}"
        return v, frame.format(v=f"I live at {v}")
    raise ValueError(kind)


LOOKALIKES = ["I was charged $1,300 at the shop.", "The loan is 50000 rupees over 24 months at 5.75%.",
              "I paid on 12/05/2024 and again on 03/06/2024.", "My reference is 4829 and the ticket is 123456.",
              "The meeting is at 10:30 on Friday.", "Transfer 2020 dollars to savings.", "Is ISO 20022 supported?",
              "Interest was 3.2% on 15000 for 2 years.", "Order 98765 arrived on 01/02/2025.", "Please raise the limit to 25,000.",
              "Version 2.1.4 of the app crashes.", "The fee is 12.50 per month.", "My card expires in 2027.", "I renewed my card in 2026.",
              "Card payments of 2500 rupees went through.", "I have 3 cards with you.", "The 2024 card offer ended."]


def main() -> None:
    """Run the recall and false-alarm measurements and print the summary."""
    ap = argparse.ArgumentParser(description="Measure the personal-data filter on labelled test cases.")
    ap.add_argument("--n", type=int, default=200, help="test cases per identifier type (default 200)")
    ap.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    ap.add_argument("--sample", type=int, default=3000, help="real synthetic questions to check for false alarms (default 3000)")
    ap.add_argument("--min-recall", type=float, default=None, help="exit with code 1 if any covered type is below this recall")
    ap.add_argument("--out", default=None, help="write every test case and its result to this CSV")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, summary = [], []
    for kind in COVERED + NOT_COVERED:
        hit = 0
        for _ in range(args.n):
            value, text = make_case(kind, rng)
            out = redact(text).text
            removed = value not in out
            hit += removed
            rows.append({"type": kind, "value": value, "input": text, "output": out, "removed": removed})
        summary.append((kind, hit / args.n, kind in COVERED))

    print(f"Personal-data filter {FILTER_VERSION}: recall on {args.n} generated cases per type (seed {args.seed})\n")
    print(f"{'type':16s}{'recall':>8s}   note")
    low = []
    for kind, rec, covered in summary:
        note = "" if covered else "NOT designed to be caught (free text): a known gap"
        print(f"{kind:16s}{rec:8.1%}   {note}")
        if covered and args.min_recall is not None and rec < args.min_recall:
            low.append(kind)
    cov = [r for k, r, c in summary if c]
    print(f"\nCovered types: mean recall {sum(cov) / len(cov):.1%}, lowest {min(cov):.1%}.")

    # false alarms on real synthetic questions
    pool = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("data/synthetic/synthetic_bitext_*.csv"))]) if glob.glob("data/synthetic/synthetic_bitext_*.csv") else pd.DataFrame()
    if len(pool):
        sample = pool["instruction"].dropna().sample(min(args.sample, len(pool)), random_state=args.seed).tolist()
        changed, fired = 0, {}
        for q in sample:
            r = redact(q)
            if r.findings:
                changed += 1
                for k in r.findings:
                    fired[k] = fired.get(k, 0) + 1
        print(f"\nOn {len(sample)} real synthetic questions: {changed} changed ({changed / len(sample):.1%}). Types that fired: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(fired.items())) or 'none'}.")
        print("These questions contain a card's last four digits, which the filter treats as an identifier on purpose.")
    wrong = [t for t in LOOKALIKES if redact(t).text != t]
    print(f"\nTricky look-alikes (amounts, dates, rates, references): {len(LOOKALIKES) - len(wrong)} of {len(LOOKALIKES)} left unchanged.")
    for t in wrong:
        print(f"  CHANGED: {t}  ->  {redact(t).text}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\nPer-case results written to {args.out}")
    if low:
        print(f"\nBelow the required recall of {args.min_recall:.0%}: {', '.join(low)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
