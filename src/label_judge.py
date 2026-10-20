"""
Asks a model which request type a customer question expresses, in six different ways, and measures how often each way is right.

Why this exists: the question generator keeps a generated question only if a model, the "judge", picks the intended request type
from a list of 23 or 27. In a real run the judge rejected most good-looking questions, and it picked unrelated types (a card question
"judged as" an international transfer hold). Before trusting the generator, the judge itself must be measured. Your pool holds 16,000
questions whose labels are known to be right, so the judge can be tested on them directly, with no generator involved.

The six methods (the same code is used to measure the judge and, with --label-method, to run the generator's label check):

 number     one question: the options are numbered "category: intent" lines and the model answers with a number (the original check)
 shuffled   the same, with the options in a random order, to test whether the position of the right answer matters
 name       the model must reply with the option's text instead of a number
 two-step   first the category (5 or 6 choices), then the intent within it (at most 12): two small questions instead of one big one
 examples   as "number", but every option shows one real example question of that type, so the model sees what each label means
 category   the easier old check: the model already knows the category and picks only among that category's intents. It cannot catch a
            question that belongs to another category, so it is a reference point and not a recommendation

Accuracy here means the judge named exactly the right category and intent. "None of these" and an unreadable reply count as wrong,
and the unreadable replies are reported separately. Chance is about 4%, because there are 23 to 27 options.

Everything takes the model call as an argument (classify_fn(prompt) -> text), so it is tested without Ollama.
Used by scripts/evaluate_label_judge.py and by src/question_generator.py.
"""
import math
import random
import re

import pandas as pd

METHODS = ("number", "shuffled", "name", "two-step", "examples", "category")


def words(snake: str) -> str:
    """Turn a label such as card_declined_while_travelling into plain words."""
    return snake.replace("_", " ").lower()


def _norm(text: str) -> str:
    """Lower-case, drop punctuation and extra spaces."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


def option_text(option) -> str:
    """How an option is shown to the model: 'category: intent words' for a (category, intent) pair, or the intent words alone."""
    if isinstance(option, tuple):
        return f"{words(option[0])}: {words(option[1])}"
    return words(option)


def build_label_prompt(message: str, options: list, examples: dict = None) -> str:
    """Ask which request type a message expresses, with a final 'none of these' option. Each option is an intent name or a
    (category, intent) pair; the pair form shows the category, which tells apart similar intents in different categories.
    If examples maps an option to a sample question, that question is shown beside the option."""
    lines = []
    for i, o in enumerate(options, start=1):
        line = f"{i}. {option_text(o)}"
        example = (examples or {}).get(o)
        if example:
            line += f' (for example: "{example}")'
        lines.append(line)
    lines.append(f"{len(options) + 1}. none of these")
    return ("Which request type does this customer message express? Choose the single best match.\n"
            f'Message: "{message}"\nOptions:\n' + "\n".join(lines) + "\nAnswer with the number only.")


def build_name_prompt(message: str, options: list) -> str:
    """As build_label_prompt, but the model replies with the option's own text, not a number."""
    lines = [f"- {option_text(o)}" for o in options] + ["- none of these"]
    return ("Which request type does this customer message express? Choose the single best match.\n"
            f'Message: "{message}"\nOptions:\n' + "\n".join(lines) + "\nAnswer with the exact option text only.")


def build_category_prompt(message: str, categories: list) -> str:
    """First step of the two-step method: which category does the message belong to?"""
    lines = [f"{i}. {words(c)}" for i, c in enumerate(categories, start=1)] + [f"{len(categories) + 1}. none of these"]
    return ("Which category does this customer message belong to? Choose the single best match.\n"
            f'Message: "{message}"\nCategories:\n' + "\n".join(lines) + "\nAnswer with the number only.")


def parse_label_choice(text, options: list):
    """Read the number in a model's reply. Returns ("option", the chosen option), ("none", None) for 'none of these', or
    ("unparsed", None) if there is no usable answer."""
    m = re.search(r"\d+", str(text or ""))
    if not m:
        return "unparsed", None
    k = int(m.group())
    if 1 <= k <= len(options):
        return "option", options[k - 1]
    return ("none", None) if k == len(options) + 1 else ("unparsed", None)


def parse_label_answer(text, options: list):
    """The chosen intent, or None for 'none of these' or no usable answer."""
    kind, item = parse_label_choice(text, options)
    return item if kind == "option" else None


def match_name(reply, options: list):
    """Read a reply that names an option. An exact match wins; otherwise the longest option text found inside the reply
    (so 'activate card international usage' is not mistaken for 'activate card'). Returns ("option", option), ("none", None)
    or ("unparsed", None)."""
    r = _norm(reply)
    if not r:
        return "unparsed", None
    texts = [(_norm(option_text(o)), o) for o in options]
    for t, o in texts:
        if t == r:
            return "option", o
    hits = [(len(t), o) for t, o in texts if t and t in r]
    if hits:
        return "option", max(hits, key=lambda h: h[0])[1]
    return ("none", None) if "none of these" in r else ("unparsed", None)


class JudgeContext:
    """What the judging methods need to know about the request types: every (category, intent) pair, which business unit a category
    belongs to, and (for the examples method) some real questions of each type."""

    def __init__(self, pairs, unit_for_category=None, examples: dict = None):
        self.pairs = sorted(set(pairs))
        self._unit_fn = unit_for_category
        self.examples = examples or {}
        self._by_unit, self._by_category = {}, {}

    def unit(self, category: str) -> str:
        if self._unit_fn is None:
            from .pipeline import unit_for_category  # imported only when needed: the pipeline pulls in heavier packages
            self._unit_fn = unit_for_category
        return self._unit_fn(category)

    def options_for_unit(self, category: str) -> list:
        u = self.unit(category)
        if u not in self._by_unit:
            self._by_unit[u] = [p for p in self.pairs if self.unit(p[0]) == u]
        return self._by_unit[u]

    def options_for_category(self, category: str) -> list:
        if category not in self._by_category:
            self._by_category[category] = [p for p in self.pairs if p[0] == category]
        return self._by_category[category]

    def categories_of_unit(self, category: str) -> list:
        return sorted({p[0] for p in self.options_for_unit(category)})

    def example_map(self, options: list, rng: random.Random, exclude: str) -> dict:
        """One example question per option, never the question being judged."""
        out = {}
        for o in options:
            pool = [q for q in self.examples.get(o, []) if _norm(q) != _norm(exclude)]
            if pool:
                out[o] = rng.choice(pool)[:110].rstrip()
        return out


def judge(method: str, classify_fn, message: str, truth: tuple, ctx: JudgeContext, rng: random.Random = None):
    """Judge one message. truth is its intended (category, intent): the unit's options are taken from it, as the generator does, and the
    category method also uses its category. Returns (kind, prediction, position): kind is "option", "none" or "unparsed"; prediction is the
    chosen (category, intent) or None; position is where the truth stood in the list the model saw (1 is first), or None if it was not there."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    rng = rng or random.Random(0)
    cat = truth[0]
    options = list(ctx.options_for_category(cat) if method == "category" else ctx.options_for_unit(cat))
    if method == "shuffled":
        rng.shuffle(options)
    if method == "name":
        kind, pred = match_name(classify_fn(build_name_prompt(message, options)), options)
    elif method == "two-step":
        cats = ctx.categories_of_unit(cat)
        kind, chosen = parse_label_choice(classify_fn(build_category_prompt(message, cats)), cats)
        if kind != "option":
            return kind, None, None
        options = ctx.options_for_category(chosen)
        kind, pred = parse_label_choice(classify_fn(build_label_prompt(message, options)), options)
    else:
        examples = ctx.example_map(options, rng, message) if method == "examples" else None
        kind, pred = parse_label_choice(classify_fn(build_label_prompt(message, options, examples)), options)
    return kind, pred, (options.index(truth) + 1 if truth in options else None)


# ---------- measuring the judge ----------

def wilson(k: int, n: int, z: float = 1.96):
    """A 95% confidence interval for a proportion k of n (the Wilson score interval), as (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_cases(pool: pd.DataFrame, n: int, seed: int, even: bool = True) -> list:
    """Questions with known labels to test the judge on. even=True takes the same number from every request type (so rare types are
    tested as often as common ones; with fewer questions than types, a random set of types); even=False draws from the pool as it is."""
    rng = random.Random(seed)
    groups = pool.groupby(["category", "intent"])["instruction"].apply(list)
    keys = list(groups.index)
    if not even:
        rows = pool.sample(n=min(n, len(pool)), random_state=seed)
        return [{"message": r.instruction, "category": r.category, "intent": r.intent} for r in rows.itertuples()]
    if n < len(keys):
        picks = [(k, 1) for k in rng.sample(keys, n)]
    else:
        per, extra = divmod(n, len(keys))
        bonus = set(rng.sample(keys, extra))
        picks = [(k, per + (1 if k in bonus else 0)) for k in keys]
    cases = []
    for k, count in picks:
        for q in rng.sample(groups[k], min(count, len(groups[k]))):
            cases.append({"message": q, "category": k[0], "intent": k[1]})
    rng.shuffle(cases)
    return cases


def run_evaluation(cases: list, methods, classify_fn, ctx: JudgeContext, seed: int = 1, progress=None) -> pd.DataFrame:
    """Judge every case with every method. One row per case and method."""
    rows = []
    for mi, method in enumerate(methods):
        for ci, c in enumerate(cases):
            calls = {"n": 0}

            def counted(prompt, _calls=calls):
                _calls["n"] += 1
                return classify_fn(prompt)
            truth = (c["category"], c["intent"])
            kind, pred, position = judge(method, counted, c["message"], truth, ctx, random.Random(seed * 1000003 + ci * 31 + mi))
            n_options = len(ctx.options_for_category(truth[0]) if method == "category" else ctx.options_for_unit(truth[0]))
            rows.append({"method": method, "message": c["message"], "true_category": truth[0], "true_intent": truth[1], "kind": kind,
                         "pred_category": pred[0] if pred else "", "pred_intent": pred[1] if pred else "", "correct": pred == truth,
                         "position": position, "n_options": n_options, "calls": calls["n"]})
            if progress:
                progress(method, ci + 1, len(cases))
    return pd.DataFrame(rows)


def summarize_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of each method with its 95% interval, the share of unreadable replies and the model calls per question."""
    out = []
    for method, g in df.groupby("method", sort=False):
        k, n = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"method": method, "n": n, "correct": k, "accuracy": k / n, "low": lo, "high": hi,
                    "unreadable": float((g["kind"] == "unparsed").mean()), "calls_per_question": float(g["calls"].mean())})
    return pd.DataFrame(out)


def position_table(df: pd.DataFrame, method: str) -> dict:
    """Accuracy by where the right answer stood in the list: first, middle and last third. A large drop toward the end means the judge
    cannot cope with long lists."""
    g = df[(df["method"] == method) & df["position"].notna()]
    out = {}
    for name, lo, hi in (("first third", 0, 1 / 3), ("middle third", 1 / 3, 2 / 3), ("last third", 2 / 3, 1.0001)):
        frac = (g["position"] - 1) / g["n_options"]
        part = g[(frac >= lo) & (frac < hi)]
        out[name] = (float(part["correct"].mean()) if len(part) else None, len(part))
    return out


def tests_per_type(df: pd.DataFrame, method: str) -> int:
    """The largest number of times any one request type was tested under a method. Small values mean per-type results are rough."""
    g = df[df["method"] == method].groupby(["true_category", "true_intent"]).size()
    return int(g.max()) if len(g) else 0


def confusion_table(df: pd.DataFrame, method: str, top: int = 8) -> list:
    """The most common mix-ups under one method, as (intended, what the judge chose, count). The judge's choice is a request type, or
    'none of these' or 'no usable answer'."""
    w = df[(df["method"] == method) & (~df["correct"])]
    chosen = w.apply(lambda r: f"{r['pred_category']} / {r['pred_intent']}" if r["kind"] == "option"
                     else ("none of these" if r["kind"] == "none" else "no usable answer"), axis=1)
    pairs = (w["true_category"] + " / " + w["true_intent"] + "\t" + chosen).value_counts().head(top)
    return [(k.split("\t")[0], k.split("\t")[1], int(v)) for k, v in pairs.items()]


def hardest_table(df: pd.DataFrame, method: str, top: int = 6, min_n: int = 1, only_errors: bool = True) -> list:
    """The request types the judge gets right least often under one method, as (name, correct, tested) tuples. By default only
    types with at least one error are listed, so a judge that is right every time gives an empty list. min_n is 1, so a type tested
    only once and judged wrong is still listed (a higher value would hide every error when each type was tested once)."""
    g = df[df["method"] == method].groupby(["true_category", "true_intent"])["correct"].agg(["sum", "count"])
    g = g[g["count"] >= min_n]
    if only_errors:
        g = g[g["sum"] < g["count"]]
    g = g.assign(rate=g["sum"] / g["count"]).sort_values(["rate", "count"], ascending=[True, False]).head(top)
    return [(f"{c} / {i}", int(r["sum"]), int(r["count"])) for (c, i), r in g.iterrows()]
