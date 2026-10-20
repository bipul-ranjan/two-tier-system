"""
Generates fresh, varied customer questions with a SEPARATE small language model, so a test run does not have to draw
from the same fixed pool of 16,000 questions again and again.

Why: the fixed pool repeats across runs (in the logs, about 28% of rows had the exact wording of another row). That flatters
the cache, lets the router meet questions it was effectively trained on, and hides how the system copes with wording it has
never seen. Freshly generated questions test all three honestly.

How it works, for each question to be generated:
 1. Draw the request types for the whole batch up front (a category and an intent, such as CARD / block_card), in proportion to the
    existing pool or with a chosen share of exception scenarios. Each is a slot, filled in turn with up to six tries of the SAME request
    type, so rejections cannot bend the mix toward types the model finds easy. A type it cannot produce leaves its slot empty and is reported.
 2. Build a prompt from that request type, a randomly chosen tone (casual, formal, stressed ...), a few made-up details
    (an amount, the last four digits of a card, a merchant, a city, a time) and three existing questions of the same type shown
    as examples of the topic, with an instruction not to copy their wording.
 3. Ask a general local model (default qwen2.5:7b, run by Ollama) for one new message. It is NOT one of the fine-tuned helpers,
    and it never sees any helper's or Claude's answers.
 4. Clean the text, and reject it if it is empty, too short or long, not a customer message, an exact duplicate of a question in
    the pool or in an earlier batch, or a near-duplicate (by meaning) of one already accepted.
 5. Optionally check the label: ask a model (by default the same one; a larger, separate judge can be named) which request type the message expresses, among the intents of that
    category plus "none of these", and keep the question only if it picks the intended one. This matters because the cache's third
    gate and every per-intent analysis rely on the label being right. By default the choice is among EVERY intent of the question's
    business unit (23 for payments, 27 for retail bank), so a message that belongs to another category (a transfer problem written
    up as a "contact customer service" question, say) is caught. label_scope="category" restores the older, easier check among the
    intents of the same category only; it passes almost any message for a category with one or two intents. The way the judge is asked
    can also be chosen (label_method: number, shuffled, name, two-step, examples, category; see src/label_judge.py and
    scripts/evaluate_label_judge.py, which measures each on questions with known labels).

Reproducibility: each question gets its own seed, derived from the batch seed and the attempt number, and the seed, model,
prompt version, tone and label-check result are saved with the question. The same seed, model and prompt give the same questions.

What it cannot tell you: generated questions are still synthetic. They test unfamiliar wording, not real customers. The label
check is a small model checking a small model, so spot-read the output before relying on it (scripts/generate_questions.py --show).

Pure functions take their model calls as arguments (generate_fn, classify_fn, embed_fn), so everything is testable without Ollama.
Used by scripts/generate_questions.py. The pipeline reads the result with --questions (see src/pipeline.py).
"""
import glob
import hashlib
import os
import random
import re

import numpy as np
import pandas as pd
import requests

from .config import OLLAMA_URL
from .label_judge import (METHODS, JudgeContext, build_label_prompt, judge, option_text, parse_label_answer,  # noqa: F401
                          parse_label_choice, words)

GENERATOR_MODEL = "qwen2.5:7b"
# How the judge is asked by default. Measured on 100 pool questions with known labels (scripts/evaluate_label_judge.py), qwen2.5:7b scored 84% with
# "examples" against 79% with "number" (the intervals overlap, so this is a lean and not a proof). Other methods remain selectable.
DEFAULT_LABEL_METHOD = "examples"
PROMPT_VERSION = "q-v1"
# Where a "fresh questions" run keeps its batch. Each such run REPLACES current_batch.csv (the previous batch is deleted), and only this
# file is ever deleted: batches you made by hand with scripts/generate_questions.py are never touched.
CURRENT_BATCH_PATH = "data/generated/current_batch.csv"
# One-way fingerprints (not the questions) of every question a fresh run has used, so a deleted question cannot come back.
SEEN_LEDGER_PATH = "data/generated/seen_questions.txt"
EXCEPTION_CATEGORIES = {"PAYMENT_EXCEPTION", "RETAIL_EXCEPTION"}

STYLES = [
    "short and casual, in lower case, perhaps with one small typo",
    "polite and formal",
    "stressed and urgent",
    "confused, not sure what to ask for",
    "detailed, with a short explanation of what happened",
    "very brief, a single short sentence",
    "friendly, starting with a greeting",
    "frustrated but civil",
]
CURRENCIES = [("GBP", "£"), ("USD", "$"), ("EUR", "€"), ("INR", "₹")]
MERCHANTS = ["Amazon", "Flipkart", "Uber", "Netflix", "Tesco", "Zomato", "Booking.com", "an online electronics shop", "a petrol station", "a hotel"]
CITIES = ["London", "Dubai", "Mumbai", "Toronto", "Singapore", "Berlin", "New York", "Pune", "Bangkok", "Istanbul"]
WHEN = ["yesterday", "this morning", "two days ago", "last Friday", "last week", "an hour ago", "on Monday"]
MONEY_CATEGORIES = {"CARD", "ATM", "TRANSFER", "FEES", "PAYMENT_EXCEPTION"}
REFUSAL_STARTS = ("here is", "here's", "sure", "certainly", "as an ai", "i cannot", "i can't assist", "of course")


class CutOff(str):
    """A model reply that stopped because it hit the length limit, so the message is unfinished. Otherwise ordinary text."""


def normalise(text: str) -> str:
    """Lower-case a question and drop punctuation and extra spaces, so trivially different copies compare equal."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", "", str(text).lower()).split())


def make_details(rng: random.Random, category: str) -> list:
    """Pick zero to two made-up details a customer might mention, as short phrases for the prompt."""
    options = []
    if category in MONEY_CATEGORIES or category == "LOAN":
        code, sym = rng.choice(CURRENCIES)
        low, high = (5000, 500000) if category == "LOAN" else ((20, 500) if category == "ATM" else (20, 5000))
        amount = f"{rng.randint(low, high):,}"
        options.append(rng.choice([f"an amount of {code} {amount}", f"an amount of {sym}{amount}"]))
    if category in {"CARD", "ATM", "PAYMENT_EXCEPTION"}:
        options.append(f"the card ending {rng.randint(0, 9999):04d}")
    if category in {"CARD", "PAYMENT_EXCEPTION", "FEES"}:
        options.append(f"the merchant {rng.choice(MERCHANTS)}")
    if category in {"CARD", "TRANSFER", "PAYMENT_EXCEPTION"}:
        options.append(f"the city {rng.choice(CITIES)}")
    options.append(f"it happened {rng.choice(WHEN)}")
    k = rng.choice([0, 1, 1, 2])
    return rng.sample(options, min(k, len(options)))


def build_prompt(category: str, intent: str, examples: list, style: str, details: list) -> str:
    """The instruction sent to the generator model for one question."""
    detail_line = ("Details you may weave in naturally: " + "; ".join(details) + ".\n") if details else ""
    shown = "\n".join(f"- {e}" for e in examples)
    return (
        "You write realistic messages that customers send to a bank's customer-service chat.\n"
        "Write ONE new message from a customer.\n\n"
        f"Request type: {words(category)}: {words(intent)}\n"
        f"Tone and style: {style}\n"
        f"{detail_line}"
        "Here are examples of this request type. Write something clearly different in wording and do not copy them:\n"
        f"{shown}\n\n"
        "Rules: write only the customer's message, in English, with no quotation marks and no explanation, "
        "at most two sentences, and never mention that you are an AI."
    )


def clean_message(text):
    """Tidy a model reply. Returns (message, "") if usable, or (None, reason) with reason one of
    empty, cut_off (the reply hit the length limit or ends mid-sentence on a comma, colon or dash), too_short, too_long,
    not_a_customer_message."""
    if isinstance(text, CutOff):
        return None, "cut_off"
    if not text or not str(text).strip():
        return None, "empty"
    t = str(text).strip().split("\n\n")[0].strip()
    t = re.sub(r"^(customer|message|user)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = " ".join(t.strip().strip("\"“”'`").strip().split())
    low = t.lower()
    if len(t) < 15:
        return None, "too_short"
    if len(t) > 300:
        return None, "too_long"
    if t.endswith((",", ";", ":", " -", "\u2013", "\u2014")):
        return None, "cut_off"
    if low.startswith(REFUSAL_STARTS) or "as an ai" in low or "customer message" in low:
        return None, "not_a_customer_message"
    return t, ""


def text_hash(text: str) -> str:
    """A one-way fingerprint of a question, ignoring case and punctuation. The question cannot be rebuilt from it."""
    return hashlib.sha1(normalise(text).encode("utf-8")).hexdigest()


def load_ledger(path: str = SEEN_LEDGER_PATH) -> set:
    """The fingerprints of questions used by earlier fresh runs (empty if there is no ledger yet)."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def append_ledger(hashes, path: str = SEEN_LEDGER_PATH) -> None:
    """Add fingerprints to the ledger."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(h + "\n" for h in hashes)


class Deduper:
    """Remembers questions already used and refuses repeats: exact copies always, and (if an embedding function is given)
    questions whose meaning is nearly the same as one already accepted."""

    def __init__(self, embed_fn=None, threshold: float = 0.92, blocked_hashes=None):
        self.embed_fn = embed_fn
        self.threshold = threshold
        self.exact = set()
        self.blocked = set(blocked_hashes or ())  # fingerprints of questions already used and since deleted
        self.vecs = None

    @staticmethod
    def _unit(v) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n == 0, 1, n)

    def add_exact(self, texts) -> None:
        """Remember texts so that exact copies of them are refused."""
        self.exact.update(normalise(t) for t in texts)

    def add_embedded(self, texts) -> None:
        """Remember texts for the near-duplicate check as well (needs an embedding function)."""
        texts = list(texts)
        self.add_exact(texts)
        if self.embed_fn is not None and texts:
            new = self._unit(self.embed_fn(texts))
            self.vecs = new if self.vecs is None else np.vstack([self.vecs, new])

    def check(self, text: str):
        """Return (ok, reason, nearest_similarity, vector). reason is duplicate or near_duplicate when refused."""
        if normalise(text) in self.exact or text_hash(text) in self.blocked:
            return False, "duplicate", 1.0, None
        if self.embed_fn is None:
            return True, "", float("nan"), None
        v = self._unit(self.embed_fn([text]))[0]
        nearest = float(np.max(self.vecs @ v)) if self.vecs is not None and len(self.vecs) else 0.0
        if nearest >= self.threshold:
            return False, "near_duplicate", nearest, v
        return True, "", nearest, v

    def register(self, text: str, vec) -> None:
        """Record an accepted question so later ones are compared against it."""
        self.exact.add(normalise(text))
        if self.embed_fn is not None and vec is not None:
            vec = self._unit(vec).reshape(1, -1)
            self.vecs = vec if self.vecs is None else np.vstack([self.vecs, vec])


def load_pool(synthetic_dir: str = "data/synthetic") -> pd.DataFrame:
    """Load the existing question pool (the CSVs in data/synthetic), used for examples and the exact-duplicate check."""
    files = sorted(glob.glob(f"{synthetic_dir}/synthetic_bitext_*.csv"))
    if not files:
        raise FileNotFoundError(f"No question files in {synthetic_dir}. The generator learns the request types from them.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_previous(generated_dir: str = "data/generated") -> list:
    """Questions from earlier generated batches, so a new batch does not repeat them."""
    out = []
    for f in sorted(glob.glob(f"{generated_dir}/*.csv")):
        try:
            out += pd.read_csv(f)["instruction"].dropna().tolist()
        except (KeyError, pd.errors.EmptyDataError):
            continue
    return out


def make_ollama_fn(model: str = GENERATOR_MODEL, temperature: float = 0.9, top_p: float = 0.95, num_predict: int = 160,
                   url: str = OLLAMA_URL, timeout: int = 120):
    """Build generate_fn(prompt, seed) that calls a local Ollama model. Raises a plain-language error if Ollama is not
    running or the model has not been downloaded."""
    def generate(prompt: str, seed: int) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
                   "options": {"temperature": temperature, "top_p": top_p, "num_predict": num_predict, "seed": int(seed)}}
        try:
            r = requests.post(url, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError("Cannot reach Ollama at " + url + ". Open the Ollama app (the llama icon in the menu bar) and try again.") from e
        if r.status_code == 404:
            raise RuntimeError(f"Ollama does not have the model '{model}'. Download it with:  ollama pull {model}")
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        return CutOff(text) if data.get("done_reason") == "length" else text
    return generate


def make_classify_fn(model: str = GENERATOR_MODEL, url: str = OLLAMA_URL, timeout: int = 120):
    """Build classify_fn(prompt) for the label check: the same local model, temperature 0 so the answer is steady."""
    gen = make_ollama_fn(model, temperature=0.0, top_p=1.0, num_predict=6, url=url, timeout=timeout)
    return lambda prompt: gen(prompt, 0)


def generate_questions(pool: pd.DataFrame, n: int, seed: int, generate_fn, classify_fn=None, embed_fn=None,
                       previous=(), model: str = GENERATOR_MODEL, unit: str = None, exception_share: float = None,
                       dedup_threshold: float = 0.92, dedup_pool_semantically: bool = False, per_slot_attempts: int = 6,
                       progress=None, blocked_hashes=None, label_scope: str = "unit", label_model: str = None,
                       label_method: str = DEFAULT_LABEL_METHOD):
    """Generate n new customer questions. Returns (DataFrame, stats).

    pool            the existing questions (columns instruction, category, intent), for request types and examples
    generate_fn     generate_fn(prompt, seed) -> text; make_ollama_fn(...) in real use
    classify_fn     classify_fn(prompt) -> text for the label check; None skips the check (label_check is then "skipped")
    embed_fn        embed_fn(list_of_texts) -> vectors, for the near-duplicate check; None checks exact copies only
    previous        questions from earlier batches to avoid repeating
    blocked_hashes  fingerprints (text_hash) of questions that were used and deleted, which must not come back
    label_scope     "unit" (default): the label check chooses among every intent of the business unit; "category": among the
                    intents of the same category only (easier, much less discriminating)
    unit            restrict to "payments" or "retail_bank"
    exception_share if given, the share of questions drawn from the exception categories; otherwise the pool's own mix
    label_model     the name of the model doing the label check, recorded with each question (default: the generator model)
    label_method    how the judge is asked (number, shuffled, name, two-step, examples or category; see src/label_judge.py).
                    label_scope="category" is the same as label_method="category"
    per_slot_attempts  how many times to try a slot's request type before giving up on it (default 6)
    The request types of all n slots are drawn first, so the batch keeps the intended mix. A slot that fails every try is left empty and
    named in stats["unfilled"], so the batch can be a few questions short of n. It is never filled with a different type instead."""
    if unit is not None:
        from .pipeline import unit_for_category  # imported only here: the pipeline pulls in heavier packages
        pool = pool[pool["category"].map(unit_for_category) == unit]
    if pool.empty:
        raise ValueError("No questions in the pool for that selection.")
    rng = random.Random(seed)
    groups = pool.groupby(["category", "intent"])["instruction"].apply(list)
    keys = list(groups.index)
    counts = [len(groups[k]) for k in keys]
    is_exc = [k[0] in EXCEPTION_CATEGORIES for k in keys]
    if exception_share is not None and any(is_exc) and not all(is_exc):
        e_keys = [k for k, e in zip(keys, is_exc) if e]
        n_keys = [k for k, e in zip(keys, is_exc) if not e]
    if label_scope not in ("unit", "category"):
        raise ValueError("label_scope must be 'unit' or 'category'")
    if label_method not in METHODS:
        raise ValueError(f"label_method must be one of {', '.join(METHODS)}")
    method = "category" if label_scope == "category" else label_method
    judge_ctx = JudgeContext(set(zip(pool["category"], pool["intent"])), examples={k: list(v) for k, v in groups.items()})

    dd = Deduper(embed_fn, dedup_threshold, blocked_hashes)
    dd.add_exact(pool["instruction"])
    if embed_fn is not None:
        dd.add_embedded(list(previous) + (pool["instruction"].tolist() if dedup_pool_semantically else []))
    else:
        dd.add_exact(previous)

    # Draw every slot's request type before any question is generated, so rejections cannot change the batch's mix.
    slots = []
    for _ in range(n):
        if exception_share is not None and any(is_exc) and not all(is_exc):
            slots.append(rng.choice(e_keys if rng.random() < exception_share else n_keys))
        else:
            slots.append(rng.choices(keys, weights=counts, k=1)[0])

    rows = []
    stats = {"attempts": 0, "accepted": 0, "rejected": {}, "label_checked": 0, "label_failed": 0, "slots": n, "unfilled": {},
             "confusions": {}, "by_intent": {}, "rejected_rows": []}

    def reject(reason, text=None, judged=None):
        """Count a rejection and keep the rejected text, so a person can read what was thrown away and why."""
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
        stats["rejected_rows"].append({"instruction": "" if text is None else str(text)[:400], "category": category, "intent": intent,
                                       "reason": reason, "judged_as": judged or ""})

    for key in slots:
        category, intent = key
        for _try in range(per_slot_attempts):
            stats["attempts"] += 1
            style = rng.choice(STYLES)
            details = make_details(rng, category)
            examples = rng.sample(groups[key], min(3, len(groups[key])))
            qseed = seed * 100003 + stats["attempts"]
            raw = generate_fn(build_prompt(category, intent, examples, style, details), qseed)
            text, reason = clean_message(raw)
            if text is None:
                reject(reason, raw)
                continue
            ok, reason, nearest, vec = dd.check(text)
            if not ok:
                reject(reason, text)
                continue
            label = "skipped"
            if classify_fn is not None:
                stats["label_checked"] += 1
                kind, chosen, _position = judge(method, classify_fn, text, (category, intent), judge_ctx, random.Random(qseed))
                label = "pass" if chosen == (category, intent) else "fail"
                record = stats["by_intent"].setdefault(f"{category} / {intent}", {"checked": 0, "passed": 0})
                record["checked"] += 1
                if label == "fail":
                    judged = f"{chosen[0]} / {chosen[1]}" if kind == "option" else ("none of these" if kind == "none" else "no usable answer")
                    mixup = f"{category} / {intent} -> {judged}"
                    stats["confusions"][mixup] = stats["confusions"].get(mixup, 0) + 1
                    stats["label_failed"] += 1
                    reject("label_mismatch", text, judged)
                    continue
                record["passed"] += 1
            dd.register(text, vec)
            rows.append({"instruction": text, "category": category, "intent": intent, "is_exception": category in EXCEPTION_CATEGORIES,
                         "generator_model": model, "prompt_version": PROMPT_VERSION, "seed": qseed, "style": style,
                         "label_scope": label_scope if classify_fn is not None else "skipped",
                         "label_method": method if classify_fn is not None else "skipped",
                         "label_model": (label_model or model) if classify_fn is not None else "skipped", "label_check": label,
                         "nearest_similarity": None if nearest != nearest else round(nearest, 4)})
            stats["accepted"] += 1
            if progress:
                progress(stats["accepted"], n, stats["attempts"])
            break
        else:
            name = f"{category} / {intent}"
            stats["unfilled"][name] = stats["unfilled"].get(name, 0) + 1
    cols = ["instruction", "category", "intent", "is_exception", "generator_model", "prompt_version", "seed", "style",
            "label_scope", "label_method", "label_model", "label_check", "nearest_similarity"]
    return pd.DataFrame(rows, columns=cols), stats


def confusion_lines(stats: dict, top: int = 8) -> list:
    """The most common label mix-ups as readable lines, for example 'TRANSFER / make_transfer was judged as TRANSFER / cancel_transfer x9'."""
    ranked = sorted(stats.get("confusions", {}).items(), key=lambda kv: -kv[1])[:top]
    return [f"{k.replace(' -> ', ' was judged as ')} x{v}" for k, v in ranked]


def hardest_lines(stats: dict, top: int = 5, min_checked: int = 3) -> list:
    """The request types with the lowest label-check pass rate (only those checked at least min_checked times)."""
    rows = [(k, v["passed"], v["checked"]) for k, v in stats.get("by_intent", {}).items() if v["checked"] >= min_checked]
    rows.sort(key=lambda r: (r[1] / r[2], -r[2]))
    return [f"{k}: {p} of {c} passed ({p / c:.0%})" for k, p, c in rows[:top]]


def rejected_to_frame(stats: dict) -> pd.DataFrame:
    """Every rejected candidate with its request type, the reason and (for label mismatches) what the judge chose instead."""
    return pd.DataFrame(stats.get("rejected_rows", []), columns=["instruction", "category", "intent", "reason", "judged_as"])


def mix_summary(df: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Compare a batch's mix with the pool's: the exception share of each, and the categories the pool has that the batch lacks."""
    return {"pool_exception_share": float(pool["category"].isin(EXCEPTION_CATEGORIES).mean()),
            "batch_exception_share": float(df["is_exception"].mean()) if len(df) else 0.0,
            "missing_categories": sorted(set(pool["category"]) - set(df["category"]))}


def check_model_available(model: str, url: str = OLLAMA_URL) -> None:
    """Raise a plain-language error if Ollama is not running or the model has not been downloaded."""
    tags_url = url.replace("/api/generate", "/api/tags")
    try:
        names = [m["name"] for m in requests.get(tags_url, timeout=10).json().get("models", [])]
    except requests.exceptions.RequestException as e:
        raise RuntimeError("Cannot reach Ollama. Open the Ollama app (the llama icon in the menu bar) and try again.") from e
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(f"Ollama does not have '{model}'. Download it with:  ollama pull {model}\n"
                           f"Models you have: {', '.join(names) or 'none'}")


def default_embed_fn():
    """The cache's embedding model for the near-duplicate check, or None (with a note) if it is not available."""
    try:
        from .semantic_cache import SemanticCacheIndex
        return SemanticCacheIndex._default_embed_fn()
    except Exception as e:  # a missing package, or no internet for the first download
        print(f"Note: the near-duplicate check is off ({type(e).__name__}); only exact copies will be refused.")
        return None


_DEFAULT = object()


def make_fresh_batch(n: int, seed=None, model: str = GENERATOR_MODEL, out_path: str = CURRENT_BATCH_PATH,
                     ledger_path: str = SEEN_LEDGER_PATH, pool: pd.DataFrame = None, generate_fn=None, classify_fn=_DEFAULT,
                     embed_fn=_DEFAULT, temperature: float = 0.9, progress=None, label_scope: str = "unit", label_model: str = None,
                     label_method: str = DEFAULT_LABEL_METHOD):
    """Make a batch of n fresh questions, REPLACE the previous batch with it, and return (DataFrame, stats, seed used, replaced).

    This is what `python -m src.pipeline --fresh` calls. The steps:
      1. Generate the new batch in memory. If this fails or yields nothing, the previous batch is left exactly as it was.
      2. Write it next to out_path and swap it in, which deletes the previous batch (only out_path is ever replaced).
      3. Record one-way fingerprints of the new questions in the ledger, so a later run cannot repeat them even though the
         questions themselves are gone.
    With seed=None a random seed is chosen and the ledger is used (every run differs). With an explicit seed the ledger is neither
    read nor written, so the same seed, model and prompt repeat the same batch."""
    pool = load_pool() if pool is None else pool
    use_ledger = seed is None
    seed_used = random.SystemRandom().randrange(1, 10**6) if seed is None else int(seed)
    blocked = load_ledger(ledger_path) if use_ledger else set()
    generate_fn = generate_fn or make_ollama_fn(model, temperature=temperature)
    classify_fn = make_classify_fn(label_model or model) if classify_fn is _DEFAULT else classify_fn
    embed_fn = default_embed_fn() if embed_fn is _DEFAULT else embed_fn
    df, stats = generate_questions(pool, n, seed_used, generate_fn, classify_fn=classify_fn, embed_fn=embed_fn, model=model,
                                   blocked_hashes=blocked, progress=progress, label_scope=label_scope, label_model=label_model,
                                   label_method=label_method)
    if df.empty:
        raise RuntimeError("No questions were accepted, so the previous batch was kept. Check the generator with:  ollama run " + model)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    replaced = os.path.exists(out_path)
    tmp = out_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    if use_ledger:
        append_ledger([text_hash(t) for t in df["instruction"]], ledger_path)
    return df, stats, seed_used, replaced
