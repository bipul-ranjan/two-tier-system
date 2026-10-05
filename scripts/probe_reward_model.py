"""
Zero-shot probe: does a pretrained reward model, with NO training on your data, rank Tier 1
answers the way Claude-as-judge does?

A reward model reads (query, answer) and returns one score. This script scores a sample of your
(query, Tier 1 answer) pairs and compares that score with quality_draft_overall (Claude's
judgement of the same Tier 1 answer), side by side with two baselines on the SAME rows:
Tier 1 confidence (what your router uses today) and a small text model trained on your history
with whole intents held out (the probe from the design discussion). If the pretrained model
already beats confidence zero-shot, fine-tuning it is worth the effort; if not, it isn't.

Defaults: Skywork-Reward-V2-Qwen3-0.6B, 500 rows split evenly across business units. It runs on
CPU one row at a time (a 0.6B model needs about 2.4 GB of RAM in fp32 -- close browsers first,
or pass --bf16 to halve that). The first run downloads the weights (about 1.2 GB). Scores are
saved as it goes, so Ctrl+C is safe: re-run the same command and it carries on where it stopped.

Run from the project root, inside the venv:
    python scripts/probe_reward_model.py                  # 500 rows
    python scripts/probe_reward_model.py --n 200          # a quicker look
    python scripts/probe_reward_model.py --bf16           # half the memory
    python scripts/probe_reward_model.py --model Skywork/Skywork-Reward-V2-Qwen3-1.7B
"""
import argparse
import hashlib
import math
import os
import sys
import time

import numpy as np
import pandas as pd

DEFAULT_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
DEFAULT_HISTORY = "results/logs/results_history.csv"
DEFAULT_OUT_DIR = "results/analysis"
LOW_QUALITY = 3.5             # "a below-3.5 Tier 1 answer" is what the AUC check tries to spot
ESCALATION_RATES = (0.2, 0.3, 0.4)
MAX_ANSWER_CHARS = 4000       # truncate the answer TEXT, never the token stream: the model reads its last token
FALLBACK_CLAUDE_QUALITY = 4.38


def row_key(query: str, answer: str) -> str:
    return hashlib.md5(f"{query}\x1f{answer}".encode("utf-8")).hexdigest()


def load_scorer(model_name: str, bf16: bool = False):
    """Returns score(prompt, response) -> float, following the model's published usage:
    chat template with no system prompt, duplicate BOS removed, num_labels=1, logits[0][0]."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if bf16 else torch.float32
    print(f"Loading {model_name} ({'bf16' if bf16 else 'fp32'}, {device}, transformers {transformers.__version__}) "
          f"-- the first run downloads the weights...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype, num_labels=1)
        except TypeError:                     # transformers older than the dtype= rename
            model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype, num_labels=1)
    except (ValueError, KeyError) as e:
        raise SystemExit(f"Could not load {model_name} ({e}). Qwen3 models need a recent transformers: pip install -U transformers")
    model.to(device).eval()

    def score(prompt: str, response: str) -> float:
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response[:MAX_ANSWER_CHARS]}]
        text = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            return model(**inputs).logits[0][0].float().item()

    return score


def load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run this from the project root, or pass --history")
    df = pd.read_csv(path)
    needed = ["query", "tier1_answer", "tier1_confidence_avg", "business_unit", "intent", "quality_draft_overall"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        hint = "  Run: python scripts/backfill_quality_scores.py --draft" if "quality_draft_overall" in missing else ""
        raise SystemExit(f"{path} is missing columns {missing}.{hint}")
    df = df.dropna(subset=["query", "tier1_answer", "tier1_confidence_avg", "quality_draft_overall"]).reset_index(drop=True)
    df["key"] = [row_key(q, a) for q, a in zip(df["query"], df["tier1_answer"])]
    # the same query answered identically in two runs is one pair to a reward model: count it once
    return df.drop_duplicates("key").reset_index(drop=True)


def claude_quality_of(df: pd.DataFrame) -> float:
    """Average quality of a FRESH Claude answer (decision ESCALATE, not a cache hit, current prompt)."""
    if not {"decision", "quality_overall"} <= set(df.columns):
        return FALLBACK_CLAUDE_QUALITY
    fresh = df["decision"].eq("ESCALATE")
    if "cache_hit" in df.columns:
        fresh &= ~df["cache_hit"].astype(str).str.lower().eq("true")
    if "tier2_prompt_version" in df.columns and (fresh & df["tier2_prompt_version"].eq("v2-persona")).any():
        fresh &= df["tier2_prompt_version"].eq("v2-persona")
    q = df.loc[fresh, "quality_overall"].dropna()
    return float(q.mean()) if len(q) >= 20 else FALLBACK_CLAUDE_QUALITY


def pick_rows(df: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """A fixed, repeatable sample split evenly across business units: rows are ordered by their
    hash, so the same rows are chosen on every run (which is what lets a re-run resume)."""
    units = sorted(df["business_unit"].unique())
    per_unit = max(1, n_total // len(units))
    parts = [df[df["business_unit"] == u].sort_values("key").head(per_unit) for u in units]
    return pd.concat(parts, ignore_index=True)


def read_cache(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return pd.DataFrame(columns=["key", "rm_score"])


def score_missing(sample: pd.DataFrame, scorer, cache_path: str) -> bool:
    """Score the sample rows not already in the cache, saving as it goes. False if interrupted."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    done = set(read_cache(cache_path)["key"])
    todo = sample[~sample["key"].isin(done)]
    print(f"{len(sample)} rows in the sample, {len(sample) - len(todo)} already scored, {len(todo)} to score.")
    if todo.empty:
        return True
    pending, t0 = [], time.time()

    def flush():
        if pending:
            pd.DataFrame(pending).to_csv(cache_path, mode="a", header=not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0), index=False)
            pending.clear()

    try:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            pending.append({"key": r.key, "rm_score": scorer(r.query, r.tier1_answer)})
            if i % 10 == 0:
                flush()
                rate = (time.time() - t0) / i
                print(f"  {i}/{len(todo)} scored  ({rate:.1f} s/row, about {rate * (len(todo) - i) / 60:.0f} min left)")
    except KeyboardInterrupt:
        flush()
        print("\nInterrupted. Scores so far are saved -- run the same command again to continue.")
        return False
    flush()
    return True


def text_probe_oof(df: pd.DataFrame) -> np.ndarray:
    """A small text model (TF-IDF + ridge) predicting the Tier 1 quality, trained on your history
    with WHOLE INTENTS held out so near-duplicate queries can't leak -- the same baseline as in the
    design discussion. Out-of-fold, so each row is scored by a model that never saw its intent."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    y = df["quality_draft_overall"].values
    num = np.c_[df["tier1_confidence_avg"], np.log1p(df["tier1_answer"].str.len()), np.log1p(df["query"].str.len()),
                (df["business_unit"] == sorted(df["business_unit"].unique())[-1]).astype(float)]
    num = (num - num.mean(0)) / (num.std(0) + 1e-9)
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, y, groups=df["intent"].fillna("none")):
        vq = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=2000, sublinear_tf=True).fit(df.loc[tr, "query"])
        va = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=2000, sublinear_tf=True).fit(df.loc[tr, "tier1_answer"])
        X = lambda ix: sp.hstack([vq.transform(df.loc[ix, "query"]), va.transform(df.loc[ix, "tier1_answer"]), sp.csr_matrix(num[ix])]).tocsr()
        oof[te] = Ridge(alpha=3.0).fit(X(tr), y[tr]).predict(X(te))
    return oof


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)   # never trust the input type: object arrays break numpy on some versions
    r = spearmanr(a, b)[0]
    n = len(a)
    if n > 3 and abs(r) < 1:
        z, se = math.atanh(r), 1 / math.sqrt(n - 3)
        return r, math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    return r, float("nan"), float("nan")


def delivered_quality(scored: pd.DataFrame, score_col: str, frac: float, claude_q: float) -> float:
    """Average quality delivered if the `frac` lowest-scoring rows in each business unit go to Claude:
    kept rows deliver their real Tier 1 quality, escalated rows deliver Claude's average."""
    out = []
    for _, g in scored.groupby("business_unit"):
        k = int(round(frac * len(g)))
        q = g["quality_draft_overall"].values.astype(float).copy()
        q[np.argsort(g[score_col].values.astype(float))[:k]] = claude_q
        out.append(q)
    return float(np.concatenate(out).mean())


def report(scored: pd.DataFrame, claude_q: float):
    from sklearn.metrics import roc_auc_score
    methods = [("confidence", "tier1_confidence_avg"), ("reward model", "rm_score"), ("text probe", "text_probe")]
    units = sorted(scored["business_unit"].unique())
    print("\n=== Does a pretrained reward model predict Claude's quality score for the Tier 1 answer? ===")
    counts = ", ".join(f"{u} {int((scored['business_unit'] == u).sum())}" for u in units)
    print(f"rows scored: {len(scored)} ({counts})")
    print("The reward model got NO training on your data. 'text probe' is trained on your history with whole intents held out.")

    print("\nSpearman correlation with Claude's quality score (higher is better; 95% interval in brackets):")
    print(f"  {'':14s}" + "".join(f"{name:>26s}" for name in ["all (pooled)"] + units) + f"{'mean of units':>16s}")
    results = {}
    for label, col in methods:
        cells = []
        for name in ["all"] + units:
            g = scored if name == "all" else scored[scored["business_unit"] == name]
            r, lo, hi = spearman(g[col], g["quality_draft_overall"])
            results[(label, name)] = r
            cells.append(f"{r:+.3f} [{lo:+.2f},{hi:+.2f}]")
        results[(label, "mean of units")] = float(np.mean([results[(label, u)] for u in units]))
        print(f"  {label:14s}" + "".join(f"{c:>26s}" for c in cells) + f"{results[(label, 'mean of units')]:>+16.3f}")
    print("  Note: 'pooled' also rewards a signal for knowing which unit is easier (e.g. a reward model that scores one unit's\n"
          "  answers higher overall). Routing is decided WITHIN each unit, so judge by the unit columns and 'mean of units'.")

    bad = (scored["quality_draft_overall"] < LOW_QUALITY).values
    if bad.any() and not bad.all():
        aucs = {label: roc_auc_score(bad, -scored[col]) for label, col in methods}
        print(f"\nAUC for spotting a Tier 1 answer scoring below {LOW_QUALITY} (0.5 = chance): "
              + " | ".join(f"{label} {a:.3f}" for label, a in aucs.items()))

    print(f"\nAverage quality delivered at the same escalation rate (kept rows = real Tier 1 quality, escalated rows = Claude's {claude_q:.2f}):")
    print(f"  {'escalate':>9} | {'random':>7} | {'confidence':>10} | {'reward model':>12} | {'text probe':>10} | {'oracle':>7}")
    rng = np.random.default_rng(0)
    for frac in ESCALATION_RATES:
        rnd = []
        for _ in range(20):
            tmp = scored.assign(_r=rng.random(len(scored)))
            rnd.append(delivered_quality(tmp, "_r", frac, claude_q))
        row = [np.mean(rnd)] + [delivered_quality(scored, col, frac, claude_q) for _, col in methods]
        row.append(delivered_quality(scored.assign(_o=scored["quality_draft_overall"]), "_o", frac, claude_q))
        print(f"  {frac:>8.0%} | " + " | ".join(f"{v:>{w}.3f}" for v, w in zip(row, (7, 10, 12, 10, 7))))

    m = "mean of units"
    rm, conf, probe = results[("reward model", m)], results[("confidence", m)], results[("text probe", m)]
    print(f"\nWithin units, the reward model with no training is {'above' if rm > conf else 'not above'} confidence "
          f"({rm:+.3f} vs {conf:+.3f}) and {'above' if rm > probe else 'below'} the trained text probe ({rm:+.3f} vs {probe:+.3f}).")
    print("With this many rows, differences smaller than about 0.1 are within noise.")


def main(argv=None, scorer_factory=load_scorer) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default {DEFAULT_MODEL})")
    ap.add_argument("--n", type=int, default=500, help="total rows to score, split evenly across business units (default 500)")
    ap.add_argument("--history", default=DEFAULT_HISTORY, help=f"results history CSV (default {DEFAULT_HISTORY})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"where scores are saved (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--bf16", action="store_true", help="load in bfloat16: half the memory, slower on some CPUs")
    args = ap.parse_args(argv)

    df = load_history(args.history)
    sample = pick_rows(df, args.n)
    cache_path = os.path.join(args.out_dir, f"reward_model_zero_shot_{args.model.split('/')[-1]}.csv")

    if not set(sample["key"]) <= set(read_cache(cache_path)["key"]):
        scorer = scorer_factory(args.model, args.bf16)
        if not score_missing(sample, scorer, cache_path):
            return 1
    else:
        print(f"All {len(sample)} sample rows are already scored in {cache_path}.")

    cache = read_cache(cache_path).drop_duplicates("key")
    scored = df.merge(cache, on="key", how="inner").drop_duplicates("key")
    scored = scored[scored["key"].isin(sample["key"])].reset_index(drop=True)
    if len(scored) < 30:
        print(f"Only {len(scored)} rows scored -- too few to read anything from. Run again to score more.")
        return 1
    probe = text_probe_oof(df)
    scored["text_probe"] = pd.Series(probe, index=df["key"]).reindex(scored["key"]).values
    report(scored, claude_quality_of(pd.read_csv(args.history)))
    print(f"\nScores are saved in {cache_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
