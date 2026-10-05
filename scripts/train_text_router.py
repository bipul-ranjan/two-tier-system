"""
Train the learned router that decides which Tier 1 answers to escalate (src/learned_router.py).

It learns from your own history: for every logged question it has Tier 1's answer and Claude's mark
for that answer (the quality_draft_overall column, filled for every row -- local rows copy their
final-answer marks, escalated and cache rows have their discarded draft marked). It learns to predict
that mark from the question, the answer, Tier 1's confidence and the business unit.

What this script does, in order:
  1. Evaluates the router honestly: five folds that each hold out WHOLE INTENTS, so every score comes
     from a model that never saw that question type. It prints the same measures used throughout
     the project, next to Tier 1 confidence, which the old router used.
  2. Sets the cut-point from those held-out predictions: the percentile that escalates TARGET_SHARE of
     questions (default 30%). One cut-point serves both business units; the router escalates more of a
     weaker unit's answers on its own, because it predicts lower quality for them.
  3. Trains the final model on every row and saves it to models/router/ with a small _info.json and
     what_it_learned.txt (the words and phrases that move the prediction most).

Run from the project root, inside the venv. It takes seconds:
    python scripts/train_text_router.py                       # 30% target escalation share
    python scripts/train_text_router.py --target-share 0.25   # escalate fewer questions

Retrain whenever the Tier 1 models change, or after the history has grown: the router is trained on
a particular model's answers.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.learned_router import MODEL_PATH, TARGET_SHARE, TextProbeRouter, make_version  # noqa: E402

DEFAULT_HISTORY = "results/logs/results_history.csv"
LOW_QUALITY = 3.5
FALLBACK_CLAUDE_QUALITY = 4.38


def load_history(path: str) -> pd.DataFrame:
    """Load the results history and check it has the columns and enough rows (200, over at least 5
    intents) to train the router on. Exits with a message that says what to run if not.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run this from the project root, or pass --history")
    df = pd.read_csv(path)
    needed = ["query", "tier1_answer", "tier1_confidence_avg", "business_unit", "intent", "quality_draft_overall"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        hint = " Run: python scripts/backfill_quality_scores.py --draft" if "quality_draft_overall" in missing else ""
        raise SystemExit(f"{path} is missing columns {missing}.{hint}")
    n_all = len(df)
    df = df.dropna(subset=["query", "tier1_answer", "tier1_confidence_avg", "quality_draft_overall"]).reset_index(drop=True)
    if len(df) < 200:
        raise SystemExit(f"Only {len(df)} rows have everything the router needs (of {n_all}); need at least 200. "
                         f"Run the pipeline with quality scoring on, then: python scripts/backfill_quality_scores.py --draft")
    if df["business_unit"].nunique() < 1 or df["intent"].nunique() < 5:
        raise SystemExit("Need at least 5 distinct intents to hold whole intents out for the evaluation.")
    return df


def claude_quality(df: pd.DataFrame) -> float:
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


def out_of_fold(df: pd.DataFrame, folds: int = 5) -> np.ndarray:
    """Predictions for every row from a model that never saw that row's intent."""
    from sklearn.model_selection import GroupKFold
    q, a, c, u, y = df["query"].tolist(), df["tier1_answer"].tolist(), df["tier1_confidence_avg"].values, df["business_unit"].tolist(), df["quality_draft_overall"].values
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=min(folds, df["intent"].nunique())).split(df, groups=df["intent"].fillna("none")):
        r = TextProbeRouter.fit([q[i] for i in tr], [a[i] for i in tr], c[tr], [u[i] for i in tr], y[tr])
        oof[te] = r.predict([q[i] for i in te], [a[i] for i in te], c[te], [u[i] for i in te])
    return oof


def delivered_per_unit(df, score, frac, claude_q):
    """Average delivered quality if the `frac` lowest-scoring answers in EACH unit go to Claude."""
    units = df["business_unit"].values
    q = df["quality_draft_overall"].values.astype(float).copy()
    for u in np.unique(units):
        ix = np.where(units == u)[0]
        q[ix[np.argsort(score[ix])[: int(round(frac * len(ix)))]]] = claude_q
    return float(q.mean())


def main(argv=None) -> int:
    """Evaluate the router on held-out intents, set the cut-point from a target escalation share, train
    the final model on every row, and save it with its info file and the list of words it weighs
    most.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=DEFAULT_HISTORY)
    ap.add_argument("--target-share", type=float, default=TARGET_SHARE, help="share of questions to escalate (default 0.30)")
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args(argv)
    if not 0.02 <= args.target_share <= 0.98:
        raise SystemExit("--target-share must be between 0.02 and 0.98")

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    df = load_history(args.history)
    claude_q = claude_quality(df)
    y = df["quality_draft_overall"].values.astype(float)
    units = sorted(df["business_unit"].unique())
    print(f"{len(df)} rows, {df['intent'].nunique()} intents, units: {', '.join(units)}. "
          f"Average Tier 1 quality {y.mean():.2f}; Claude's average {claude_q:.2f}.")

    print("\nStep 1 of 3: honest evaluation (5 folds, each holding out whole intents)...")
    oof = out_of_fold(df)
    conf = df["tier1_confidence_avg"].values
    u_arr = df["business_unit"].values

    print("\nCorrelation with Claude's quality marks, within each unit (higher is better):")
    print(f"  {'':16s}" + "".join(f"{u:>14s}" for u in units) + f"{'mean of units':>16s}")
    for label, score in (("Tier 1 confidence", conf), ("Learned router", oof)):
        rs = [spearmanr(score[u_arr == u], y[u_arr == u])[0] for u in units]
        print(f"  {label:16s}" + "".join(f"{r:>+14.3f}" for r in rs) + f"{np.mean(rs):>+16.3f}")
    if (y < LOW_QUALITY).any() and (y >= LOW_QUALITY).any():
        print(f"AUC for spotting an answer below {LOW_QUALITY} (0.5 = chance): confidence {roc_auc_score(y < LOW_QUALITY, -conf):.3f}, "
              f"learned router {roc_auc_score(y < LOW_QUALITY, -oof):.3f}")

    cutoff = float(np.quantile(oof, args.target_share))
    esc = oof < cutoff
    delivered_cut = float(np.where(esc, claude_q, y).mean())
    print(f"\nStep 2 of 3: the cut-point. Escalating {args.target_share:.0%} of questions means escalating answers predicted below {cutoff:.3f}.")
    print("  On the held-out predictions that escalates:  " + ", ".join(f"{u} {esc[u_arr == u].mean():.0%}" for u in units))
    print(f"  Average quality delivered at that cut-point: {delivered_cut:.3f}   "
          f"(escalating by confidence instead, the same share in each unit: {delivered_per_unit(df, conf, args.target_share, claude_q):.3f}; "
          f"by chance: {y.mean() + args.target_share * (claude_q - y.mean()):.3f})")

    print("\nStep 3 of 3: training the final model on every row and saving it...")
    version = make_version()
    meta = {"version": version, "trained_rows": int(len(df)), "intents": int(df["intent"].nunique()),
            "target_share": args.target_share, "claude_quality_assumed": round(claude_q, 3),
            "implied_minimum_gain": round(claude_q - cutoff, 3),
            "heldout_correlation_mean_of_units": round(float(np.mean([spearmanr(oof[u_arr == u], y[u_arr == u])[0] for u in units])), 3),
            "heldout_share_escalated_by_unit": {u: round(float(esc[u_arr == u].mean()), 3) for u in units}}
    router = TextProbeRouter.fit(df["query"].tolist(), df["tier1_answer"].tolist(), conf, df["business_unit"].tolist(), y, cutoff=cutoff, meta=meta)
    router.save(args.model_path)

    learned = router.top_features()
    path = os.path.join(os.path.dirname(args.model_path) or ".", "what_it_learned.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"What the learned router weighs ({version}). Weights are quality marks added or removed; "
                f"words are associations in the training data, not causes.\n")
        for part in ("question", "answer"):
            f.write(f"\nWords and phrases in the {part.upper()} that push the predicted quality DOWN:\n")
            f.write("  " + ", ".join(f"{w} ({c:+.2f})" for w, c in learned[part]["down"]) + "\n")
            f.write(f"Words and phrases in the {part.upper()} that push it UP:\n")
            f.write("  " + ", ".join(f"{w} ({c:+.2f})" for w, c in learned[part]["up"]) + "\n")

    print(f"\nSaved {args.model_path} ({version}), its _info.json, and {path}.")
    print(f"The router will escalate an answer when its predicted quality is below {cutoff:.3f} "
          f"(Claude's {claude_q:.2f} minus a minimum worthwhile gain of {claude_q - cutoff:.2f}).")
    print("Run the pipeline as usual:  python -m src.pipeline 100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
