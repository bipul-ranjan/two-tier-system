"""
Stage 1 of the learned router: can a small language model, reading the query, the Tier 1 answer
and the Tier 1 confidence, predict how good that answer is better than confidence alone?

Evaluation only -- nothing here touches the pipeline. It predicts the five quality dimensions
Claude scores (correctness, completeness, tone, safety, clarity; the overall score is their
average) and is judged the way the earlier probe was: five folds that each hold out WHOLE
INTENTS, so near-duplicate queries can't leak; the same metrics; the same baselines (Tier 1
confidence and the TF-IDF text probe). If you have run scripts/probe_reward_model.py, its
reward-model scores are compared on the rows they cover.

Two modes (both use MiniLM, all-MiniLM-L6-v2 by default):
  frozen    MiniLM reads each query and answer once; a ridge model learns on top. About 7 minutes
            on one CPU core, full 5-fold evaluation. Run this first: it shows whether MiniLM's
            representation adds anything over word counts.
  finetune  MiniLM itself is trained (query+answer read together, confidence fed to the head).
            Much slower on CPU (hours for 5 folds; measured 0.3 s per training row on one core)
            but minutes on a GPU, e.g. a Colab session. Each fold is saved as it finishes, so
            Ctrl+C is safe and a re-run continues; --stop-after 1 runs a single fold to take a
            first look.

Run from the project root, inside the venv:
    python scripts/train_router.py                              # frozen, 5 folds
    python scripts/train_router.py --mode finetune --stop-after 1
    python scripts/train_router.py --mode finetune --epochs 2 --max-length 192   # faster CPU run
  python scripts/train_router.py --mode finetune --report-only                 # report on saved folds, train nothing
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_reward_model import (ESCALATION_RATES, LOW_QUALITY, claude_quality_of, delivered_quality,  # noqa: E402
                                load_history, read_cache, spearman, text_probe_oof)

DEFAULT_BASE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HISTORY = "results/logs/results_history.csv"
DEFAULT_OUT_DIR = "results/analysis"
DIMS = ["correctness", "completeness", "tone", "safety", "clarity"]
LABELS = [f"quality_draft_{d}" for d in DIMS]


# ----------------------------------------------------------------------------- data

def load_data(path: str, n: int = None, seed: int = 0) -> pd.DataFrame:
    df = load_history(path)
    missing = [c for c in LABELS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}. Run: python scripts/backfill_quality_scores.py --draft")
    df = df.dropna(subset=LABELS).reset_index(drop=True)
    df["tier1_confidence_min"] = df["tier1_confidence_min"].fillna(df["tier1_confidence_avg"]) if "tier1_confidence_min" in df.columns else df["tier1_confidence_avg"]
    if n:
        df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    return df


def numeric_features(df: pd.DataFrame, units: list) -> np.ndarray:
    """What the router knows besides the text: how confident Tier 1 was, how long the texts are, which unit."""
    cols = [df["tier1_confidence_avg"], df["tier1_confidence_min"], np.log1p(df["tier1_answer"].str.len()), np.log1p(df["query"].str.len())]
    cols += [(df["business_unit"] == u).astype(float) for u in units]
    return np.c_[tuple(cols)].astype(np.float32)


def make_folds(df: pd.DataFrame, k: int):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=k).split(df, groups=df["intent"].fillna("none")))


# ----------------------------------------------------------------------------- encoder

def load_encoder(base_model: str, device: str):
    from transformers import AutoModel, AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(base_model), AutoModel.from_pretrained(base_model).to(device)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not load {base_model} ({e}). It needs internet the first time; all-MiniLM-L6-v2 is already in your cache "
                         f"if the semantic cache has run.")


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed_texts(texts, tokenizer, encoder, device, max_length, batch_size=32, label=""):
    """Mean-pooled, L2-normalised embeddings -- what all-MiniLM-L6-v2 does inside sentence-transformers."""
    import torch
    order = np.argsort([len(t) for t in texts])              # similar lengths together: less padding
    out = np.zeros((len(texts), encoder.config.hidden_size), dtype=np.float32)
    encoder.eval()
    t0 = time.time()
    for b, start in enumerate(range(0, len(texts), batch_size)):
        idx = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.inference_mode():
            v = torch.nn.functional.normalize(mean_pool(encoder(**enc).last_hidden_state, enc["attention_mask"]), dim=-1)
        out[idx] = v.cpu().numpy()
        if label and (b + 1) % 40 == 0:
            done = start + len(idx)
            print(f"  {label}: {done}/{len(texts)}  (about {(time.time() - t0) / done * (len(texts) - done) / 60:.0f} min left)")
    return out


def get_embeddings(df, base_model, max_length, device, out_dir):
    """Query and answer embeddings for every row, cached on disk so a re-run is instant."""
    path = os.path.join(out_dir, f"router_embeddings_{os.path.basename(base_model.rstrip('/'))}_L{max_length}.npz")
    cached = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        cached = {k: (q, a) for k, q, a in zip(z["keys"], z["q"], z["a"])}
    todo = df[~df["key"].isin(cached)]
    if len(todo):
        print(f"Embedding {len(todo)} rows with {base_model} on {device} ({len(df) - len(todo)} already cached)...")
        tokenizer, encoder = load_encoder(base_model, device)
        q = embed_texts(todo["query"].tolist(), tokenizer, encoder, device, max_length, label="queries")
        a = embed_texts(todo["tier1_answer"].tolist(), tokenizer, encoder, device, max_length, label="answers")
        cached.update({k: (qq, aa) for k, qq, aa in zip(todo["key"], q, a)})
        os.makedirs(out_dir, exist_ok=True)
        keys = np.array(list(cached))
        np.savez(path, keys=keys, q=np.stack([cached[k][0] for k in keys]), a=np.stack([cached[k][1] for k in keys]))
    return (np.stack([cached[k][0] for k in df["key"]]), np.stack([cached[k][1] for k in df["key"]]))


# ----------------------------------------------------------------------------- mode 1: frozen encoder + ridge

def frozen_oof(df, folds, base_model, max_length, device, out_dir):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    units = sorted(df["business_unit"].unique())
    q, a = get_embeddings(df, base_model, max_length, device, out_dir)
    X = np.c_[q, a, q * a, (q * a).sum(1, keepdims=True), numeric_features(df, units)]
    Y = df[LABELS].values.astype(np.float64)
    oof = np.zeros_like(Y)
    for f, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
        Ytr = (Y[tr] - ymu) / ysd
        # ridge strength chosen by an inner split that ALSO holds out whole intents, using training rows only
        best, best_err = None, np.inf
        for alpha in (100.0, 300.0, 1000.0, 3000.0, 10000.0):
            err = 0.0
            for itr, ite in GroupKFold(n_splits=3).split(Xtr, groups=df["intent"].fillna("none").values[tr]):
                err += ((Ridge(alpha=alpha).fit(Xtr[itr], Ytr[itr]).predict(Xtr[ite]) - Ytr[ite]) ** 2).mean()
            if err < best_err:
                best, best_err = alpha, err
        oof[te] = Ridge(alpha=best).fit(Xtr, Ytr).predict(Xte) * ysd + ymu
        print(f"  fold {f + 1}/{len(folds)}: ridge alpha {best:g}")
    return pd.DataFrame(oof, columns=[f"p_{d}" for d in DIMS]).assign(key=df["key"].values)


# ----------------------------------------------------------------------------- mode 2: fine-tuned encoder

def build_net(encoder, n_numeric):
    import torch

    class RouterNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Sequential(torch.nn.Linear(encoder.config.hidden_size + n_numeric, 256), torch.nn.ReLU(),
                                            torch.nn.Dropout(0.1), torch.nn.Linear(256, len(DIMS)))

        def forward(self, enc, numeric):
            pooled = mean_pool(self.encoder(**enc).last_hidden_state, enc["attention_mask"])
            return self.head(torch.cat([pooled, numeric], dim=1))

    return RouterNet()


def finetune_fold(df, tr, te, units, args, device, fold):
    import torch
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(args.seed + fold)
    rng = np.random.default_rng(args.seed + fold)
    tokenizer, encoder = load_encoder(args.base_model, device)
    net = build_net(encoder, 4 + len(units)).to(device)
    num = numeric_features(df, units)
    mu, sd = num[tr].mean(0), num[tr].std(0) + 1e-6
    num = (num - mu) / sd
    Y = df[LABELS].values.astype(np.float32)
    ymu, ysd = Y[tr].mean(0), Y[tr].std(0)
    Yz = (Y - ymu) / ysd
    queries, answers = df["query"].tolist(), df["tier1_answer"].tolist()

    def encode(idx):
        return tokenizer([queries[i] for i in idx], [answers[i] for i in idx], padding=True, truncation="only_second",
                         max_length=args.max_length, return_tensors="pt").to(device)

    opt = torch.optim.AdamW([{"params": net.encoder.parameters(), "lr": args.lr}, {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    net.train(); t0 = time.time(); done = 0
    for epoch in range(args.epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss = torch.nn.functional.mse_loss(net(encode(idx), torch.tensor(num[idx], device=device)), torch.tensor(Yz[idx], device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
            done += 1
            if done % 25 == 0:
                rate = (time.time() - t0) / done
                print(f"    fold {fold + 1} epoch {epoch + 1}/{args.epochs} step {done}/{steps} loss {loss.item():.3f}  (about {rate * (steps - done) / 60:.0f} min left)")
    net.eval(); out = np.zeros((len(te), len(DIMS)), dtype=np.float32)
    order = np.argsort([len(answers[i]) for i in te])
    with torch.inference_mode():
        for s in range(0, len(te), 64):
            j = order[s:s + 64]; idx = te[j]
            out[j] = net(encode(idx), torch.tensor(num[idx], device=device)).cpu().numpy()
    return out * ysd + ymu


def finetune_oof(df, folds, args, device, out_dir):
    units = sorted(df["business_unit"].unique())
    tag = f"finetune_{os.path.basename(args.base_model.rstrip('/'))}_L{args.max_length}_E{args.epochs}_F{len(folds)}_S{args.seed}_N{len(df)}"
    path = os.path.join(out_dir, f"router_oof_{tag}.csv")
    os.makedirs(out_dir, exist_ok=True)
    pred_cols = [f"p_{d}" for d in DIMS]
    # An empty accumulator must have NUMBER columns: combined with an all-object empty table, newer pandas
    # leaves the whole predictions column typed "object", and correlation then fails on some numpy/scipy versions.
    have = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(
        {"key": pd.Series(dtype=str), "fold": pd.Series(dtype=int), **{c: pd.Series(dtype=float) for c in pred_cols}})
    ran = 0
    for f, (tr, te) in enumerate(folds):
        if set(df["key"].values[te]) <= set(have["key"]):
            print(f"fold {f + 1}/{len(folds)}: already done")
            continue
        if args.report_only:
            continue
        if args.stop_after and ran >= args.stop_after:
            break
        print(f"fold {f + 1}/{len(folds)}: training on {len(tr)} rows, testing on {len(te)} rows from unseen intents ({device})")
        try:
            pred = finetune_fold(df, tr, te, units, args, device, f)
        except KeyboardInterrupt:
            print("\nInterrupted. Finished folds are saved -- run the same command again to continue.")
            return None
        new = pd.DataFrame(pred, columns=[f"p_{d}" for d in DIMS]).assign(key=df["key"].values[te], fold=f)
        have = pd.concat([have, new], ignore_index=True)
        have.to_csv(path, index=False)
        ran += 1
    have[pred_cols] = have[pred_cols].astype(float)
    return have.drop(columns="fold", errors="ignore")


# ----------------------------------------------------------------------------- evaluation

def boot_diff(S, a, b, frac, claude_q, B=1000, seed=0):
    """95% interval of the difference in delivered quality (a minus b), resampling whole intents."""
    rng = np.random.default_rng(seed)
    groups = {i: g for i, g in S.groupby("intent")}
    names = list(groups)
    diffs = []
    for _ in range(B):
        d = pd.concat([groups[i] for i in rng.choice(names, len(names), replace=True)], ignore_index=True)
        diffs.append(delivered_quality(d, a, frac, claude_q) - delivered_quality(d, b, frac, claude_q))
    return np.percentile(diffs, [2.5, 97.5])


def report(S, claude_q, rm_rows=None):
    from sklearn.metrics import roc_auc_score
    S = S.copy()
    units = sorted(S["business_unit"].unique())
    signals = [("confidence", "conf"), ("text probe", "text_probe"), ("router", "router")]
    for _, col in signals:
        S[col] = pd.to_numeric(S[col]).astype(float)
    print("\n=== Does a small language model read the answer better than confidence does? ===")
    print(f"rows evaluated: {len(S)} ({', '.join(f'{u} {int((S.business_unit == u).sum())}' for u in units)}) from {S['intent'].nunique()} intents, "
          f"every one scored by a model that never saw its intent.")

    def table(frame, sigs, title):
        print(f"\n{title}")
        print(f"  {'':14s}" + "".join(f"{u:>26s}" for u in units) + f"{'mean of units':>16s}")
        res = {}
        for label, col in sigs:
            cells, rs = [], []
            for u in units:
                g = frame[frame.business_unit == u]
                r, lo, hi = spearman(g[col], g["quality_draft_overall"]); rs.append(r)
                cells.append(f"{r:+.3f} [{lo:+.2f},{hi:+.2f}]")
            res[label] = float(np.mean(rs))
            print(f"  {label:14s}" + "".join(f"{c:>26s}" for c in cells) + f"{res[label]:>+16.3f}")
        return res

    res = table(S, signals, "Spearman correlation with Claude's quality score, within each unit (95% interval in brackets):")

    bad = (S["quality_draft_overall"] < LOW_QUALITY).values
    if bad.any() and not bad.all():
        print(f"\nAUC for spotting a Tier 1 answer scoring below {LOW_QUALITY} (0.5 = chance): "
              + " | ".join(f"{label} {roc_auc_score(bad, -S[col]):.3f}" for label, col in signals))

    print("\nWhich quality dimension does each signal track? (Spearman within unit, averaged over units)")
    print(f"  {'':14s}" + "".join(f"{d:>14s}" for d in DIMS))
    for label, col in signals:
        vals = [np.mean([spearman(S.loc[S.business_unit == u, col], S.loc[S.business_unit == u, f"quality_draft_{d}"])[0] for u in units]) for d in DIMS]
        print(f"  {label:14s}" + "".join(f"{v:>+14.3f}" for v in vals))

    print(f"\nAverage quality delivered at the same escalation rate (kept rows = real Tier 1 quality, escalated rows = Claude's {claude_q:.2f}):")
    print(f"  {'escalate':>9} | {'random':>7} | {'confidence':>10} | {'text probe':>10} | {'router':>7} | {'oracle':>7}")
    rng = np.random.default_rng(0)
    for frac in ESCALATION_RATES:
        rnd = np.mean([delivered_quality(S.assign(_r=rng.random(len(S))), "_r", frac, claude_q) for _ in range(20)])
        row = [rnd] + [delivered_quality(S, col, frac, claude_q) for _, col in signals] + [delivered_quality(S.assign(_o=S["quality_draft_overall"]), "_o", frac, claude_q)]
        print(f"  {frac:>8.0%} | " + " | ".join(f"{v:>{w}.3f}" for v, w in zip(row, (7, 10, 10, 7, 7))))
    rnd30 = np.mean([delivered_quality(S.assign(_r=rng.random(len(S))), "_r", 0.3, claude_q) for _ in range(30)])
    orc30 = delivered_quality(S.assign(_o=S["quality_draft_overall"]), "_o", 0.3, claude_q)
    print("  share of the possible gain (random -> oracle) captured at 30%: "
          + " | ".join(f"{label} {(delivered_quality(S, col, 0.3, claude_q) - rnd30) / (orc30 - rnd30):.0%}" for label, col in signals))

    print("\nIs the difference beyond noise? Delivered quality at 30%, 95% interval resampling whole intents:")
    for a, b, la, lb in (("router", "conf", "router", "confidence"), ("router", "text_probe", "router", "text probe")):
        lo, hi = boot_diff(S, a, b, 0.3, claude_q)
        pt = delivered_quality(S, a, 0.3, claude_q) - delivered_quality(S, b, 0.3, claude_q)
        print(f"  {la} minus {lb:<11s} {pt:+.3f}   [{lo:+.3f}, {hi:+.3f}]   " + ("beyond noise" if lo > 0 or hi < 0 else "cannot be told apart from zero"))

    if rm_rows is not None:
        R = S.merge(rm_rows, on="key", how="inner", suffixes=("", "_rm"))
        if len(R) >= 100 and R["business_unit"].nunique() == len(units):
            rsig = [("confidence", "conf"), ("reward model", "rm_score"), ("text probe", "text_probe"), ("router", "router")]
            table(R, rsig, f"Same comparison on the {len(R)} rows that also have zero-shot reward-model scores (scripts/probe_reward_model.py):")
            print("  delivered quality at 30%: " + " | ".join(f"{l} {delivered_quality(R, c, 0.3, claude_q):.3f}" for l, c in rsig))
        else:
            print(f"\n(Only {len(R)} evaluated rows have reward-model scores -- too few to compare on; run more folds.)")
    r, p = res["router"], res["text probe"]
    print(f"\nWithin units the router is {'above' if r > p else 'below'} the text probe ({r:+.3f} vs {p:+.3f}) and "
          f"{'above' if r > res['confidence'] else 'below'} confidence ({r:+.3f} vs {res['confidence']:+.3f}). Differences under about 0.1 are within noise.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--base-model", default=DEFAULT_BASE, help=f"encoder (default {DEFAULT_BASE})")
    ap.add_argument("--history", default=DEFAULT_HISTORY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--folds", type=int, default=5, help="held-out-intent folds (default 5)")
    ap.add_argument("--stop-after", type=int, default=0, help="finetune: stop after this many NEW folds (take a first look)")
    ap.add_argument("--report-only", action="store_true", help="finetune: train nothing, just report on the folds already saved (use the same settings as the run that saved them)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256, help="tokens of query+answer the model reads (default 256)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate for finetune")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="use only a random sample of this many rows (quick smoke test)")
    args = ap.parse_args(argv)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_data(args.history, args.n, args.seed)
    folds = make_folds(df, args.folds)
    print(f"{len(df)} rows, {df['intent'].nunique()} intents, {args.folds} folds that each hold out whole intents. Mode: {args.mode} ({device}).")

    if args.mode == "frozen":
        preds = frozen_oof(df, folds, args.base_model, args.max_length, device, args.out_dir)
    else:
        preds = finetune_oof(df, folds, args, device, args.out_dir)
        if preds is None:
            return 1
    S = df.merge(preds, on="key", how="inner")
    if len(S) < 50:
        print("No finished folds saved for these settings -- use exactly the settings of the run that saved them, or run without --report-only."
              if args.report_only else "Too few rows evaluated yet -- run again to finish more folds.")
        return 1
    pred_cols = [f"p_{d}" for d in DIMS]
    S[pred_cols] = S[pred_cols].apply(pd.to_numeric).astype(float)
    S["router"] = S[pred_cols].mean(axis=1)
    S["conf"] = S["tier1_confidence_avg"]
    S["text_probe"] = pd.Series(text_probe_oof(df), index=df["key"]).reindex(S["key"]).values
    rm_path = os.path.join(args.out_dir, "reward_model_zero_shot_Skywork-Reward-V2-Qwen3-0.6B.csv")
    rm_rows = read_cache(rm_path) if os.path.exists(rm_path) else None
    report(S, claude_quality_of(pd.read_csv(args.history)), rm_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
