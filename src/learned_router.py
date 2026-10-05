"""
Learned router: decides whether Tier 1's answer is good enough to send, by PREDICTING how good it is.

It reads four things that exist the moment Tier 1 has answered -- the customer's question, Tier 1's
answer, Tier 1's confidence, and the business unit -- and predicts the answer's quality (1-5, the
overall mark Claude-as-judge would give it). The question is escalated when the prediction falls
below a cut-point. Quality itself is never an input: it only exists after Claude has marked an
answer, so it is the thing the router learns to predict (the quality_draft_overall column).

This is the "text probe" that stage 1 chose over fine-tuned and frozen MiniLM routers: TF-IDF counts
of the words and two-word phrases in the question and the answer, plus four numbers, fed to a ridge
regression. Held out by whole intents it scored +0.477 against +0.212 for confidence, and the
fine-tuned MiniLM router tied it, at about five hours of training. It decides in about a millisecond.

Train it with:   python scripts/train_text_router.py
Use it with:     python -m src.pipeline 100          (the default router)

The saved model is a scikit-learn object, so it only loads under the scikit-learn version that
trained it. A different version raises a clear error; the fix is to retrain (seconds).
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np

MODEL_PATH = "models/router/text_probe_router.joblib"
ROUTER_KIND = "text-probe"

TARGET_SHARE = 0.30        # share of questions escalated: the cut-point is the matching percentile
RIDGE_ALPHA = 3.0
MAX_FEATURES = 2000        # per text (question, answer)
MIN_DF = 3                 # a word or phrase must occur in at least this many rows to count


class RouterNotTrainedError(RuntimeError):
    """Raised when the learned router's model file is missing or unusable. Never silently ignored:
    the pipeline stops with the message, so nobody believes the learned router ran when it did not."""


def _vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=MIN_DF, max_features=MAX_FEATURES, sublinear_tf=True)


def _numeric(answers, queries, confidences, units, known_units) -> np.ndarray:
    """What the router knows besides the words: Tier 1's confidence, how long the question and the
    answer are, and which business unit it is. An unfamiliar unit gets all zeros in the unit columns."""
    ans_len = np.log1p([len(a or "") for a in answers])
    qry_len = np.log1p([len(q or "") for q in queries])
    unit_cols = [[1.0 if u == k else 0.0 for u in units] for k in known_units]
    return np.column_stack([np.asarray(confidences, dtype=float), ans_len, qry_len, *unit_cols]).astype(float)


class TextProbeRouter:
    """Fitted router. Build one with TextProbeRouter.fit(...), or load a saved one with .load()."""

    def __init__(self):
        self.vq = self.va = self.ridge = None
        self.mean = self.std = None
        self.units = []
        self.cutoff = None
        self.meta = {}

    # ----------------------------------------------------------------- features
    def _matrix(self, queries, answers, confidences, units):
        import scipy.sparse as sp
        num = _numeric(answers, queries, confidences, units, self.units)
        num = (num - self.mean) / self.std
        return sp.hstack([self.vq.transform(queries), self.va.transform(answers), sp.csr_matrix(num)]).tocsr()

    # ----------------------------------------------------------------- training
    @classmethod
    def fit(cls, queries, answers, confidences, units, quality, cutoff=None, meta=None):
        """Train on every row: question text, Tier 1 answer text, Tier 1 confidence, business unit, and
        the Claude-judged quality of that Tier 1 answer."""
        from sklearn.linear_model import Ridge
        r = cls()
        queries, answers, units = list(queries), list(answers), list(units)
        r.units = sorted(set(units))
        num = _numeric(answers, queries, confidences, units, r.units)
        r.mean, r.std = num.mean(0), num.std(0) + 1e-9
        r.vq, r.va = _vectorizer().fit(queries), _vectorizer().fit(answers)
        X = r._matrix(queries, answers, confidences, units)
        r.ridge = Ridge(alpha=RIDGE_ALPHA).fit(X, np.asarray(quality, dtype=float))
        r.cutoff = cutoff
        r.meta = dict(meta or {})
        return r

    # ----------------------------------------------------------------- use
    def predict(self, queries, answers, confidences, units) -> np.ndarray:
        """Predicted quality (1-5 scale) for each question and answer."""
        return self.ridge.predict(self._matrix(list(queries), list(answers), confidences, list(units)))

    def predict_one(self, query, answer, confidence, unit) -> float:
        return float(self.predict([query], [answer], [confidence], [unit])[0])

    def decide(self, predicted_quality: float) -> str:
        """ESCALATE when the predicted quality is below the cut-point, otherwise LOCAL."""
        return "ESCALATE" if predicted_quality < self.cutoff else "LOCAL"

    @property
    def version(self) -> str:
        return self.meta.get("version", ROUTER_KIND)

    # ----------------------------------------------------------------- explain
    def top_features(self, k=12):
        """The words and phrases that push the predicted quality down or up the most, for each text."""
        coef = self.ridge.coef_
        nq, na = len(self.vq.vocabulary_), len(self.va.vocabulary_)
        out = {}
        for label, names, c in (("question", self.vq.get_feature_names_out(), coef[:nq]),
                                ("answer", self.va.get_feature_names_out(), coef[nq:nq + na])):
            order = np.argsort(c)
            out[label] = {"down": [(str(names[i]), float(c[i])) for i in order[:k]],
                          "up": [(str(names[i]), float(c[i])) for i in order[::-1][:k]]}
        return out

    # ----------------------------------------------------------------- save / load
    def save(self, path: str = MODEL_PATH) -> None:
        import sklearn
        self.meta["sklearn_version"] = sklearn.__version__
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        with open(os.path.splitext(path)[0] + "_info.json", "w", encoding="utf-8") as f:
            json.dump({**self.meta, "cutoff": self.cutoff, "units": self.units}, f, indent=2)

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "TextProbeRouter":
        import sklearn
        if not os.path.exists(path):
            raise RouterNotTrainedError(
                f"The learned router has not been trained yet: {path} does not exist.\n"
                f"Train it first (takes seconds):  python scripts/train_text_router.py\n"
                f"Or, to compare against the old rule on purpose:  python -m src.pipeline 100 --threshold-router")
        try:
            router = joblib.load(path)
        except Exception as e:
            raise RouterNotTrainedError(f"{path} could not be loaded ({type(e).__name__}: {e}).\n"
                                        f"Retrain it:  python scripts/train_text_router.py") from e
        saved = router.meta.get("sklearn_version")
        if saved and saved != sklearn.__version__:
            raise RouterNotTrainedError(
                f"{path} was trained with scikit-learn {saved}, but {sklearn.__version__} is installed. "
                f"Retrain it (takes seconds):  python scripts/train_text_router.py")
        if router.cutoff is None:
            raise RouterNotTrainedError(f"{path} has no cut-point. Retrain it:  python scripts/train_text_router.py")
        return router


def make_version() -> str:
    """A short label for this training run, shown on the dashboard so a router change is visible."""
    return f"{ROUTER_KIND}-{datetime.now().strftime('%d%b%y-%H%M')}"
