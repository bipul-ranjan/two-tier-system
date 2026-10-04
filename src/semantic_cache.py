"""
Semantic cache: sits between Tier 1 and Tier 2. When Tier 1's confidence is too low to
answer locally, check whether a sufficiently similar query has already been answered by
Tier 2 before paying for a fresh Claude call.

Single-threshold design: a candidate match is served directly if its similarity clears
SIMILARITY_THRESHOLD and two cheap, free checks pass -- no Claude verification call at any
point, so a cache hit never costs anything beyond the embedding lookup itself. (An earlier
version of this module added a second, "gray zone" threshold band with a Claude verification
call for ambiguous matches -- removed here by request: it only ever saved ~64% of a full
escalation's cost on that narrow band, not the much larger saving a true cache hit gives, and
the two free checks below already catch the single most costly failure mode -- negation --
without needing a model call to do it.)

Cache entries are seeded from Tier 2 (Claude) answers only -- the highest-quality source
already in results_history.csv -- not from Tier 1's own (lower, and per Finding #2, not
quality-equivalent) answers.
"""
import re

import numpy as np
import pandas as pd

SIMILARITY_THRESHOLD = 0.84  # at/above this + cheap checks pass: serve directly
                              # below this: cache miss, escalate to Tier 2 as normal

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, local, free -- no per-query API cost

# Negation markers checked for presence/asymmetry between the new query and the candidate
# cache entry's original query -- deliberately simple and conservative (a word-presence
# check, not real NLI) because the cost of a false "polarity matches" here is a wrong
# answer served with total confidence, so this check is only ever used to REJECT a
# candidate, never to approve one on its own.
#
# Deliberately restricted to TRUE grammatical negators only -- not action verbs like
# "cancel" or "stop". An earlier version of this set included those, and it broke on
# exactly the case it was meant to catch: "cancel my card" and "don't cancel my card"
# both contain "cancel", so a set that includes "cancel" sees both as "negated" and
# calls them consistent -- the opposite of correct. What actually distinguishes them is
# "don't", a grammatical negator, not the action word they share.
_NEGATION_WORDS = {
    "not", "n't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't",
    "can't", "cannot", "never", "no", "without", "unless", "except",
}


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z']+", text.lower()))


def negation_consistent(query_a: str, query_b: str) -> bool:
    """True if the two queries have the same negation "polarity" -- both contain a negation
    marker, or neither does. A mismatch (one negated, one not) is exactly the failure mode
    responsible for the most-cited class of semantic-cache false positives, so this check
    exists specifically to catch it."""
    a_neg = bool(_tokenize(query_a) & _NEGATION_WORDS)
    b_neg = bool(_tokenize(query_b) & _NEGATION_WORDS)
    return a_neg == b_neg


def entity_consistent(new_category, new_intent, cached_category, cached_intent) -> bool:
    """True if the structured category/intent labels match. This is a stand-in for a general
    named-entity check: on this project's labelled synthetic data, category+intent already
    encodes exactly the kind of parameter difference (domestic vs international, account type,
    etc.) that causes false cache hits elsewhere in the literature. On genuinely unlabelled
    production queries (no category/intent known in advance), this check would need to be
    replaced with real entity/slot extraction -- noted here deliberately, not silently assumed
    away, since it's a real scope boundary of this prototype."""
    return new_category == cached_category and new_intent == cached_intent


class SemanticCacheIndex:
    """An in-memory, brute-force cosine-similarity index. Appropriate at this project's scale
    (thousands, not millions, of cached entries) -- a full vector database would be premature
    infrastructure here, and brute-force cosine similarity on a few thousand normalized
    384-dim vectors is sub-second."""

    def __init__(self, embed_fn=None):
        self._embed_fn = embed_fn or self._default_embed_fn()
        self.queries = []
        self.answers = []
        self.categories = []
        self.intents = []
        self.embeddings = None  # (n, d) array, L2-normalized rows

    @staticmethod
    def _default_embed_fn():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)
        return lambda texts: model.encode(texts, normalize_embeddings=True)

    def build(self, history_df: pd.DataFrame):
        """Populate the index from Tier 2 (escalated) rows in results_history.csv.

        Safe to call on a brand-new project with no history yet -- results_history.csv not
        existing at all, or existing but completely empty (0 bytes, no header), both produce
        an empty DataFrame with no columns from HistoryLock.read(). Indexing a missing column
        on that DataFrame raises KeyError rather than returning nothing, which is why this is
        checked explicitly instead of just relying on .empty (a DataFrame with 0 rows but a
        real header has the needed columns and .empty is True either way -- it's the missing
        *columns*, not just missing rows, that needs guarding against).
        """
        needed = {"decision", "final_answer", "query"}
        if not needed.issubset(history_df.columns):
            self.embeddings = np.zeros((0, 1))
            return
        esc = history_df[(history_df["decision"] == "ESCALATE") & history_df["final_answer"].notna()]
        esc = esc.dropna(subset=["query"])
        if esc.empty:
            self.embeddings = np.zeros((0, 1))
            return
        self.queries = esc["query"].tolist()
        self.answers = esc["final_answer"].tolist()
        self.categories = esc["category"].tolist() if "category" in esc.columns else [None] * len(esc)
        self.intents = esc["intent"].tolist() if "intent" in esc.columns else [None] * len(esc)
        self.embeddings = np.asarray(self._embed_fn(self.queries))

    def best_match(self, query: str):
        """Returns (index, similarity) of the closest cached query, or (None, 0.0) if the
        index is empty."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return None, 0.0
        q_emb = np.asarray(self._embed_fn([query]))[0]
        sims = self.embeddings @ q_emb  # rows are already L2-normalized -> dot product = cosine sim
        best_i = int(np.argmax(sims))
        return best_i, float(sims[best_i])


def try_semantic_cache(index: SemanticCacheIndex, query: str, category=None, intent=None):
    """The full decision. Returns a dict with keys {answer, similarity, path, matched_query}
    on a hit, or None on a miss (caller should fall through to a normal Tier 2 escalation).
    No Claude call at any point -- a hit costs nothing beyond the embedding lookup."""
    idx, sim = index.best_match(query)
    if idx is None or sim < SIMILARITY_THRESHOLD:
        return None

    cached_query = index.queries[idx]
    cached_answer = index.answers[idx]
    cheap_checks_pass = (
        negation_consistent(query, cached_query)
        and entity_consistent(category, intent, index.categories[idx], index.intents[idx])
    )
    if not cheap_checks_pass:
        return None

    return {"answer": cached_answer, "similarity": sim, "path": "direct", "matched_query": cached_query}
