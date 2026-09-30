"""
Quality estimation: computes REAL accuracy against ground truth (not the
confidence proxy router.py uses at decision time), and combines it with
cost into the Q(x) - lambda*cost score used to evaluate/tune the cascade.

Important distinction, worth keeping straight:
- router.py's confidence score is used AT INFERENCE TIME, before the true
  answer is known -- it's a proxy for quality, necessarily imperfect.
- This module is used OFFLINE, during evaluation, where ground truth
  answers ARE available (from Banking77's label_text, FinQA's exe_ans).
  This is where Q(x) - lambda*cost actually becomes computable and
  meaningful -- as a way to score and compare cascade configurations
  after the fact, not as a live routing rule.

judge_answer_quality() below is a THIRD, separate scoring mechanism for a
different situation: the live payments/retail_bank pipeline answers free-text
Bitext-style queries with no ground-truth label at all, so neither
banking77_quality nor finqa_quality applies. It uses Claude as an LLM-judge
instead (the same general approach as MT-Bench / Prometheus-style academic
evaluation) -- necessarily a softer, more subjective signal than exact
ground-truth matching, and worth naming as a limitation in your methodology
section for exactly that reason.

Part of the src/ package -- run from the project root with:
    python -m src.quality
"""
import json
import re
import time
import requests


def normalize_text(text: str) -> str:
    """Loose normalization so 'Card Payment Decline Inquiry' can be
    compared against a canonical label like 'card_payment_not_recognised'
    -- lowercase, strip punctuation, collapse underscores/spaces.
    """
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def banking77_quality(predicted_intent: str, true_label_text: str) -> float:
    """Approximate accuracy for a Banking77 intent prediction.

    IMPORTANT LIMITATION: Tier 1's prompt asks the model to freely name an
    intent (e.g. "Card Payment Decline Inquiry"), not to pick from
    Banking77's exact 77 canonical labels (e.g. "card_payment_not_recognised").
    This function does a normalized substring match as a rough proxy, not
    an exact-match accuracy score. If you need a rigorous number for your
    dissertation, the more defensible fix is to constrain Tier 1's prompt
    to output one of the 77 canonical labels directly, then use exact
    match here instead -- flag this as a known approximation either way.
    """
    pred_norm = normalize_text(predicted_intent)
    true_norm = normalize_text(true_label_text)

    if pred_norm == true_norm:
        return 1.0

    pred_words = set(pred_norm.split())
    true_words = set(true_norm.split())
    if not true_words:
        return 0.0

    overlap = len(pred_words & true_words) / len(true_words)
    return overlap  # partial credit, not binary -- see limitation note above


def finqa_quality(predicted_answer, true_exe_ans, tolerance: float = 0.01) -> float:
    """Numeric accuracy for a FinQA answer -- 1.0 if within tolerance of
    the ground truth executed answer, else 0.0. This is a well-defined,
    defensible metric (unlike banking77_quality above) since FinQA's
    exe_ans is already a clean float, not free text needing normalization.
    """
    try:
        pred_val = float(predicted_answer)
        true_val = float(true_exe_ans)
    except (TypeError, ValueError):
        return 0.0

    if true_val == 0:
        return 1.0 if abs(pred_val) < tolerance else 0.0

    relative_error = abs(pred_val - true_val) / abs(true_val)
    return 1.0 if relative_error <= tolerance else 0.0


QUALITY_JUDGE_MODEL = "claude-sonnet-5"  # a more careful judge than the cheap Tier 2 escalation
# model -- quality scoring for a dissertation is worth the modest extra cost over Haiku.


def _extract_text(response) -> str:
    """response.content can include a ThinkingBlock (the model's internal reasoning) before
    the actual text answer, when extended thinking is active for this model/account --
    content[0] is NOT reliably the answer. Concatenate every block whose type is "text"
    instead of assuming position 0, which is what actually crashed here (ThinkingBlock has
    no .text attribute)."""
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")

QUALITY_DIMS = ["correctness", "completeness", "tone", "safety", "clarity"]

_QUALITY_RUBRIC = """You are scoring a customer-support answer from a retail banking assistant. Score the ANSWER against the QUERY on five dimensions, each 1-5 (1=poor, 5=excellent):

1. correctness: Is the factual/procedural content accurate for a retail bank?
2. completeness: Does it fully address what the customer asked?
3. tone: Is the tone appropriate -- professional, and empathetic if the query involves distress, fraud, hardship, or a vulnerable situation?
4. safety: Does it avoid giving inappropriate legal/financial advice, and correctly point to escalation/human help where that is required (e.g. fraud, legal matters, safeguarding)?
5. clarity: Is it clear and well organised?

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"correctness": <1-5>, "completeness": <1-5>, "tone": <1-5>, "safety": <1-5>, "clarity": <1-5>, "note": "<one sentence justification>"}}

QUERY: {query}

ANSWER: {answer}"""


def judge_answer_quality_local(model: str, query: str, answer: str, ollama_url: str = None, retries: int = 2, retry_delay_s: float = 1.0) -> dict | None:
    """Same rubric as judge_answer_quality(), but scored by a LOCAL Ollama model instead of
    Claude -- cheap and fast enough to run over an entire run's worth of rows, but a weaker,
    less validated judge (see the module-level note above). Intended as a broad first-pass
    signal, not as the numbers you report: compare against judge_answer_quality() on an
    overlapping sample (scripts/compare_quality_judges.py) before trusting it on its own for
    anything you cite. model must already be pulled in Ollama (e.g. "llama3.2:3b") -- a
    model NOT already fine-tuned as one of your production assistants is the right choice
    here, so the judge isn't evaluating its own family's output style.
    """
    from .config import OLLAMA_URL as _DEFAULT_OLLAMA_URL
    url = ollama_url or _DEFAULT_OLLAMA_URL
    prompt = _QUALITY_RUBRIC.format(query=query, answer=answer)
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False}, timeout=60)
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip()
            # local models are more prone to wrapping the JSON in a sentence or two --
            # fall back to extracting the first {...} block if a direct parse fails.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                start, end = text.find("{"), text.rfind("}")
                if start == -1 or end == -1:
                    raise
                parsed = json.loads(text[start:end + 1])
            if not all(k in parsed for k in QUALITY_DIMS):
                raise ValueError(f"missing expected keys: {parsed}")
            return parsed
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(retry_delay_s)
    print(f"    WARNING: local quality judge ({model}) failed after {retries} attempts ({last_error}) -- leaving this row unscored")
    return None


def judge_answer_quality(client, query: str, answer: str, retries: int = 3, retry_delay_s: float = 2.0) -> dict | None:
    """Score one (query, answer) pair on the five QUALITY_DIMS using Claude as an LLM-judge.
    Returns a dict with each dimension 1-5 plus a "note" justification, or None if every
    retry attempt failed to parse -- callers should skip that row rather than crash the
    whole scoring run over one bad response."""
    prompt = _QUALITY_RUBRIC.format(query=query, answer=answer)
    last_error = None
    for attempt in range(retries):
        try:
            # thinking explicitly disabled: extended thinking (if on by default for this model/
            # account) consumes part of max_tokens for its own reasoning before the model even
            # starts the actual JSON answer -- with max_tokens=300, that left too little (or
            # zero) room for the answer itself, producing truncated/empty JSON that failed to
            # parse. Disabling it removes the failure mode outright rather than just padding
            # max_tokens and hoping the split works out.
            resp = client.messages.create(model=QUALITY_JUDGE_MODEL, max_tokens=300,
                                           thinking={"type": "disabled"}, messages=[{"role": "user", "content": prompt}])
            text = _extract_text(resp).strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip()
            parsed = json.loads(text)
            if not all(k in parsed for k in QUALITY_DIMS):
                raise ValueError(f"missing expected keys: {parsed}")
            return parsed
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(retry_delay_s)
    print(f"    WARNING: quality judge failed after {retries} attempts ({last_error}) -- leaving this row unscored")
    return None


def score(quality: float, cost: float, lam: float = 10.0) -> float:
    """Q(x) - lambda*cost.

    lam (lambda) has units of "quality points per dollar" -- it's the
    dial that sets how much cost you're willing to trade for quality.
    lam=10 means you'd accept spending $1 for a 10-point (i.e. full-scale,
    since quality is 0-1) quality gain; higher lam values are more
    cost-sensitive, lower values are more quality-sensitive.
    """
    return quality - (lam * cost)


if __name__ == "__main__":
    # Banking77 example -- same case from your earlier tier1.py test
    q1 = banking77_quality("Card Payment Decline Inquiry", "card_payment_not_recognised")
    print(f"Banking77 quality (partial-credit overlap): {q1:.2f}")

    # FinQA example -- using the real record we verified earlier
    q2 = finqa_quality(predicted_answer=3.8, true_exe_ans=3.8)
    print(f"FinQA quality (exact numeric match): {q2}")

    q3 = finqa_quality(predicted_answer=3.75, true_exe_ans=3.8, tolerance=0.02)
    print(f"FinQA quality (within 2% tolerance): {q3}")

    # Combining into the routing/evaluation score
    s = score(quality=q2, cost=0.000655, lam=10.0)
    print(f"Score = Q - lambda*cost = {q2} - 10*0.000655 = {s:.4f}")
