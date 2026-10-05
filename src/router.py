"""
The original routing rule: escalate when Tier 1's confidence is below a pass mark.

route(confidence, threshold) returns "LOCAL" when confidence >= threshold (a score exactly at the pass mark stays
local) and "ESCALATE" otherwise. THRESHOLD (0.5) is only the default for calling route() on its own. The pipeline
passes each business unit's own pass mark (THRESHOLDS in pipeline.py: 0.71 for payments and 0.60 for retail bank),
because the two models write their confidence on different scales.

Status. Since 5 October 2026 this rule is no longer the pipeline's default. The learned router (learned_router.py)
predicts how good the answer is from the question, the answer, the confidence and the business unit, and it
predicts quality far better than confidence alone does. This rule is kept for two reasons: it runs on purpose with
    python -m src.pipeline 100 --threshold-router
so the two routers can be compared on the same questions, and tests/test_router.py documents its boundary behaviour.

See it work (no dependencies, no Ollama):    python -m src.router
"""

THRESHOLD = 0.5


def route(confidence_score: float, threshold: float = THRESHOLD) -> str:
    """Return 'LOCAL' if confidence clears the threshold, else 'ESCALATE'."""
    return "LOCAL" if confidence_score >= threshold else "ESCALATE"


if __name__ == "__main__":
    for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"confidence={score} -> {route(score)}")
