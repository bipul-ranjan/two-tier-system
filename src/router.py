"""
Confidence-based routing: decide whether the Tier 1 output is resolved
locally or must escalate to Tier 2 (the shared LLM).
"""

THRESHOLD = 0.5


def route(confidence_score: float, threshold: float = THRESHOLD) -> str:
    """Return 'LOCAL' if confidence clears the threshold, else 'ESCALATE'."""
    return "LOCAL" if confidence_score >= threshold else "ESCALATE"


if __name__ == "__main__":
    for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"confidence={score} -> {route(score)}")
