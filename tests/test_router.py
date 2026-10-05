"""
Basic unit tests for the routing logic — the one piece of this project
with no external dependencies (no Ollama, no API, no files), so it's
the right place to start if you're adding tests for the first time.

Run from the project root with:
    pip install pytest
    pytest tests/
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.router import route, THRESHOLD


def test_high_confidence_resolves_locally():
    """A high confidence score routes to LOCAL."""
    assert route(0.9) == "LOCAL"


def test_low_confidence_escalates():
    """A low confidence score routes to ESCALATE."""
    assert route(0.1) == "ESCALATE"


def test_exact_threshold_resolves_locally():
    # route() uses >=, so a score exactly at the threshold should stay local
    """A score exactly at the pass mark stays LOCAL (the boundary is >=)."""
    assert route(THRESHOLD) == "LOCAL"


def test_custom_threshold_override():
    """Passing a pass mark of your own changes the decision."""
    assert route(0.6, threshold=0.7) == "ESCALATE"
    assert route(0.8, threshold=0.7) == "LOCAL"
