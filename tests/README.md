# tests/

Automated tests for the pipeline. Right now this only covers
`src/router.py`, and that's a deliberate starting point, not an
oversight — routing is the one file with zero external dependencies
(no Ollama, no API keys, no downloaded data), so it's the only piece
that can be tested instantly, anywhere, without any setup.

## Running the tests

```
pip install pytest
pytest tests/
```

From the project root (not from inside this folder) — `pytest` finds
tests automatically as long as you run it from somewhere above this
directory.

## What's covered

| Test | Checks |
|---|---|
| `test_high_confidence_resolves_locally` | A high confidence score routes to `LOCAL` |
| `test_low_confidence_escalates` | A low confidence score routes to `ESCALATE` |
| `test_exact_threshold_resolves_locally` | A score exactly at the threshold counts as `LOCAL`, not `ESCALATE` — confirms the `>=` boundary behaves as intended |
| `test_custom_threshold_override` | Passing a custom threshold actually changes the routing decision |

## What's not covered yet, and why

`tier1.py` and `tier2_escalate.py` aren't tested here because they make
real network calls (to Ollama, to the LLM API) — testing them properly
would need mocking those calls out, which is a reasonable next step if
you want to expand this folder, but wasn't required to get a working
pipeline running. Worth mentioning as a limitation in your write-up
rather than leaving unstated.
