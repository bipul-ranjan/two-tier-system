# Controls matrix

Each control, what it is for, whether it exists, and where the evidence is. **Status values are literal:** *Implemented* means the code
does it and a test or log shows it. *Partial* means some of it exists. *Not built* means it does not exist yet and is in the gap register.

| ID | Control | Objective | Status | Evidence |
|---|---|---|---|---|
| C1 | Personal-data filter on everything that leaves or is stored | Stop patterned identifiers reaching Claude, the cache lookup, the examiner or the log | **Implemented** (`regex-v3`), limited to patterned identifiers | `src/privacy.py`; `tests/test_privacy.py` (19 tests); `pii_*` log columns |
| C2 | Measurement of the filter | Know what it catches and misses | **Partial**: measured on generated cases only | `scripts/evaluate_privacy_filter.py`; `DATA_FLOW.md` |
| C3 | Router decision recorded on every row | Make each keep-or-escalate decision auditable | **Implemented** | Columns `router`, `router_predicted_quality`, `router_cutoff`, `decision` |
| C4 | Cache decision recorded on every hit | Let a reviewer check each re-used answer | **Implemented** | Columns `cache_hit`, `cache_similarity`, `cache_matched_query`, decision `CACHE` |
| C5 | Cache safety gates | Stop a near-match with opposite meaning or a different request being re-used | **Implemented**: similarity at least 0.95, negation check, category and intent check | `src/semantic_cache.py`. The rate of wrong matches is **not measured** |
| C6 | Cache scoped per customer | Stop one customer's answer being served to another | **Not built** | `GAP_REGISTER.md` G4 |
| C7 | Expiry and invalidation of cached answers | Stop stale policy answers being re-used | **Not built** | G4 |
| C8 | Quality-gated cache admission | Stop a weak Claude answer being remembered | **Not built** | G4 |
| C9 | Hand-off of exception cases to a human | Keep fraud, bereavement and hardship out of automated answers | **Not built**: exception questions escalate to Claude, not to a person | G7 |
| C10 | Independent model validation | An effective challenge by people who did not build it | **Not done** | G6 |
| C11 | Human check of Claude's quality marks | Test the assumption every quality figure rests on | **Not done** | G6 |
| C12 | Third-party exit plan for the Claude API | Continue if the provider is unavailable or withdrawn | **Not done** (the router and cache do not depend on the provider's model, which helps) | G12 |
| C13 | Data flow documented | Show where text goes and what protects it | **Implemented** | `DATA_FLOW.md` |
| C14 | Access control on logs and the dashboard | Limit who can read stored text | **Not built** | G5 |
| C15 | Log retention and deletion rules | Keep text no longer than needed | **Not built**: a backup script exists for redaction only | G5 |
| C16 | Disclosure that the user is talking to AI | EU AI Act Article 50 (enforceable from 2 August 2026) | **Not built**: no customer-facing front end exists | G9 |
| C17 | Monitoring with thresholds and alerts | Detect drift in quality, escalation share or cost | **Partial**: the dashboard shows trends; there are no thresholds or alerts | Dashboard; G10 |
| C18 | Change control | Know which version produced each answer | **Partial**: router version label, filter version, model names, 158 tests; no formal approval step | `models/router/*_info.json`; `pii_filter` column |
| C19 | Reproducibility | Repeat a result | **Partial**: optional fixed seed; package versions are not locked | `SEED` in `src/pipeline.py` |
| C20 | Adversarial testing (prompt injection, data leakage) | Find ways the system can be made to misbehave | **Not done** | G8 |
| C21 | Minimise what the examiner receives | Reduce exposure to the provider | **Not built**: the harness marks every row (sampling or an in-house judge would reduce it) | `DATA_FLOW.md`; G2 |
| C22 | Secrets handling | Keep the API key out of code and Git | **Implemented**: the key is read from an environment variable | `src/tier2_escalate.py`; `.gitignore` |
| C23 | Test-question source recorded on every row | Tell a run on fresh questions from one on the fixed pool | **Implemented** | `question_source` log column (`synthetic-pool`, a file name, or `generated:<model>:seed<N>`); `--fresh`, `--questions` and `--seed` options; `tests/test_question_generator.py` |
| C24 | The label judge is measured before it is trusted | Stop a weak small-model judge deciding which generated test questions to keep | **Implemented** (the tool); **no measurement on a real model has been recorded yet** | `scripts/evaluate_label_judge.py`; `src/label_judge.py`; `tests/test_label_judge.py` |
