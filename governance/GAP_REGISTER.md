# Gap register

Every known gap, honestly rated. **Severity is for a production use on real customer data.** For the prototype on synthetic data the
gaps are accepted, which is why the scope statement on page one of any pack must say so. Owners are to be assigned.

"Before real data" means the gap must be closed before any real customer text enters the system.

| ID | Gap | Why it matters | Severity | Before real data? | Closing action |
|---|---|---|---|---|---|
| G1 | The filter does not catch free-text names, street addresses, employer names, or identifiers in formats it does not know (0% on free-text names and addresses in our generated test) | Identifying text can reach Claude and the logs | High | **Yes** | Add a trained entity-recognition model; set a recall target with the privacy team; test on real, labelled samples |
| G2 | The examiner (quality marking) receives a filtered copy of every question and answer, including local ones | Exposure to the provider is 100% of rows in this harness | High | **Yes** | Mark a sample, or on redacted text only, or with a judge inside your boundary |
| G3 | Log rows written before the filter hold the original text | Raw identifiers sit in files that can be copied | Medium | **Yes** | Run `scripts/redact_history.py --apply`, then retrain the router |
| G4 | The cache has no per-customer scoping, no expiry, and no quality gate | One customer's answer could be served to another; stale or weak answers are repeated | High | **Yes** | Scope entries per customer, add an age limit and a clearing step on policy change, admit only answers marked 4 or more |
| G5 | No access control, retention or deletion rules for logs and the dashboard | Stored text is readable by anyone with the files | High | **Yes** | Add authentication, role-based access, a retention period and deletion |
| G6 | No independent validation, and Claude's marks are unchecked by people | Every quality figure rests on an untested examiner | High | **Yes** | Independent validation; about 200 answers marked by a banking professional, compared with Claude's marks |
| G7 | Exception cases (fraud, bereavement, hardship, complaints) are not handed to a human | An automated answer to a vulnerable customer carries conduct risk | High | **Yes** | Route these types to a specialist, with the draft attached |
| G8 | No adversarial testing | Prompt injection or leakage could expose data or alter answers | Medium to High | **Yes** | Run a red-team set against the helpers, router and cache |
| G9 | No disclosure to users that they are talking to AI | Required under the EU AI Act, Article 50, for EU users | Medium | Yes, for EU users | Add a disclosure in the customer front end |
| G10 | No monitoring thresholds or alerts; server cost not measured | Drift and cost overruns go unnoticed | Medium | Yes | Set thresholds on quality, escalation share and cost; measure server cost for a break-even figure |
| G11 | All evidence is on synthetic, template-generated questions | Results may not carry over to real phrasing | High | **Yes** | Re-test on human-written questions in a controlled pilot |
| G12 | No written exit plan for the Claude API | Third-party risk: continuity if the provider changes terms or is unavailable | Medium | Yes | Write and test an exit plan; document the fallback to local answers or a human queue |
| G13 | The filter's recall is measured on generated cases only | The 100% figure is a ceiling, not an estimate for real text | Medium | **Yes** | Measure on a real, labelled sample |
| G14 | The use case could drift into an EU AI Act high-risk purpose (for example creditworthiness) | High-risk obligations would then apply (deferred to 2 December 2027) | Low now | Confirm | State the intended use, and re-assess before any change of purpose |
| G15 | No named model owners, approvals or review cadence | SS1/23 expects accountability and an inventory with owners | Medium | Yes | Assign owners and approval steps in your model risk process |
| G16 | Generated test questions are synthetic, and a small model checks their labels (by choosing among every intent of the business unit, which is stricter than before but still a small model's judgement) | A wrong label would distort per-intent analysis and the cache's third gate; real customers word things differently | Medium | No | Read a sample of every batch; report the label-check pass rate and the confusion report; read the rejected candidates (`--save-rejected`); measure the judge on known-label questions first (`scripts/evaluate_label_judge.py`) and pick the best way of asking (`--label-method`); use a larger, separate judge (`--label-model`); add human-written questions in a controlled pilot (see G11) |

## The order I would close them in

1. **G3** is quick and reduces exposure immediately.
2. **G1, G13 and G2** together decide how much text can leave the building.
3. **G4 and G7** decide whether customers can be harmed by a wrong or misdirected answer.
4. **G6** makes the quality evidence credible to a reviewer.
5. **G5, G8, G9, G10, G12 and G15** are the governance and operational steps around those.
