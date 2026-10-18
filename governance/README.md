# governance/

The evidence base for taking this prototype toward a governed pilot. It is organised the way a model-risk or compliance reviewer
reads: what the system is, what data goes where, what controls exist, what is missing, and what must be true before real customer
data is used.

**Status in one sentence: a research prototype, run on synthetic data only, not approved for customer data.**

This folder is evidence, not a certificate. Nothing here says the system is compliant, secure or approved. Only your risk,
compliance and privacy functions can decide that, after independent validation. It is not legal advice.

| File | What it holds |
|---|---|
| [`MODEL_INVENTORY.md`](MODEL_INVENTORY.md) | Every model and component: what it is, where it runs, what it sees, and its validation status |
| [`DATA_FLOW.md`](DATA_FLOW.md) | What data leaves your computer at each step, what is filtered, and how much reaches the external provider |
| [`CONTROLS.md`](CONTROLS.md) | A controls matrix: each control, its status (implemented, partial, not built) and where the evidence is |
| [`GAP_REGISTER.md`](GAP_REGISTER.md) | Every known gap with severity, closing action, and whether it must close before real data |

## How the files map to the rules you are likely asked about

These summaries come from secondary sources. Read the primary texts, and ask your compliance team which apply to your institution.

| Rule | What it asks for | Where this folder answers it | Status |
|---|---|---|---|
| UK PRA SS1/23, model risk management (in force since 17 May 2024; AI/ML models are in scope; the firm stays accountable for vendor models) | An inventory with risk tiering, governance, documented development and testing, independent validation, mitigants | `MODEL_INVENTORY.md`, `CONTROLS.md`, `GAP_REGISTER.md` | Inventory drafted. Tiers are proposed, not assigned. No independent validation |
| EU AI Act, Article 50 (chatbot disclosure, enforceable from 2 August 2026) | Tell people they are interacting with an AI system | `CONTROLS.md` C16 | Not built: there is no customer-facing front end |
| EU AI Act, high-risk obligations (Annex III, deferred to 2 December 2027) | Only if the system is used for an Annex III purpose such as scoring creditworthiness | `GAP_REGISTER.md` G14 | This assistant is informational. Confirm that it stays so |
| DORA, Article 28 (EU financial entities) | A register of third-party ICT arrangements, contract provisions, and tested exit strategies | `DATA_FLOW.md`, `GAP_REGISTER.md` G12 | Data flow documented. No exit plan |
| UK / EU data protection law | A data protection impact assessment, minimisation, processor terms | `GAP_REGISTER.md` G1 to G5 | Not assessed here. Needed before any real data |

## Keeping it true

Update these files whenever a model, a prompt, a threshold, the filter or a data flow changes. The router version label, the filter
version (`pii_filter` column) and the 45 automated tests are the change evidence the code already produces.
