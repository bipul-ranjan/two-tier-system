# dashboard/

A live dashboard for the two-tier system, and nothing else. It reads the pipeline's own logs from `results/logs/`, so there is nothing
to configure and nothing to keep in sync. A small FastAPI backend (`backend/main.py`) serves the data; a React page (`frontend/`)
shows it. The built page is committed in `frontend/dist`, so **Node is not needed** just to use it.

## Run it

Once, inside the project's virtual environment:

```
pip install -r dashboard/backend/requirements.txt
```

Then, from the project root (the folder that contains `src/` and `dashboard/`):

```
python -m uvicorn dashboard.backend.main:app --port 8000
```

Open **http://localhost:8000** and leave that terminal open (closing it stops the dashboard; press `Ctrl+C` to stop it yourself). The page
refreshes every five seconds, so you can leave it open while `python -m src.pipeline` runs. Port 8000 busy? Add `--port 8010` and open that port.
To read logs from another folder, set the environment variable `TWO_TIER_LOGS_DIR` before starting it.

## What it shows

**Left: the run list.** Every run, newest first, with its size, share answered locally and average confidence. Small tags mark a run
that is the `latest`, or the first after a change: **new models** (a different Tier 1 model), **new router** (a retrained router, or a switch
between the threshold rule and the learned router). The page follows the newest run until you pick another. "Overview across all runs" shows
every run together.

**A run's page**

* **A plain-language summary**: how many questions stayed local, were served from the cache, or went to Claude; the router in use and its
  cut-point (or pass marks); average confidence; Claude's own confidence; answer quality; median times; Claude spend.
* **Where each answer landed**: one dot per question. For a run decided by the **learned router** the dots are placed by the quality the router
  *predicted* for Tier 1's answer, against its **cut-point** line (left of the line is escalated). For an older run decided by the **threshold rule** they are
  placed by confidence against the threshold line(s). Diamonds are exception questions.
* **Answer quality**, in the four sources that actually produced answers: the payments model and the retail-bank model (when they answered
  locally), Claude (fresh answers only) and the cache (re-served answers), each with its five marks.
* **Time per question** by path, **by business unit**, **normal and exception questions**, and the **latest questions** of the run with their
  confidence, predicted quality, outcome and time.

**Across runs.** Trend charts of the share answered locally / from the cache / by Claude (per business unit), average confidence, answer quality
(per source), and median time. Dashed markers show where something changed: the local models (**new model**), the older router's pass mark,
Claude's prompt wording (**new Claude prompt**), and the router (**new router**). This is the view for before-and-after comparisons.

Logs from older versions of the pipeline still display, with missing pieces left blank: no `router` column means the threshold rule decided,
and an old cache hit recorded as `ESCALATE` with `cache_hit` true is shown as a cache hit.

## Where the data comes from

| File in `results/logs/` | Used for |
|---|---|
| `results_history.csv` | Every run (preferred) |
| `results_log_combined.csv` | The latest run only, if there is no history file yet |

`python -m src.evaluate` does not need to be run for the dashboard. Endpoints, if you want the raw data: `GET /api/health`, `GET /api/runs`,
`GET /api/run?run_id=...`.

## Comparing runs fairly

1. Use the **same questions** for every run you compare: set `SEED` to a number near the top of `src/pipeline.py`. With `SEED = None` each run draws
   different questions, so differences may only reflect that.
2. Give retrained models **new names** (for example `payment-assistant-v2`) in `src/config.py`. The log records the model per row, and the "new models"
   marker comes from that; if you reuse a name, before and after look identical.
3. To compare the two routers, run `python -m src.pipeline 100` and `python -m src.pipeline 100 --threshold-router` with the same `SEED`.
4. Confidence is how sure the model was of its own wording, not whether the answer is right. A higher confidence after retraining is not by itself
   a better answer, which is why the quality views exist.

## Troubleshooting

| You see | What to do |
|---|---|
| "No results yet" | Run `python -m src.pipeline 10` from the project root |
| "Cannot reach the dashboard backend" | Start the backend with the command above and reload the page |
| "Only the latest run is available" | There is no `results_history.csv` yet. Run `python scripts/backfill_run_id.py` once, then run the pipeline again |
| `Address already in use` | Another program is using port 8000: use `--port 8010` |
| `No module named fastapi` (or `uvicorn`) | The virtual environment is not active, or the install line was skipped. See below |
| The page shows old numbers after an update | Hard-refresh the browser (`Ctrl+F5`) so it loads the new build |

**The "wrong Python" problem.** If `No module named ...` appears even though you installed the packages, your prompt probably has no `(venv)` in front, so the
*system* Python is running instead of the project's. Activate the environment and install again:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned      # Windows PowerShell, one-time for this window
.\venv\Scripts\Activate.ps1                                           # macOS/Linux: source venv/bin/activate
python -m pip install -r dashboard\backend\requirements.txt
```

## Changing the page (optional; needs Node 18 or newer)

```
cd dashboard/frontend
npm install
npm run dev        # live preview at http://localhost:5173, forwarding /api to the backend on port 8000
npm run build      # rebuilds frontend/dist, which the backend serves
```

`frontend/node_modules` is ignored by Git. `frontend/dist` is committed so the dashboard works straight after a clone. **Each build writes
new, uniquely named files into `dist/assets/` and deletes the old ones on your disk; when you commit, delete the old bundle files from Git too**
(`git add -A` does this), or the repository fills with dead copies.
