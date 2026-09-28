# dashboard/

A live dashboard for the two-tier system, and nothing else. It reads the pipeline's own logs from
`results/logs/`, so there is nothing to configure and nothing to keep in sync.

## What it shows

- **Runs** (left): every pipeline run, newest first, with its average confidence. Click a run to look at it.
  The page follows the newest run on its own until you pick another one.
- **A plain-language summary** of the selected run: how many queries stayed local, how many went to Claude,
  average confidence against the threshold, median times, Claude spend.
- **Where each answer landed**: one dot per query, placed by confidence, with the router's threshold as a line.
  Diamonds are exception queries.
- **Across runs**: average confidence, share answered locally, or median time, run by run and per business
  unit. A dashed marker shows where the local models changed. This is the view for before and after retraining.
- **Time per query**, **by business unit**, **normal and exception queries**, and the **latest queries** of the run.

The page refreshes every five seconds, so you can leave it open while `python -m src.pipeline` runs.

## Run it

Once, inside your virtual environment:

```
pip install -r dashboard/backend/requirements.txt
```

Then, from the project root (the folder that contains `src/` and `dashboard/`):

```
python -m uvicorn dashboard.backend.main:app --port 8000
```

Open http://localhost:8000. Node is not needed: the built frontend is in `frontend/dist`.

## Where the data comes from

| File in `results/logs/` | Used for |
|---|---|
| `results_history.csv` | every run (preferred) |
| `results_log_combined.csv` | the latest run only, if there is no history file yet |

`evaluate.py` does not need to be run for the dashboard. Runs recorded by older versions of the pipeline
(no `run_id`, no timestamps, no threshold) still appear, with the missing pieces left blank.
To read logs from another folder, set `TWO_TIER_LOGS_DIR` before starting the backend.

## Comparing runs before and after retraining

1. Use the same queries for every run you compare. Set `SEED` to a number at the top of `src/pipeline.py`;
   with `SEED = None` each run draws different queries, so confidence differences may only reflect that.
2. Give retrained models new names (for example `payment-assistant-v2`) in `src/config.py`. The log records the
   model per row, and the "new models" marker and tag come from that. If you reuse the old names, before and
   after look identical.
3. Confidence is how sure the model was of its own wording, not whether the answer is right. A higher
   confidence after retraining is not by itself proof of a better answer.

## Troubleshooting

| You see | What to do |
|---|---|
| "No results yet" | Run `python -m src.pipeline 10` from the project root. |
| "Cannot reach the dashboard backend" | Start the backend with the command above and reload the page. |
| "Only the latest run is available" | There is no `results_history.csv` yet. Run `python scripts/backfill_run_id.py` once, then run the pipeline again. |
| `Address already in use` | Another program is using port 8000. Start with `--port 8010` and open that port instead. |
| `No module named fastapi` | Run the `pip install` line above in the same virtual environment. |

## Changing the page (optional, needs Node 18 or newer)

```
cd dashboard/frontend
npm install
npm run dev        # http://localhost:5173, forwards /api to the backend on :8000
npm run build      # rebuilds frontend/dist, which the backend serves
```

`dashboard/frontend/node_modules` is ignored by Git. `frontend/dist` is kept, so the dashboard works straight
after a clone.
